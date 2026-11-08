import json
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

import requests
from logzero import logger
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from ..classifier.base import Example
from ..classifier.decider import ModelUnavailable
from ..classifier.factory import build_classifier
from ..classifier.normalize import normalize_text
from ..models import AccountConfig, FintsTransaction
from .base import BaseApp, BaseAppConfig

REQUEST_TIMEOUT = 30
# Actual matches imported transactions to existing ones with the same amount up to this many days apart.
MATCH_WINDOW = timedelta(days=7)


class CategorizeConfig(BaseModel):
    """Suggest categories for new transactions, learned from the categorized ones in the configured Actual accounts."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: Literal["light", "full"] = "light"
    history_since: date = date(2020, 1, 1)
    history_until: date | None = None
    min_confidence: float = Field(0.5, ge=0, le=1)
    assign_new_payees: bool = False
    new_payee_min_confidence: float = Field(0.8, ge=0, le=1)
    descriptions: dict[str, str] = {}

    @model_validator(mode="after")
    def _check(self):
        if self.history_until and self.history_until < self.history_since:
            raise ValueError("history_until must not be before history_since")
        if self.enabled and self.descriptions and self.model != "full":
            raise ValueError("descriptions are only used by model: full")
        return self


class ActualAppConfig(BaseAppConfig):
    # Unknown keys only warn, so configs with stray keys keep working; `categorize` itself is strict.
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_unknown_keys(cls, data):
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                logger.warning(f"Ignoring unknown keys in the Actual app config: {', '.join(unknown)}")
        return data

    actual_api_url: HttpUrl
    actual_api_key: str
    actual_sync_id: str
    actual_account_ids: list[str]
    actual_encryption_password: str | None = None
    categorize: CategorizeConfig = CategorizeConfig()

    @field_validator("categorize", mode="before")
    @classmethod
    def _empty_categorize(cls, value):
        return {} if value is None else value  # a bare `categorize:` with everything commented out


class ActualApp(BaseApp):
    is_written_account = False

    def __init__(self, config: ActualAppConfig) -> None:
        self.config = config

    def __str__(self):
        return "Actual App Connection"

    def create_intermediary(self, transactions: tuple) -> str:
        return json.dumps(transactions, indent=2)

    def _headers(self) -> dict:
        headers = {"x-api-key": self.config.actual_api_key, "accept": "application/json"}
        if self.config.actual_encryption_password:
            headers["budget-encryption-password"] = self.config.actual_encryption_password
        return headers

    def _get(self, path: str, **params):
        url = f"{str(self.config.actual_api_url).rstrip('/')}/budgets/{self.config.actual_sync_id}{path}"
        response = requests.get(url, headers=self._headers(), params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["data"]

    def _transactions(self, account_id: str, since: date) -> list[dict]:
        return self._get(f"/accounts/{account_id}/transactions", since_date=since.isoformat())

    def _category_labels(self) -> tuple[dict[str, str], set[str]]:
        """Return (category id -> label, labels of hidden categories).

        Visible categories keep their name unless several visible ones share it (in different groups); those, and
        hidden categories whose name is taken, get the id appended.
        """
        category_list = self._get("/categories")
        hidden_groups = self._hidden_group_ids()
        hidden_ids = {c["id"] for c in category_list if c.get("hidden") or c.get("group_id") in hidden_groups}
        visible = Counter(c["name"] for c in category_list if c["id"] not in hidden_ids)
        hidden = Counter(c["name"] for c in category_list if c["id"] in hidden_ids)

        def unique(c) -> bool:
            if c["id"] in hidden_ids:
                return visible[c["name"]] == 0 and hidden[c["name"]] == 1
            return visible[c["name"]] == 1

        labels = {c["id"]: c["name"] if unique(c) else f"{c['name']} ({c['id']})" for c in category_list}
        return labels, {labels[i] for i in hidden_ids}

    def _hidden_group_ids(self) -> set[str]:
        """Ids of hidden category groups: all categories in them count as hidden."""
        return {g["id"] for g in self._get("/categorygroups") if g.get("hidden")}

    def _budget_account_ids(self) -> list[str]:
        """The configured accounts that are on budget; Actual does not use categories in off-budget accounts."""
        accounts = {a["id"]: a for a in self._get("/accounts")}
        configured = self.config.actual_account_ids
        if missing := [a for a in configured if a not in accounts]:
            logger.warning(f"Accounts not found in Actual, not categorized: {', '.join(missing)}")
        if offbudget := [a for a in configured if accounts.get(a, {}).get("offbudget")]:
            logger.info(f"Not categorizing off-budget accounts: {', '.join(offbudget)}")
        return [a for a in configured if a in accounts and not accounts[a].get("offbudget")]

    def _load_history(
        self, account_ids: list[str], labels: dict[str, str], since: date, until: date | None
    ) -> tuple[list[Example], set[str]]:
        """Return (categorized transactions in Actual oldest first as training examples, normalized transfer payees).

        Transfers are not learned from; their payees (from `since` up to today, ignoring `until`) are returned so they
        are never assigned a category, since Actual could match the import to a transfer and fill in its category.
        """
        payees = {p["id"]: p["name"] for p in self._get("/payees")}
        history: list[Example] = []
        transfer_payees: set[str] = set()
        for account_id in account_ids:
            for tx in self._transactions(account_id, since):
                # The bank's payee name, as sent by cleanab, so it matches new transactions even when the
                # payee was renamed in Actual; manually entered transactions only have the Actual name.
                payee = tx.get("imported_payee") or payees.get(tx.get("payee"), "") or ""
                if tx.get("transfer_id"):
                    transfer_payees.add(normalize_text(payee))
                    continue
                category = labels.get(tx.get("category")) if tx.get("category") else None
                if not category or tx.get("subtransactions") or (until and tx["date"] > until.isoformat()):
                    continue
                history.append(
                    Example(
                        payee=payee,
                        purpose=tx.get("notes") or "",
                        category=category,
                        date=date.fromisoformat(tx["date"]),
                    )
                )
        transfer_payees.discard("")
        return sorted(history, key=lambda e: e.date), transfer_payees

    def _existing(self, account_ids: list[str], transactions: tuple) -> tuple[set[str], dict[tuple, list[date]]]:
        """What is already in Actual around the incoming transactions, regardless of history dates.

        Returns the imported_ids, and (account, amount) -> dates of transfers and splits entered by hand: Actual may
        match an import to those (same amount, a few days apart) and would then fill in their empty category.
        """
        dates = [date.fromisoformat(t["date"]) for t in transactions if t.get("date")]
        if not dates:
            return set(), {}
        since = min(dates) - MATCH_WINDOW
        imported_ids: set[str] = set()
        manual: dict[tuple, list[date]] = {}
        for account_id in account_ids:
            for tx in self._transactions(account_id, since):
                day = date.fromisoformat(tx["date"])
                if tx.get("imported_id"):
                    imported_ids.add(tx["imported_id"])
                elif tx.get("transfer_id") or tx.get("subtransactions") or tx.get("is_parent"):
                    manual.setdefault((account_id, tx.get("amount")), []).append(day)
                # Actual may also match a single part of a split (also of an imported one: its parts have no
                # imported_id), e.g. a transfer inside it.
                for child in tx.get("subtransactions") or []:
                    if not child.get("imported_id"):
                        manual.setdefault((account_id, child.get("amount")), []).append(day)
        return imported_ids, manual

    def prepare_transactions(self, transactions: tuple) -> tuple:
        settings = self.config.categorize
        if not settings.enabled:
            return transactions
        try:
            try:
                return self._categorize(transactions, settings.model)
            except ModelUnavailable as exc:
                # e.g. the `models` group is not installed or the model download failed: still use the light model.
                logger.warning(f"The full model is unavailable ({exc}), using the light model instead")
                return self._categorize(transactions, "light")
        except Exception:
            # Categories are a convenience: never block the import because of them.
            logger.exception("Categorization failed, importing without categories")
            return transactions

    def _categorize(self, transactions: tuple, model: str) -> tuple:
        settings = self.config.categorize
        account_ids = self._budget_account_ids()
        if not account_ids:
            return transactions
        labels, hidden = self._category_labels()
        history, transfer_payees = self._load_history(
            account_ids, labels, settings.history_since, settings.history_until
        )
        if not history:
            logger.warning("No categorized transactions found in Actual, skipping categorization")
            return transactions
        learned = {e.category for e in history}
        categories = sorted(learned - hidden)  # hidden categories are never offered
        if not categories:
            logger.warning("Only hidden categories in the history, skipping categorization")
            return transactions

        if model == "full":
            described = set(settings.descriptions)
            if unknown := sorted(described - set(labels.values())):
                logger.warning(f"Descriptions for unknown categories are ignored: {', '.join(unknown)}")
            if unhidden := sorted(described & hidden):
                logger.warning(f"Hidden categories are never suggested: {', '.join(unhidden)}")
            if unused := sorted((described & set(labels.values())) - learned - hidden):
                logger.warning(f"Categories without categorized transactions cannot be suggested: {', '.join(unused)}")

        existing, manual = self._existing(account_ids, transactions)
        category_ids = {label: cid for cid, label in labels.items()}
        classifier = build_classifier(model, settings.descriptions)
        classifier.fit(history)

        # Decide everything first and apply at the end, so a failure halfway leaves no transaction categorized.
        assignments: list[tuple[dict, str]] = []
        messages: list[str] = []
        suggested = 0
        skipped: Counter[str] = Counter()
        for transaction in transactions:
            if transaction.get("_account_id") not in account_ids:
                skipped["in an off-budget or unknown account"] += 1
                continue  # off-budget (or unknown) account: no categories there
            if transaction.get("imported_id") in existing:
                skipped["already in Actual"] += 1
                continue  # already in Actual: never touch what is there
            if self._may_match_manual(transaction, manual):
                skipped["possibly matching a manual transfer or split"] += 1
                continue  # Actual may match this to a transfer or split entered by hand, which gets no category
            payee = transaction.get("imported_payee") or transaction["payee_name"]
            if normalize_text(payee) in transfer_payees:
                skipped["from a transfer payee"] += 1
                continue  # a transfer payee: Actual may match this to a transfer, which gets no category
            example = Example(payee=payee, purpose=transaction.get("notes") or "")
            known = classifier.knows(example)
            past = classifier.history_prediction(example, categories) if known else None
            if past is not None and past.category is None:
                skipped["whose past categories are all hidden"] += 1
                continue  # all its past categories are hidden
            prediction = classifier.predict(example, categories)
            if prediction.category is None:
                skipped["without a prediction"] += 1
                continue
            if past is not None:
                # A payee seen before only gets a category it already had. If the model leans elsewhere (e.g. because
                # of the purpose text), or is less sure than the payee's history, use the payee's majority category.
                if prediction.category not in (past.scores or {}) or prediction.confidence < past.confidence:
                    prediction = past
                allowed = True
                threshold = settings.min_confidence
            else:
                allowed = settings.assign_new_payees
                threshold = settings.new_payee_min_confidence
            if allowed and prediction.confidence >= threshold:
                assignments.append((transaction, category_ids[prediction.category]))
                messages.append(f"Category {prediction.category} ({prediction.confidence:.2f}): {example.payee}")
                continue
            suggested += 1
            messages.append(
                f"Suggested category {prediction.category} ({prediction.confidence:.2f}), not assigned: {example.payee}"
            )
        # Logged only now, so a run that fails halfway (and is retried with the light model) logs nothing.
        for message in messages:
            logger.info(message)
        for transaction, category_id in assignments:
            transaction["category"] = category_id
        assigned = len(assignments)
        summary = f"Categories: {assigned} assigned, {suggested} only suggested"
        if skipped:
            summary += "; skipped " + ", ".join(f"{count} {reason}" for reason, count in skipped.items())
        logger.info(summary)
        return transactions

    @staticmethod
    def _may_match_manual(transaction: dict, manual: dict[tuple, list[date]]) -> bool:
        if not transaction.get("date"):
            return False
        day = date.fromisoformat(transaction["date"])
        candidates = manual.get((transaction.get("_account_id"), transaction.get("amount")), [])
        return any(abs(day - other) <= MATCH_WINDOW for other in candidates)

    def create_transactions(self, transactions: list[dict]) -> tuple[list, list]:
        """Create transactions in Actual.

        Args:
            transactions (list[dict]): Transactions to create in Actual.

        Returns:
            tuple[list, list]: A tuple containing lists of new and duplicate transactions.
        """
        url = str(self.config.actual_api_url).rstrip("/")
        sync_id = self.config.actual_sync_id
        headers = self._headers()

        # Each transaction has a key _account_id specifying the actual account id.
        # We need to split the transactions by account id to assign them correctly.
        transactions_by_account = {}
        transaction: dict
        for transaction in transactions:
            account_id = transaction.pop("_account_id")
            if account_id not in self.config.actual_account_ids:
                logger.warning(f"Skipping transaction for unknown account id {account_id}")
                continue
            transactions_by_account.setdefault(account_id, []).append(transaction)

        # Send transactions in chunks of 100
        new, duplicates = [], []
        for account_id, transactions in transactions_by_account.items():
            for i in range(0, len(transactions), 100):
                chunk = transactions[i : i + 100]

                response = requests.post(
                    f"{url}/budgets/{sync_id}/accounts/{account_id}/transactions/import",
                    headers=headers,
                    json={"transactions": chunk},
                    timeout=REQUEST_TIMEOUT,
                )

                if not response.ok:
                    logger.error(f"Failed creating transactions for account {account_id}: \n\n{response.text}")
                    break

                report = response.json().get("data", {})
                logger.info(f"Received import report:\n{report}")
                new += report.get("added", [])
                duplicates += report.get("updated", [])
        return new, duplicates

    def augment_transaction(self, transaction: FintsTransaction, account: AccountConfig):
        if not self.is_written_account:
            logger.debug(f"Writing transactions to account {account}")
            self.is_written_account = True
        payee_name = (transaction.applicant_name or "Unnamed")[:50]

        return {
            "_account_id": account.per_app_id,
            "date": transaction.date.isoformat(),
            "payee_name": payee_name,
            "imported_payee": transaction.applicant_name or "Unnamed",
            "notes": transaction.purpose,
            "amount": int(Decimal(transaction.amount) / 10),
            "imported_id": transaction.import_id,
        }


Config = ActualAppConfig
App = ActualApp
