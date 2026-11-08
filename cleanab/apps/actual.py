import json
from datetime import date
from decimal import Decimal
from typing import Literal

import requests
from logzero import logger
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from ..classifier.base import Example
from ..classifier.factory import build_classifier
from ..models import AccountConfig, FintsTransaction
from .base import BaseApp, BaseAppConfig

REQUEST_TIMEOUT = 30


class CategorizeConfig(BaseModel):
    """Suggest categories for new transactions, learned from the ones you already categorized in Actual."""

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
        if self.descriptions and self.model != "full":
            raise ValueError("descriptions are only used by model: full")
        return self


class ActualAppConfig(BaseAppConfig):
    model_config = ConfigDict(extra="forbid")

    actual_api_url: HttpUrl
    actual_api_key: str
    actual_sync_id: str
    actual_account_ids: list[str]
    actual_encryption_password: str | None = None
    categorize: CategorizeConfig = CategorizeConfig()


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

    def _transactions(self, account_id: str, since: date, until: date | None = None) -> list[dict]:
        params = {"since_date": since.isoformat()}
        if until:
            params["until_date"] = until.isoformat()
        return self._get(f"/accounts/{account_id}/transactions", **params)

    def _category_labels(self) -> dict[str, str]:
        """Return category id -> label. Names shared by several categories (in different groups) get the id appended."""
        category_list = self._get("/categories")
        counts: dict[str, int] = {}
        for c in category_list:
            counts[c["name"]] = counts.get(c["name"], 0) + 1
        return {c["id"]: c["name"] if counts[c["name"]] == 1 else f"{c['name']} ({c['id']})" for c in category_list}

    def _load_history(self, labels: dict[str, str], since: date, until: date | None) -> list[Example]:
        """Return the categorized transactions in Actual, oldest first, as training examples."""
        payees = {p["id"]: p["name"] for p in self._get("/payees")}
        history: list[Example] = []
        for account_id in self.config.actual_account_ids:
            for tx in self._transactions(account_id, since, until):
                category = labels.get(tx.get("category")) if tx.get("category") else None
                if not category or tx.get("transfer_id") or tx.get("subtransactions"):
                    continue
                history.append(
                    Example(
                        # The bank's payee name, as sent by cleanab, so it matches new transactions even when the
                        # payee was renamed in Actual; manually entered transactions only have the Actual name.
                        payee=tx.get("imported_payee") or payees.get(tx.get("payee"), "") or "",
                        purpose=tx.get("notes") or "",
                        category=category,
                        date=date.fromisoformat(tx["date"]),
                    )
                )
        return sorted(history, key=lambda e: e.date)

    def _existing_imported_ids(self, transactions: tuple) -> set[str]:
        """imported_ids already in Actual for the period of the incoming transactions, regardless of history dates."""
        dates = [date.fromisoformat(t["date"]) for t in transactions if t.get("date")]
        if not dates:
            return set()
        since = min(dates)
        return {
            tx["imported_id"]
            for account_id in self.config.actual_account_ids
            for tx in self._transactions(account_id, since)
            if tx.get("imported_id")
        }

    def prepare_transactions(self, transactions: tuple) -> tuple:
        settings = self.config.categorize
        if not settings.enabled:
            return transactions
        try:
            return self._categorize(transactions)
        except Exception:
            # Categories are a convenience: never block the import because of them.
            logger.exception("Categorization failed, importing without categories")
            return transactions

    def _categorize(self, transactions: tuple) -> tuple:
        settings = self.config.categorize
        labels = self._category_labels()
        history = self._load_history(labels, settings.history_since, settings.history_until)
        if not history:
            logger.warning("No categorized transactions found in Actual, skipping categorization")
            return transactions
        existing = self._existing_imported_ids(transactions)
        category_ids = {label: cid for cid, label in labels.items()}
        classifier = build_classifier(settings.model, settings.descriptions)
        classifier.fit(history)
        categories = sorted({e.category for e in history})

        unknown = sorted(set(settings.descriptions) - set(labels.values()))
        if unknown:
            logger.warning(f"Descriptions for unknown categories are ignored: {', '.join(unknown)}")
        unused = sorted(set(settings.descriptions) & set(labels.values()) - set(categories))
        if unused:
            logger.warning(f"Categories without categorized transactions cannot be suggested: {', '.join(unused)}")

        # Decide everything first and apply at the end, so a failure halfway leaves no transaction categorized.
        assignments: list[tuple[dict, str]] = []
        suggested = 0
        for transaction in transactions:
            if transaction.get("imported_id") in existing:
                continue  # already in Actual: never touch what is there
            payee = transaction.get("imported_payee") or transaction["payee_name"]
            example = Example(payee=payee, purpose=transaction.get("notes") or "")
            prediction = classifier.predict(example, categories)
            if prediction.category is None:
                continue
            known = classifier.knows(example)
            if known:
                # A payee seen before only gets a category it already had: if the model leans elsewhere (e.g. because
                # of the purpose text), fall back to the payee's majority category.
                past = classifier.history_prediction(example, categories)
                if prediction.category not in (past.scores or {}):
                    prediction = past
                allowed = prediction.category is not None
                threshold = settings.min_confidence
            else:
                allowed = settings.assign_new_payees
                threshold = settings.new_payee_min_confidence
            if allowed and prediction.confidence >= threshold:
                assignments.append((transaction, category_ids[prediction.category]))
                logger.info(f"Category {prediction.category} ({prediction.confidence:.2f}): {example.payee}")
                continue
            suggested += 1
            logger.info(
                f"Suggested category {prediction.category} ({prediction.confidence:.2f}), not assigned: {example.payee}"
            )
        for transaction, category_id in assignments:
            transaction["category"] = category_id
        assigned = len(assignments)
        logger.info(f"Categories: {assigned} assigned, {suggested} only suggested")
        return transactions

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
