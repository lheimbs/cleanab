import json
from datetime import date
from decimal import Decimal
from typing import Literal

import requests
from logzero import logger
from pydantic import BaseModel, ConfigDict, HttpUrl

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
    min_confidence: float = 0.5
    assign_new_payees: bool = False
    new_payee_min_confidence: float = 0.8
    descriptions: dict[str, str] = {}


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

    def _load_reference_data(self, since: date, until: date | None):
        """Return (category name -> id, history examples, imported_ids already in Actual)."""
        category_list = self._get("/categories")
        category_ids = {c["name"]: c["id"] for c in category_list}
        names = {c["id"]: c["name"] for c in category_list}
        payees = {p["id"]: p["name"] for p in self._get("/payees")}
        history: list[Example] = []
        existing: set[str] = set()
        for account_id in self.config.actual_account_ids:
            params = {"since_date": since.isoformat()}
            if until:
                params["until_date"] = until.isoformat()
            for tx in self._get(f"/accounts/{account_id}/transactions", **params):
                if tx.get("imported_id"):
                    existing.add(tx["imported_id"])
                category = names.get(tx.get("category")) if tx.get("category") else None
                if not category or tx.get("transfer_id") or tx.get("subtransactions"):
                    continue
                history.append(
                    Example(
                        payee=payees.get(tx.get("payee"), "") or tx.get("imported_payee") or "",
                        purpose=tx.get("notes") or "",
                        category=category,
                        date=date.fromisoformat(tx["date"]),
                    )
                )
        return category_ids, sorted(history, key=lambda e: e.date), existing

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
        category_ids, history, existing = self._load_reference_data(settings.history_since, settings.history_until)
        if not history:
            logger.warning("No categorized transactions found in Actual, skipping categorization")
            return transactions
        classifier = build_classifier(settings.model, settings.descriptions)
        classifier.fit(history)
        categories = sorted({e.category for e in history if e.category in category_ids})

        assigned = suggested = 0
        for transaction in transactions:
            if transaction.get("imported_id") in existing:
                continue  # already in Actual: never touch what is there
            example = Example(payee=transaction["payee_name"], purpose=transaction.get("notes", ""))
            prediction = classifier.predict(example, categories)
            if prediction.category is None:
                continue
            known = classifier.knows(example)
            threshold = settings.min_confidence if known else settings.new_payee_min_confidence
            if known or settings.assign_new_payees:
                if prediction.confidence >= threshold:
                    transaction["category"] = category_ids[prediction.category]
                    assigned += 1
                    logger.info(f"Category {prediction.category} ({prediction.confidence:.2f}): {example.payee}")
                    continue
            suggested += 1
            logger.info(
                f"Suggested category {prediction.category} ({prediction.confidence:.2f}), not assigned: {example.payee}"
            )
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
        headers = {
            "x-api-key": self.config.actual_api_key,
            "accept": "application/json",
        }
        if self.config.actual_encryption_password:
            headers["budget-encryption-password"] = self.config.actual_encryption_password

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
