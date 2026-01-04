import json
from decimal import Decimal

import requests
from logzero import logger
from pydantic import HttpUrl

from ..models import AccountConfig, FintsTransaction
from .base import BaseApp, BaseAppConfig


class ActualAppConfig(BaseAppConfig):
    actual_api_url: HttpUrl
    actual_api_key: str
    actual_sync_id: str
    actual_account_ids: list[str]
    actual_encryption_password: str | None = None


class ActualApp(BaseApp):
    def __init__(self, config: ActualAppConfig) -> None:
        self.config = config

    def __str__(self):
        return "Actual App Connection"

    def create_intermediary(self, transactions: tuple) -> str:
        return json.dumps(transactions, indent=2)

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
            headers["budget-encryption-password"] = (
                self.config.actual_encryption_password
            )

        # Each transaction has a key _account_id specifying the actual account id.
        # We need to split the transactions by account id to assign them correctly.
        transactions_by_account = {}
        transaction: dict
        for transaction in transactions:
            account_id = transaction.pop("_account_id")
            if account_id not in self.config.actual_account_ids:
                logger.warning(
                    f"Skipping transaction for unknown account id {account_id}"
                )
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
                )

                if not response.ok:
                    logger.error(f"Failed creating transactions: \n\n{response.text}")
                    return new, duplicates

                report = response.json().get('data', {})
                logger.info(f"Received import report:\n{report}")
                new += report.get("added", [])
                duplicates += report.get("updated", [])
        return new, duplicates
    
    is_written_account = False

    def augment_transaction(
        self, transaction: FintsTransaction, account: AccountConfig
    ):
        if not self.is_written_account:
            logger.debug(f"Writing transactions to account {account}")
            self.is_written_account = True
        payee_name = transaction.applicant_name
        if len(payee_name) > 50:
            payee_name = payee_name[:50]

        return {
            "_account_id": account.per_app_id,
            "date": transaction.date.isoformat(),
            "payee_name": transaction.applicant_name or "Unnamed",
            "imported_payee": transaction.applicant_name or "Unnamed",
            "notes": transaction.purpose,
            "amount": int(Decimal(transaction.amount) / 10),
            "imported_id": transaction.import_id,
        }


Config = ActualAppConfig
App = ActualApp
