
from decimal import Decimal
import json
from typing import Union

import requests
from pydantic import AnyHttpUrl
from logzero import logger

from ..models import AccountConfig, FintsTransaction
from .base import BaseApp, BaseAppConfig


class ActualAppConfig(BaseAppConfig):
    actual_api_url: AnyHttpUrl
    actual_api_key: str
    actual_sync_id: str
    actual_account_id: str
    actual_encryption_password: Union[str, None] = None


class ActualApp(BaseApp):

    def __init__(self, config: ActualAppConfig) -> None:
        self.config = config

    def __str__(self):
        return "Actual App Connection"

    def create_intermediary(self, transactions: list[dict]) -> str:
        return json.dumps(transactions, indent=2)

    def create_transactions(self, transactions):
        url = self.config.actual_api_url.rstrip('/')
        sync_id = self.config.actual_sync_id
        account_id = self.config.actual_account_id
        headers = {
            "x-api-key": self.config.actual_api_key,
            "accept": "application/json"
        }
        if self.config.actual_encryption_password:
            headers["budget-encryption-password"] = self.config.actual_encryption_password

        # Send transactions in chunks of 100
        new, duplicates = [], []
        for i in range(0, len(transactions), 100):
            chunk = transactions[i:i + 100]

            response = requests.post(
                f"{url}/budgets/{sync_id}/accounts/{account_id}/transactions/import",
                headers=headers,
                json={"transactions": chunk}
            )

            if not response.ok:
                logger.error(f"Failed creating transactions: \n\n{response.text}")
                return new, duplicates

            report = response.json()
            logger.info(f"Received import report:\n{report}")
            new += report.get('added', [])
            duplicates += report.get('updated', [])
        return new, duplicates

    def augment_transaction(
        self, transaction: FintsTransaction, account: AccountConfig
    ):
        payee_name = transaction.applicant_name
        if len(payee_name) > 50:
            payee_name = payee_name[:50]

        return {
            "account": self.config.actual_account_id,
            "date": transaction.date.isoformat(),
            "payee_name": transaction.applicant_name or "Unnamed",
            "imported_payee": transaction.applicant_name or "Unnamed",
            "notes": transaction.purpose,
            "amount": int(Decimal(transaction.amount) / 10),
            "imported_id": transaction.import_id,
        }


Config = ActualAppConfig
App = ActualApp
