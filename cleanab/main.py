from datetime import date, timedelta
from itertools import chain

from logzero import logger

from cleanab.models.config import Config

from .cleaner import FieldCleaner
from .fints import process_fints_account
from .holdings import process_holdings
from .models import AccountConfig
from .models.enums import AccountType
from .transactions import process_transaction

TODAY = date.today()


class Cleanab:

    def __init__(self, *, config: Config, dry_run=False, test=False, verbose=False, save=False):
        self.config = config
        self.dry_run = dry_run
        self.test = test
        self.verbose = verbose
        self.save = save

        if self.test:
            self.dry_run = True
            self.verbose = True

    def setup_app_connections(self):
        self.config.load_apps()

    def setup(self):
        self.setup_app_connections()
        for app in self.config.apps.keys():
            logger.info(f"Loaded App {app}")
        self.accounts = self.config.accounts
        logger.debug("Creating field cleaner instance")
        self.cleaner = FieldCleaner(
            self.config.replacements,
            self.config.finalizer,
        )

        self.earliest = max(
            [
                TODAY - timedelta(days=self.config.timespan.maximum_days),
                self.config.timespan.earliest_date,
            ]
        )
        logger.info(f"Checking back until {self.earliest}")

    def _get_fints_transactions(self, account):
        if self.test and account.has_account_cache:
            raw_transactions = account.read_account_cache()
        else:
            raw_transactions = process_fints_account(
                account,
                earliest=self.earliest,
                latest=TODAY,
                product_id=self.config.cleanab.fints_product_id,
            )
            account.write_account_cache(raw_transactions)
        return raw_transactions

    def processor(self, account):
        logger.info(f"Processing {account}")

        try:
            raw_transactions = self._get_fints_transactions(account)

            if account.account_type == AccountType.HOLDING:
                return []
                processed_transactions = list(
                    process_holdings(
                        account,
                        raw_transactions,
                        self.accounts_api,
                        self.budget_id,
                        min_delta=self.config.cleanab.minimum_holdings_delta,
                    )
                )
            else:
                processed_transactions = list(
                    self.process_account_transactions(
                        raw_transactions,
                        account,
                    )
                )
            logger.info(f"Got {len(processed_transactions)} new transactions")

            if self.save:
                account.write_cleaned_account_cache(processed_transactions)

            return processed_transactions
        except Exception:
            logger.exception("Processing %s failed", account)

            return []

    def run(self):
        processed_transactions = list(zip(
            *chain.from_iterable(self.processor(account) for account in self.accounts)
        ))

        if not processed_transactions:
            logger.warning("No transactions found")
            return

        for i, app_connection in enumerate(self.config.get_apps()):
            transactions = processed_transactions[i]
            if self.dry_run:
                logger.info("Dry-run, not creating transactions")
                if intermediary := app_connection.create_intermediary(
                    transactions
                ):
                    logger.debug(f"{app_connection}: Intermediary:\n\n{intermediary}\n\n")

                return

            logger.info(f"Creating transactions in {app_connection}")
            new, duplicates = app_connection.create_transactions(
                transactions
            )

            logger.info(f"Created {new} new transactions")
            logger.info(f"Saw {duplicates} duplicates")

    def process_account_transactions(self, transactions: list, account: AccountConfig):
        apps = self.config.get_apps()
        for transaction in transactions:
            if not transaction:
                continue

            processed_transaction = process_transaction(transaction, self.cleaner)
            if not processed_transaction:
                continue

            agumented_transaction = [
                app.augment_transaction(processed_transaction, account)
                for app in apps
            ]
            yield agumented_transaction
