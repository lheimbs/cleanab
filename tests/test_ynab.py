from types import SimpleNamespace
from uuid import uuid4

import pytest

from cleanab.apps import ynab5


@pytest.fixture
def ynab_app():
    return ynab5.App(ynab5.Config(access_token="tok", budget_id=uuid4()))


def test_ynab_augment_transaction(ynab_app, transaction, account):
    account.default_cleared = True
    result = ynab_app.augment_transaction(transaction, account)
    assert result.amount == -12340
    assert result.payee_name == "Some Shop"
    assert result.memo == "Thanks"
    assert result.import_id == "abc123"
    assert result.account_id == "acc-1"
    assert result.cleared == "cleared"
    assert result.approved is False


def test_ynab_uncleared_by_default(ynab_app, transaction, account):
    assert ynab_app.augment_transaction(transaction, account).cleared == "uncleared"


def test_ynab_create_transactions(ynab_app, monkeypatch):
    seen = {}

    class FakeApi:
        def __init__(self, client):
            pass

        def create_transaction(self, budget_id, wrapper):
            seen["budget_id"] = budget_id
            return SimpleNamespace(data=SimpleNamespace(transaction_ids=["t1"], duplicate_import_ids=["d1"]))

    monkeypatch.setattr(ynab5, "TransactionsApi", FakeApi)
    assert ynab_app.create_transactions([]) == (["t1"], ["d1"])
    assert seen["budget_id"] == ynab_app._budget_id


def test_ynab_create_intermediary(ynab_app, transaction, account):
    import json

    result = json.loads(ynab_app.create_intermediary([ynab_app.augment_transaction(transaction, account)]))
    assert result[0]["import_id"] == "abc123"
    assert result[0]["amount"] == -12340
