import csv
import io
import json

import pytest

from cleanab.apps import actual, firefly_iii_fidi


class FakeResponse:
    def __init__(self, ok=True, payload=None, text=""):
        self.ok = ok
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class Calls(list):
    def __init__(self):
        super().__init__()
        self.responses = []


@pytest.fixture
def actual_app():
    return actual.App(
        actual.Config(
            actual_api_url="http://actual.local:5007/",
            actual_api_key="key",
            actual_sync_id="sync",
            actual_account_ids=["acc-1", "acc-2"],
            actual_encryption_password="pw",
        )
    )


@pytest.fixture
def post_calls(monkeypatch):
    calls = Calls()
    responses = calls.responses

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0) if responses else FakeResponse(payload={"data": {"added": ["a"], "updated": []}})

    monkeypatch.setattr(actual.requests, "post", fake_post)
    monkeypatch.setattr(firefly_iii_fidi.requests, "post", fake_post)
    return calls


# --- Actual ---


def test_actual_augment_transaction(actual_app, transaction, account):
    result = actual_app.augment_transaction(transaction, account)
    assert result == {
        "_account_id": "acc-1",
        "date": "2024-01-02",
        "payee_name": "Some Shop",
        "imported_payee": "Some Shop",
        "notes": "Thanks",
        "amount": -1234,  # cents
        "imported_id": "abc123",
    }


def test_actual_payee_is_truncated(actual_app, transaction, account):
    transaction.applicant_name = "x" * 80
    assert len(actual_app.augment_transaction(transaction, account)["payee_name"]) == 50


def test_actual_create_transactions_splits_by_account(actual_app, post_calls):
    txs = [{"_account_id": "acc-1", "n": 1}, {"_account_id": "acc-2", "n": 2}, {"_account_id": "unknown", "n": 3}]
    new, duplicates = actual_app.create_transactions(txs)
    assert [url for url, _ in post_calls] == [
        "http://actual.local:5007/budgets/sync/accounts/acc-1/transactions/import",
        "http://actual.local:5007/budgets/sync/accounts/acc-2/transactions/import",
    ]
    assert post_calls[0][1]["json"] == {"transactions": [{"n": 1}]}
    assert post_calls[0][1]["headers"]["x-api-key"] == "key"
    assert post_calls[0][1]["headers"]["budget-encryption-password"] == "pw"
    assert new == ["a", "a"]
    assert duplicates == []


def test_actual_chunks_of_100(actual_app, post_calls):
    actual_app.create_transactions([{"_account_id": "acc-1", "n": i} for i in range(250)])
    assert [len(kw["json"]["transactions"]) for _, kw in post_calls] == [100, 100, 50]


def test_actual_failure_does_not_abort_other_accounts(actual_app, post_calls):
    post_calls.responses.append(FakeResponse(ok=False, text="boom"))
    new, _ = actual_app.create_transactions([{"_account_id": "acc-1"}, {"_account_id": "acc-2"}])
    assert len(post_calls) == 2
    assert new == ["a"]


def test_actual_create_intermediary(actual_app):
    assert json.loads(actual_app.create_intermediary([{"a": 1}])) == [{"a": 1}]


# --- Firefly III ---


@pytest.fixture
def firefly_app(post_calls):
    return firefly_iii_fidi.App(
        firefly_iii_fidi.Config(
            fidi_url="http://fidi.local/",
            default_account_id=7,
            auto_import_secret="secret",
            personal_access_token="token",
        )
    )


def test_firefly_augment_transaction(firefly_app, transaction, account):
    result = firefly_app.augment_transaction(transaction, account)
    assert result["amount"] == "-12.34"
    assert result["account-name"] == "Main"
    assert result["external-id"] == "abc123"


def test_firefly_intermediary_is_csv(firefly_app, transaction, account):
    row = firefly_app.augment_transaction(transaction, account)
    rows = list(csv.DictReader(io.StringIO(firefly_app.create_intermediary([row]))))
    assert rows == [row]


def test_firefly_create_transactions(firefly_app, post_calls):
    post_calls.responses.append(FakeResponse(text="ok\n\n done "))
    assert firefly_app.create_transactions([]) == ([], [])
    url, kwargs = post_calls[0]
    assert url == "http://fidi.local/autoupload"
    assert kwargs["params"] == {"secret": "secret"}
    assert kwargs["headers"]["Authorization"] == "Bearer token"
    assert json.loads(kwargs["files"]["json"])["default_account"] == 7


def test_firefly_failure_returns_empty(firefly_app, post_calls):
    post_calls.responses.append(FakeResponse(ok=False, text="nope"))
    assert firefly_app.create_transactions([]) == ([], [])
