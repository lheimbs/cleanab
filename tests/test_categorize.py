from datetime import date

import pytest

from cleanab.apps import actual

CATEGORIES = [{"id": "c-food", "name": "Food"}, {"id": "c-bills", "name": "Bills"}]
PAYEES = [{"id": "p-grocer", "name": "Grocer"}, {"id": "p-insurer", "name": "Insurer"}]


def history_tx(i, payee, category, imported_payee=None):
    return {
        "date": f"2023-01-{i:02d}",
        "payee": payee,
        "imported_payee": imported_payee,
        "category": category,
        "notes": "",
        "imported_id": f"h{i}",
    }


class FakeActual:
    """Stands in for `ActualApp._get`; filters account transactions by the date params like the HTTP API."""

    def __init__(self, transactions, categories=CATEGORIES, payees=PAYEES):
        self.transactions = transactions
        self.data = {"/categories": categories, "/payees": payees}
        self.calls = []

    def __call__(self, path, **params):
        self.calls.append((path, params))
        if path.endswith("/transactions"):
            since = params.get("since_date", "0000")
            until = params.get("until_date", "9999")
            return [t for t in self.transactions if since <= t["date"] <= until]
        return self.data[path]


def make_app(monkeypatch, transactions, **categorize):
    app = actual.App(
        actual.Config(
            actual_api_url="http://actual.local/",
            actual_api_key="k",
            actual_sync_id="s",
            actual_account_ids=["acc"],
            categorize={"enabled": True, **categorize},
        )
    )
    fake = FakeActual(transactions)
    monkeypatch.setattr(app, "_get", fake)
    return app, fake


HISTORY = [history_tx(i, "p-grocer", "c-food") for i in range(1, 6)] + [
    history_tx(i, "p-insurer", "c-bills") for i in range(6, 11)
]


@pytest.fixture
def app(monkeypatch):
    txs = [*HISTORY, {"date": "2024-05-01", "payee": "p-insurer", "category": None, "imported_id": "already"}]
    return make_app(monkeypatch, txs)[0]


def new_tx(name, imported_id="new", day="2024-05-02"):
    return {"date": day, "payee_name": name, "notes": "", "imported_id": imported_id, "_account_id": "acc"}


def test_known_payee_gets_category(app):
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx["category"] == "c-food"


def test_new_payee_only_suggested_by_default(app):
    (tx,) = app.prepare_transactions((new_tx("Totally New Shop"),))
    assert "category" not in tx


def test_new_payee_assigned_when_enabled(app):
    app.config.categorize.assign_new_payees = True
    app.config.categorize.new_payee_min_confidence = 0.0
    (tx,) = app.prepare_transactions((new_tx("Grocer Market North"),))
    assert tx["category"] == "c-food"


def test_known_payee_below_threshold_not_assigned(app):
    app.config.categorize.min_confidence = 1.0
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert "category" not in tx


def test_existing_transactions_untouched(app):
    (tx,) = app.prepare_transactions((new_tx("Grocer", imported_id="already", day="2024-05-01"),))
    assert "category" not in tx


def test_existing_transactions_outside_history_window_untouched(monkeypatch):
    txs = [*HISTORY, {"date": "2024-05-01", "payee": "p-grocer", "category": None, "imported_id": "already"}]
    app, fake = make_app(monkeypatch, txs, history_until=date(2023, 12, 31))
    (tx,) = app.prepare_transactions((new_tx("Grocer", imported_id="already", day="2024-05-01"),))
    assert "category" not in tx
    history_params = [p for path, p in fake.calls if path.endswith("/transactions") and "until_date" in p]
    assert history_params == [{"since_date": "2020-01-01", "until_date": "2023-12-31"}]


def test_history_respects_dates(monkeypatch):
    # Only bills in the window: a grocer transaction outside it must not be learned from.
    txs = [history_tx(1, "p-grocer", "c-food"), *(history_tx(i, "p-insurer", "c-bills") for i in range(6, 11))]
    app, _ = make_app(monkeypatch, txs, history_since=date(2023, 1, 5))
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx.get("category") != "c-food"


def test_renamed_payee_is_still_known(monkeypatch):
    txs = [history_tx(i, "p-grocer", "c-food", imported_payee="GROCER MARKET 123") for i in range(1, 6)] + [
        history_tx(i, "p-insurer", "c-bills") for i in range(6, 11)
    ]
    app, _ = make_app(monkeypatch, txs)
    tx = new_tx("GROCER MARKET 123")
    tx["imported_payee"] = "GROCER MARKET 123"
    (tx,) = app.prepare_transactions((tx,))
    assert tx["category"] == "c-food"


def test_duplicate_category_names_keep_their_ids(monkeypatch):
    categories = [{"id": "c1", "name": "Misc"}, {"id": "c2", "name": "Misc"}, {"id": "c-bills", "name": "Bills"}]
    txs = [history_tx(i, "p-grocer", "c1") for i in range(1, 6)] + [
        history_tx(i, "p-insurer", "c-bills") for i in range(6, 11)
    ]
    app, fake = make_app(monkeypatch, txs)
    fake.data["/categories"] = categories
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx["category"] == "c1"


def test_disabled_by_default():
    app = actual.App(
        actual.Config(actual_api_url="http://a/", actual_api_key="k", actual_sync_id="s", actual_account_ids=["a"])
    )
    txs = (new_tx("Grocer"),)
    assert app.prepare_transactions(txs) is txs


def test_failure_does_not_block_import(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("actual down")

    monkeypatch.setattr(app, "_get", boom)
    txs = (new_tx("Grocer"),)
    assert app.prepare_transactions(txs) == txs
    assert "category" not in txs[0]


def test_get_builds_url_and_headers(monkeypatch):
    app = actual.App(
        actual.Config(
            actual_api_url="http://actual.local/api/",
            actual_api_key="k",
            actual_sync_id="s",
            actual_account_ids=["a"],
            actual_encryption_password="pw",
        )
    )
    seen = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": ["ok"]}

    def fake_get(url, headers, params, timeout):
        seen.update(url=url, headers=headers, params=params)
        return Response()

    monkeypatch.setattr(actual.requests, "get", fake_get)
    assert app._get("/accounts/a/transactions", since_date="2024-01-01") == ["ok"]
    assert seen["url"] == "http://actual.local/api/budgets/s/accounts/a/transactions"
    assert seen["headers"]["x-api-key"] == "k"
    assert seen["headers"]["budget-encryption-password"] == "pw"
    assert seen["params"] == {"since_date": "2024-01-01"}


BASE = {"actual_api_url": "http://a/", "actual_api_key": "k", "actual_sync_id": "s", "actual_account_ids": ["a"]}


def test_misplaced_or_misspelled_keys_are_rejected():
    with pytest.raises(ValueError):
        actual.Config(**BASE, categorise={"enabled": True})
    with pytest.raises(ValueError):
        actual.Config(**BASE, categorize={"enable": True})


@pytest.mark.parametrize(
    "categorize",
    [
        {"min_confidence": 1.5},
        {"new_payee_min_confidence": -0.1},
        {"history_since": "2024-01-01", "history_until": "2023-01-01"},
        {"descriptions": {"Food": "groceries"}},  # only used by model: full
    ],
)
def test_invalid_settings_are_rejected(categorize):
    with pytest.raises(ValueError):
        actual.Config(**BASE, categorize=categorize)
