import pytest

from cleanab.apps import actual
from cleanab.main import Cleanab  # noqa: F401  (import check: main wires prepare_transactions)

CATEGORIES = [{"id": "c-food", "name": "Food"}, {"id": "c-bills", "name": "Bills"}]
PAYEES = [{"id": "p-rewe", "name": "Rewe"}, {"id": "p-hkk", "name": "hkk"}]


def history_tx(i, payee, category):
    return {"date": f"2023-01-{i:02d}", "payee": payee, "category": category, "notes": "", "imported_id": f"h{i}"}


@pytest.fixture
def app(monkeypatch):
    app = actual.App(
        actual.Config(
            actual_api_url="http://actual.local/",
            actual_api_key="k",
            actual_sync_id="s",
            actual_account_ids=["acc"],
            categorize={"enabled": True},
        )
    )
    txs = [history_tx(i, "p-rewe", "c-food") for i in range(1, 6)] + [
        history_tx(i, "p-hkk", "c-bills") for i in range(6, 11)
    ]
    txs.append({"date": "2024-05-01", "payee": "p-hkk", "category": None, "imported_id": "already"})
    data = {"/categories": CATEGORIES, "/payees": PAYEES, "/accounts/acc/transactions": txs}
    monkeypatch.setattr(app, "_get", lambda path, **params: data[path])
    return app


def new_tx(name, imported_id="new"):
    return {"payee_name": name, "notes": "", "imported_id": imported_id, "_account_id": "acc"}


def test_known_payee_gets_category(app):
    (tx,) = app.prepare_transactions((new_tx("Rewe"),))
    assert tx["category"] == "c-food"


def test_new_payee_only_suggested_by_default(app):
    (tx,) = app.prepare_transactions((new_tx("Totally New Shop"),))
    assert "category" not in tx


def test_new_payee_assigned_when_enabled(app):
    app.config.categorize.assign_new_payees = True
    app.config.categorize.new_payee_min_confidence = 0.0
    (tx,) = app.prepare_transactions((new_tx("Rewe Markt Nord"),))
    assert tx["category"] == "c-food"


def test_existing_transactions_untouched(app):
    (tx,) = app.prepare_transactions((new_tx("Rewe", imported_id="already"),))
    assert "category" not in tx


def test_disabled_by_default():
    app = actual.App(
        actual.Config(actual_api_url="http://a/", actual_api_key="k", actual_sync_id="s", actual_account_ids=["a"])
    )
    txs = (new_tx("Rewe"),)
    assert app.prepare_transactions(txs) is txs


def test_failure_does_not_block_import(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("actual down")

    monkeypatch.setattr(app, "_get", boom)
    txs = (new_tx("Rewe"),)
    assert app.prepare_transactions(txs) == txs
    assert "category" not in txs[0]
