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
        self.data = {
            "/categories": categories,
            "/payees": payees,
            "/accounts": [{"id": "acc", "offbudget": False}, {"id": "savings", "offbudget": True}],
            "/categorygroups": [],
        }
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
            actual_account_ids=["acc", "savings"],
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
    assert ("/accounts/acc/transactions", {"since_date": "2020-01-01"}) in fake.calls


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


def test_unknown_top_level_keys_only_warn(monkeypatch):
    warnings = []
    monkeypatch.setattr(actual.logger, "warning", warnings.append)
    config = actual.Config(**BASE, categorise={"enabled": True})
    assert not config.categorize.enabled
    assert any("categorise" in w for w in warnings)


def test_misspelled_categorize_keys_are_rejected():
    with pytest.raises(ValueError):
        actual.Config(**BASE, categorize={"enable": True})


@pytest.mark.parametrize(
    "categorize",
    [
        {"min_confidence": 1.5},
        {"new_payee_min_confidence": -0.1},
        {"history_since": "2024-01-01", "history_until": "2023-01-01"},
        {"enabled": True, "descriptions": {"Food": "groceries"}},  # only used by model: full
    ],
)
def test_invalid_settings_are_rejected(categorize):
    with pytest.raises(ValueError):
        actual.Config(**BASE, categorize=categorize)


def test_failure_halfway_assigns_nothing(app, monkeypatch):
    from cleanab.classifier.hybrid import HybridClassifier

    original = HybridClassifier.predict

    def predict(self, example, categories):
        if example.payee == "Boom":
            raise RuntimeError("model missing")
        return original(self, example, categories)

    monkeypatch.setattr(HybridClassifier, "predict", predict)
    txs = (new_tx("Grocer", imported_id="a"), new_tx("Boom", imported_id="b"))
    app.prepare_transactions(txs)
    assert all("category" not in tx for tx in txs)


def test_transfers_splits_and_deleted_categories_are_not_learned(monkeypatch):
    txs = [
        *HISTORY,
        *({**history_tx(i, "p-other", "c-food"), "transfer_id": "t"} for i in range(11, 16)),
        *({**history_tx(i, "p-other", "c-food"), "subtransactions": [{}]} for i in range(16, 21)),
        *(history_tx(i, "p-other", "c-deleted") for i in range(21, 26)),
    ]
    app, _ = make_app(monkeypatch, txs, assign_new_payees=False)
    payees = [*PAYEES, {"id": "p-other", "name": "Other"}]
    app._get.data["/payees"] = payees
    (tx,) = app.prepare_transactions((new_tx("Other"),))
    assert "category" not in tx  # "Other" is unknown: only learned from skipped rows


@pytest.mark.parametrize("assign_new_payees", [False, True])
def test_known_payee_never_gets_a_category_it_did_not_have(monkeypatch, assign_new_payees):
    # "Grocer" was only ever Food, but many similarly named payees are Bills: the n-gram model may say Bills.
    txs = [history_tx(1, "p-grocer", "c-food")] + [
        {**history_tx(i, None, "c-bills"), "imported_payee": f"Grocer Store {i}"} for i in range(2, 12)
    ]
    app, _ = make_app(
        monkeypatch, txs, min_confidence=0.0, assign_new_payees=assign_new_payees, new_payee_min_confidence=0.0
    )
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx["category"] == "c-food"  # the payee's own history wins over similar payees


def test_category_reaches_the_import_request(app, monkeypatch):
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    sent = {}

    class Response:
        ok = True

        def json(self):
            return {"data": {"added": ["x"], "updated": []}}

    def fake_post(url, headers, json, timeout):
        sent.update(json)
        return Response()

    monkeypatch.setattr(actual.requests, "post", fake_post)
    app.create_transactions([tx])
    assert sent["transactions"][0]["category"] == "c-food"


def test_off_budget_accounts_get_no_categories_and_are_not_learned(monkeypatch):
    # The off-budget account's history says "Grocer" is Bills; it must neither be learned nor assigned there.
    txs = [*HISTORY]
    app, fake = make_app(monkeypatch, txs)
    savings = [history_tx(i, "p-grocer", "c-bills") for i in range(20, 30)]
    original = fake.__call__

    def get(path, **params):
        if path == "/accounts/savings/transactions":
            return savings
        return original(path, **params)

    monkeypatch.setattr(app, "_get", get)
    on_budget, off_budget = (
        new_tx("Grocer", imported_id="a"),
        {**new_tx("Grocer", imported_id="b"), "_account_id": "savings"},
    )
    app.prepare_transactions((on_budget, off_budget))
    assert on_budget["category"] == "c-food"
    assert "category" not in off_budget


def test_hidden_categories_are_never_assigned(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY)
    fake.data["/categories"] = [{"id": "c-food", "name": "Food", "hidden": True}, {"id": "c-bills", "name": "Bills"}]
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx.get("category") != "c-food"


def test_categories_in_hidden_groups_are_never_assigned(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY)
    fake.data["/categories"] = [
        {"id": "c-food", "name": "Food", "group_id": "g-old"},
        {"id": "c-bills", "name": "Bills"},
    ]
    fake.data["/categorygroups"] = [{"id": "g-old", "hidden": True}]
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx.get("category") != "c-food"


def test_known_payee_with_only_hidden_categories_is_not_suggested(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY)
    fake.data["/categories"] = [{"id": "c-food", "name": "Food", "hidden": True}, {"id": "c-bills", "name": "Bills"}]
    infos = []
    monkeypatch.setattr(actual.logger, "info", infos.append)
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert "category" not in tx
    assert not any("None" in line for line in infos)


def test_stale_account_id_does_not_disable_the_others(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY)
    app.config.actual_account_ids = ["acc", "deleted"]
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx["category"] == "c-food"
    assert not any(path == "/accounts/deleted/transactions" for path, _ in fake.calls)


def test_empty_categorize_section_is_allowed():
    assert not actual.Config(**BASE, categorize=None).categorize.enabled


def test_descriptions_may_stay_when_disabled():
    assert not actual.Config(**BASE, categorize={"descriptions": {"Food": "groceries"}}).categorize.enabled


def test_transfer_payees_never_get_a_category(monkeypatch):
    txs = [*HISTORY, {**history_tx(20, "p-grocer", None), "transfer_id": "t"}]
    app, _ = make_app(monkeypatch, txs, assign_new_payees=True, new_payee_min_confidence=0.0)
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert "category" not in tx


def test_full_model_failure_falls_back_to_light(monkeypatch):
    from cleanab.classifier import decider

    def broken(self):
        raise decider.ModelUnavailable("install the model extra")

    monkeypatch.setattr(decider.DeciderClassifier, "_load", broken)
    app, _ = make_app(monkeypatch, HISTORY, model="full")
    known, new = new_tx("Grocer", imported_id="a"), new_tx("Totally New Shop", imported_id="b")
    app.prepare_transactions((known, new))
    assert known["category"] == "c-food"
    assert "category" not in new


@pytest.mark.parametrize("placeholder", ["Unknown", "Unnamed"])
def test_payee_less_transactions_are_not_one_known_payee(monkeypatch, placeholder):
    txs = [*HISTORY, *(history_tx(i, None, "c-bills", imported_payee=placeholder) for i in range(20, 26))]
    app, _ = make_app(monkeypatch, txs)
    tx = {**new_tx(placeholder), "imported_payee": placeholder, "notes": "card payment"}
    (tx,) = app.prepare_transactions((tx,))
    assert "category" not in tx  # treated as a new payee: only suggested by default


def test_possible_match_with_manual_transfer_gets_no_category(monkeypatch):
    manual = {"date": "2024-05-04", "payee": "p-x", "amount": -500, "transfer_id": "t", "category": None}
    app, _ = make_app(monkeypatch, [*HISTORY, manual])
    near = {**new_tx("Grocer", imported_id="a"), "amount": -500}
    other_amount = {**new_tx("Grocer", imported_id="b"), "amount": -499}
    app.prepare_transactions((near, other_amount))
    assert "category" not in near
    assert other_amount["category"] == "c-food"


def test_category_groups_failure_skips_categorization(monkeypatch):
    import requests

    app, fake = make_app(monkeypatch, HISTORY)
    original = fake.__call__

    def get(path, **params):
        if path == "/categorygroups":
            raise requests.HTTPError("404")
        return original(path, **params)

    monkeypatch.setattr(app, "_get", get)
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert "category" not in tx  # hidden groups unknown: better no categories than hidden ones


def test_full_model_does_not_retry_on_other_errors(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY, model="full")
    calls = []

    def boom(path, **params):
        calls.append(path)
        raise RuntimeError("actual down")

    monkeypatch.setattr(app, "_get", boom)
    txs = (new_tx("Grocer"),)
    assert app.prepare_transactions(txs) == txs
    assert calls.count("/accounts") == 1


def test_description_warnings_only_with_full(monkeypatch):
    from cleanab.classifier import decider

    monkeypatch.setattr(decider.DeciderClassifier, "predict", lambda self, ex, cats: decider.Prediction(None))
    warnings = []
    monkeypatch.setattr(actual.logger, "warning", warnings.append)
    app, _ = make_app(monkeypatch, HISTORY, model="full", descriptions={"Fod": "typo", "Food": "groceries"})
    app.prepare_transactions((new_tx("Grocer"),))
    assert any("unknown categories" in w and "Fod" in w for w in warnings)


def test_transfer_payees_after_history_until_are_excluded(monkeypatch):
    txs = [*HISTORY, {**history_tx(20, "p-grocer", None), "date": "2024-02-01", "transfer_id": "t"}]
    app, _ = make_app(monkeypatch, txs, history_until=date(2023, 12, 31))
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert "category" not in tx


def test_history_until_limits_learning(monkeypatch):
    late = [{**history_tx(i, "p-grocer", "c-bills"), "date": f"2024-01-{i:02d}"} for i in range(1, 20)]
    app, _ = make_app(monkeypatch, [*HISTORY, *late], history_until=date(2023, 12, 31))
    (tx,) = app.prepare_transactions((new_tx("Grocer"),))
    assert tx["category"] == "c-food"


@pytest.mark.parametrize(
    ("manual_extra", "day", "blocked"),
    [
        ({"subtransactions": [{}]}, "2024-05-04", True),
        ({"is_parent": True}, "2024-05-04", True),
        ({"transfer_id": "t"}, "2024-05-11", True),  # 7 days after the manual one
        ({"transfer_id": "t"}, "2024-05-12", False),  # 8 days: Actual would not match
    ],
)
def test_manual_split_and_match_window(monkeypatch, manual_extra, day, blocked):
    manual = {"date": "2024-05-04", "payee": "p-x", "amount": -500, "category": None, **manual_extra}
    app, _ = make_app(monkeypatch, [*HISTORY, manual])
    tx = {**new_tx("Grocer", day=day), "amount": -500}
    app.prepare_transactions((tx,))
    assert ("category" not in tx) is blocked


def test_failed_full_run_logs_assignments_once(monkeypatch):
    from cleanab.classifier import decider

    def broken(self):
        raise decider.ModelUnavailable("not installed")

    monkeypatch.setattr(decider.DeciderClassifier, "_load", broken)
    infos = []
    monkeypatch.setattr(actual.logger, "info", infos.append)
    app, _ = make_app(monkeypatch, HISTORY, model="full")
    app.prepare_transactions((new_tx("Grocer", imported_id="a"), new_tx("Totally New Shop", imported_id="b")))
    assert sum(line.startswith("Category Food") for line in infos) == 1


def test_part_of_manual_split_blocks_matching_import(monkeypatch):
    manual = {"date": "2024-05-04", "payee": "p-x", "amount": -900, "category": None, "is_parent": True}
    manual["subtransactions"] = [{"amount": -500, "transfer_id": "t"}, {"amount": -400}]
    app, _ = make_app(monkeypatch, [*HISTORY, manual])
    tx = {**new_tx("Grocer"), "amount": -500}
    app.prepare_transactions((tx,))
    assert "category" not in tx


def test_hidden_duplicate_name_does_not_rename_visible_category(monkeypatch):
    app, fake = make_app(monkeypatch, HISTORY)
    fake.data["/categories"] = [*CATEGORIES, {"id": "c-old-food", "name": "Food", "hidden": True}]
    labels, hidden = app._category_labels()
    assert labels["c-food"] == "Food"
    assert labels["c-old-food"] == "Food (c-old-food)"
    assert hidden == {"Food (c-old-food)"}


def test_part_of_imported_split_blocks_matching_import(monkeypatch):
    split = {"date": "2024-05-04", "payee": "p-x", "amount": -50, "category": None, "imported_id": "old"}
    split["subtransactions"] = [{"amount": -30, "category": "c-food"}, {"amount": -20, "transfer_id": "t"}]
    app, _ = make_app(monkeypatch, [*HISTORY, split])
    tx = {**new_tx("Grocer"), "amount": -20}
    app.prepare_transactions((tx,))
    assert "category" not in tx
