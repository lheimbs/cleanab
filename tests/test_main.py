from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cleanab import main as main_module
from cleanab.apps.base import BaseApp
from cleanab.fints import LoginSkipped
from cleanab.main import Cleanab
from cleanab.models import account_config as account_config_module
from cleanab.models.config import Config


class FakeApp(BaseApp):
    def __init__(self):
        self.created = []

    def create_transactions(self, transactions):
        self.created.append(transactions)
        return ["n"] * len(transactions), []

    def augment_transaction(self, transaction, account):
        return (account.per_app_id, transaction.import_id)

    def create_intermediary(self, transactions):
        return "intermediary"


def raw_tx(**overrides):
    data = {
        "date": date(2024, 1, 2),
        "amount": SimpleNamespace(amount=Decimal("-1.00")),
        "applicant_name": "Shop",
        "purpose": "Stuff",
    }
    data.update(overrides)
    return data


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(account_config_module, "CACHE_HOME", tmp_path)


@pytest.fixture
def make_cleanab(account_data):
    def factory(*, apps=1, **kwargs):
        config = Config.model_validate({"accounts": [account_data]})
        config._parsed_apps = [FakeApp() for _ in range(apps)]
        cleanab = Cleanab(config=config, **kwargs)
        cleanab.setup_app_connections = lambda: None
        cleanab.setup()
        return cleanab

    return factory


@pytest.fixture
def fints(monkeypatch):
    calls = []
    result = [raw_tx(), raw_tx(purpose="Other")]

    def fake(account, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(main_module, "process_fints_account", fake)
    fake.calls = calls
    fake.result = result
    return fake


def test_test_mode_implies_dry_run_and_verbose(make_cleanab):
    cleanab = make_cleanab(test=True)
    assert cleanab.dry_run and cleanab.verbose


def test_earliest_date_respects_maximum_days(make_cleanab):
    cleanab = make_cleanab()
    assert cleanab.earliest == date.today() - timedelta(days=30)


def test_processor_returns_augmented_transactions(make_cleanab, fints):
    cleanab = make_cleanab()
    result = cleanab.processor(cleanab.accounts[0])
    assert len(result) == 2
    assert all(entry[0][0] == "acc-1" for entry in result)
    assert fints.calls[0]["latest"] == cleanab.today


def test_processor_skips_future_transactions(make_cleanab, fints):
    fints.result.append(raw_tx(date=date.today() + timedelta(days=3)))
    cleanab = make_cleanab()
    assert len(cleanab.processor(cleanab.accounts[0])) == 2


def test_processor_skips_holding_accounts(make_cleanab, fints):
    cleanab = make_cleanab()
    account = cleanab.accounts[0]
    account.account_type = "holding"
    assert cleanab.processor(account) == []
    assert fints.calls == []


def test_processor_swallows_errors(make_cleanab, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bank down")

    monkeypatch.setattr(main_module, "process_fints_account", boom)
    cleanab = make_cleanab()
    assert cleanab.processor(cleanab.accounts[0]) == []


def test_test_mode_reads_cache_on_second_run(make_cleanab, fints):
    cleanab = make_cleanab(test=True)
    account = cleanab.accounts[0]
    cleanab.processor(account)
    cleanab.processor(account)
    assert len(fints.calls) == 1


def test_run_creates_transactions_per_app(make_cleanab, fints):
    cleanab = make_cleanab(apps=2)
    cleanab.run()
    first, second = cleanab.config.get_apps()
    assert len(first.created[0]) == 2
    assert len(second.created[0]) == 2


def test_dry_run_creates_nothing_in_any_app(make_cleanab, fints):
    cleanab = make_cleanab(apps=2, dry_run=True)
    cleanab.run()
    assert all(app.created == [] for app in cleanab.config.get_apps())


def test_run_without_transactions_does_nothing(make_cleanab, fints):
    fints.result.clear()
    cleanab = make_cleanab()
    cleanab.run()
    assert cleanab.config.get_apps()[0].created == []


def test_processor_skips_failed_login_without_traceback(make_cleanab, monkeypatch):
    def skipped(*args, **kwargs):
        raise LoginSkipped("Skipping login for user at url, it already failed in this run")

    warnings, tracebacks = [], []
    monkeypatch.setattr(main_module, "process_fints_account", skipped)
    monkeypatch.setattr(main_module.logger, "warning", lambda msg, *a: warnings.append(msg % a))
    monkeypatch.setattr(main_module.logger, "exception", lambda *a: tracebacks.append(a))
    cleanab = make_cleanab()
    assert cleanab.processor(cleanab.accounts[0]) == []
    assert "already failed" in warnings[-1]
    assert tracebacks == []


def test_setup_does_not_probe_the_terminal(make_cleanab, monkeypatch):
    from cleanab import tan_display

    monkeypatch.setattr(tan_display, "kitty_graphics_supported", lambda: pytest.fail("probed at startup"))
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    monkeypatch.delenv("CLEANAB_TAN_DISPLAY", raising=False)
    assert make_cleanab().tan_display == "auto"
