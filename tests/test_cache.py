import datetime
import json
from decimal import Decimal

from mt940.models import Amount

from cleanab.models import AccountConfig
from cleanab.models import account_config as account_config_module
from cleanab.utils import json_default, json_object_hook

ACCOUNT = {
    "iban": "DE89370400440532013000",
    "per_app_id": "abc",
    "fints_username": "user",
    "fints_password": "pw",
    "fints_blz": "37040044",
    "fints_endpoint": "https://fints.example.com/fints",
    "friendly_name": "Main",
}


def roundtrip(value):
    return json.loads(json.dumps(value, default=json_default), object_hook=json_object_hook)


def test_roundtrip_types():
    data = {
        "date": datetime.date(2024, 1, 2),
        "stamp": datetime.datetime(2024, 1, 2, 3, 4, 5),
        "decimal": Decimal("1.50"),
        "text": "hello",
        "none": None,
    }
    assert roundtrip(data) == data


def test_roundtrip_amounts():
    debit = roundtrip({"amount": Amount("12,34", "D", "EUR")})["amount"]
    credit = roundtrip({"amount": Amount("12,34", "C", "EUR")})["amount"]
    assert debit.amount == Decimal("-12.34")
    assert credit.amount == Decimal("12.34")
    assert debit.currency == "EUR"


def test_unknown_types_do_not_fail():
    assert roundtrip({"x": object})["x"].startswith("<class")


def test_account_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(account_config_module, "CACHE_HOME", tmp_path)
    account = AccountConfig(**ACCOUNT)
    assert not account.has_account_cache
    data = [{"date": datetime.date(2024, 1, 2), "amount": Amount("1,00", "C", "EUR")}]
    account.write_account_cache(data)
    assert account.has_account_cache
    loaded = account.read_account_cache()
    assert loaded[0]["date"] == data[0]["date"]
    assert loaded[0]["amount"].amount == Decimal("1.00")
