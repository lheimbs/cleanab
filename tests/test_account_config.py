import pytest
from pydantic import ValidationError

from cleanab.models import AccountConfig

BASE = {
    "iban": "DE89370400440532013000",
    "per_app_id": "abc",
    "fints_username": "user",
    "fints_password": "pw",
    "fints_blz": "37040044",
    "fints_endpoint": "https://fints.example.com/fints",
    "friendly_name": "Main",
}


def test_iban_is_normalized():
    account = AccountConfig(**{**BASE, "iban": "de89 3704 0044 0532 0130 00"})
    assert account.iban == "DE89370400440532013000"


def test_invalid_iban_is_rejected():
    with pytest.raises(ValidationError):
        AccountConfig(**{**BASE, "iban": "DE89370400440532013001"})
