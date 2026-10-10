import pytest

from cleanab.validators import is_blz, is_https_url, is_iban, is_uuid


@pytest.mark.parametrize(
    "iban",
    [
        "DE89370400440532013000",
        "DE89 3704 0044 0532 0130 00",
        "de89370400440532013000",
        "GB82WEST12345698765432",
        "FR1420041010050500013M02606",
    ],
)
def test_valid_ibans(iban):
    assert is_iban(iban)


@pytest.mark.parametrize(
    "iban",
    [
        "",
        "not an iban",
        "DE89370400440532013001",  # wrong check digits
        "DE00370400440532013000",
        "D189370400440532013000",
        "DE89",
    ],
)
def test_invalid_ibans(iban):
    assert not is_iban(iban)


def test_is_uuid():
    assert is_uuid("123e4567-e89b-12d3-a456-426614174000")
    assert not is_uuid("nope")


def test_is_https_url():
    assert is_https_url("https://example.com")
    assert not is_https_url("http://example.com")


def test_is_blz():
    assert is_blz("12345678")
    assert not is_blz("1234")
    assert not is_blz("abcdefgh")
