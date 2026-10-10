import pytest

ACCOUNT = {
    "iban": "DE89370400440532013000",
    "per_app_id": "acc-1",
    "fints_username": "user",
    "fints_password": "pw",
    "fints_blz": "37040044",
    "fints_endpoint": "https://fints.example.com/fints",
    "friendly_name": "Main",
}


@pytest.fixture
def account_data():
    return dict(ACCOUNT)


@pytest.fixture
def account(account_data):
    from cleanab.models import AccountConfig

    return AccountConfig(**account_data)


@pytest.fixture
def transaction():
    from datetime import date

    from cleanab.models import FintsTransaction

    return FintsTransaction(
        date=date(2024, 1, 2),
        amount=-12340,
        applicant_name="Some Shop",
        purpose="Thanks",
        import_id="abc123",
    )
