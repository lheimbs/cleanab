from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

from cleanab.cleaner import FieldCleaner
from cleanab.models.config import FinalizerFields, ReplacementFields
from cleanab.transactions import process_transaction


def make_cleaner(**replacements):
    return FieldCleaner(ReplacementFields(**replacements), FinalizerFields())


def raw(**overrides):
    data = {
        "date": date(2024, 1, 2),
        "amount": SimpleNamespace(amount=Decimal("-12.34")),
        "applicant_name": "Some Shop",
        "purpose": "Thanks",
    }
    data.update(overrides)
    return data


def test_amount_is_converted_to_milliunits():
    result = process_transaction(raw(), make_cleaner())
    assert result.amount == -12340


def test_import_id_is_stable():
    first = process_transaction(raw(), make_cleaner())
    second = process_transaction(raw(), make_cleaner())
    assert first.import_id == second.import_id
    assert first.import_id != process_transaction(raw(purpose="Other"), make_cleaner()).import_id


def test_future_transactions_are_skipped():
    future = date.today() + timedelta(days=2)
    assert process_transaction(raw(date=future), make_cleaner()) is None


def test_missing_applicant_defaults_to_unknown():
    result = process_transaction(raw(applicant_name=None, purpose=""), make_cleaner())
    assert result.applicant_name == "Unknown"


def test_credit_card_purpose_is_split():
    purpose = "AMZN Mktp DE   EUR   12,34 foo"
    result = process_transaction(raw(applicant_name="", purpose=purpose), make_cleaner())
    assert result.applicant_name == "Amzn Mktp De"  # default finalizer capitalizes


def test_long_purpose_is_truncated():
    result = process_transaction(raw(purpose="x" * 500), make_cleaner())
    assert len(result.purpose) == 200


def test_plain_string_replacement():
    cleaner = make_cleaner(applicant_name=["Visa "])
    result = process_transaction(raw(applicant_name="Visa Some Shop"), cleaner)
    assert result.applicant_name == "Some Shop"


def test_regex_replacement():
    cleaner = make_cleaner(applicant_name=[{"pattern": r"Amzn Mktp De\*.*$", "repl": "Amazon Marketplace"}])
    result = process_transaction(raw(applicant_name="Amzn Mktp De*AB12"), cleaner)
    assert result.applicant_name == "Amazon Marketplace"
