from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cleanab.models.config import Config

SAMPLE = Path(__file__).parent.parent / "config.yaml.sample"


def make_config(account_data, **extra):
    return Config.model_validate({"accounts": [account_data], **extra})


def test_defaults(account_data):
    config = make_config(account_data)
    assert config.timespan.maximum_days == 30
    assert config.cleanab.fints_product_id is None
    assert config.apps == {}


def test_accounts_required():
    with pytest.raises(ValidationError):
        Config.model_validate({"accounts": []})


def test_replacement_entries(account_data):
    config = make_config(
        account_data,
        replacements={
            "applicant_name": [
                "plain",
                {"pattern": "a+", "repl": "b"},
                ["nested", {"pattern": "c"}],
            ]
        },
    )
    assert len(config.replacements.applicant_name) == 3


def test_replacement_rejects_unknown_keys(account_data):
    with pytest.raises(ValidationError):
        make_config(account_data, replacements={"purpose": [{"pattern": "x", "bogus": 1}]})


def test_unknown_app_is_rejected(account_data):
    with pytest.raises(ValidationError, match="Unknown app"):
        make_config(account_data, apps={"nope": {}})


def test_load_actual_app(account_data):
    config = make_config(
        account_data,
        apps={
            "actual": {
                "actual_api_url": "http://localhost:5007",
                "actual_api_key": "key",
                "actual_sync_id": "sync",
                "actual_account_ids": ["acc-1"],
            }
        },
    )
    config.load_apps()
    (app,) = config.get_apps()
    assert str(app) == "Actual App Connection"


def test_load_firefly_app(account_data):
    config = make_config(
        account_data,
        apps={
            "firefly_iii_fidi": {
                "fidi_url": "http://localhost:8080",
                "default_account_id": 1,
                "auto_import_secret": "secret",
                "personal_access_token": "token",
            }
        },
    )
    config.load_apps()
    assert len(config.get_apps()) == 1


def test_sample_replacements_are_valid(account_data):
    sample = yaml.safe_load(SAMPLE.read_text())
    config = make_config(
        account_data,
        replacements=sample["replacements"],
        finalizer=sample.get("finalizer", {}),
    )
    assert config.replacements.applicant_name


def test_user_replacements_are_kept(account_data):
    config = make_config(account_data, replacements={"purpose": ["foo"]})
    assert dict(config.replacements)["purpose"] == ["foo"]


def test_user_finalizer_is_kept(account_data):
    config = make_config(account_data, finalizer={"applicant_name": {"capitalize": False}})
    assert config.finalizer.applicant_name.capitalize is False
    assert config.finalizer.purpose.capitalize is True


def test_configured_replacements_are_applied(account_data):
    from cleanab.cleaner import FieldCleaner

    config = make_config(
        account_data,
        replacements={"applicant_name": [{"pattern": "gmbh", "repl": "GmbH", "regex": False}]},
        finalizer={"applicant_name": {"capitalize": False}},
    )
    cleaner = FieldCleaner(config.replacements, config.finalizer)
    assert cleaner.clean({"applicant_name": "Foo Gmbh"})["applicant_name"] == "Foo GmbH"


def test_tan_display_is_case_insensitive():
    from cleanab.models.config import CleanabConfig

    assert CleanabConfig(tan_display=" Blocks ").tan_display == "blocks"
