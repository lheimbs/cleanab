import re

import pytest

from cleanab import utils
from cleanab.cleaner import FieldCleaner
from cleanab.models.cleaner import FinalizerDefinition, ReplacementDefinition
from cleanab.models.config import FinalizerFields, ReplacementFields


def make(replacements=None, finalizer=None):
    return FieldCleaner(ReplacementFields(**(replacements or {})), FinalizerFields(**(finalizer or {})))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("foo bar", "Foo Bar"),
        ("", ""),
        ("already Fine", "Already Fine"),
    ],
)
def test_capitalize_string(value, expected):
    assert utils.capitalize_string(value) == expected


def test_simple_replace():
    replace = utils.simple_replace_instance("foo", "bar")
    assert replace("a foo b foo") == ("a bar b bar", {})


def test_regex_sub_case_insensitive_by_default():
    sub = utils.regex_sub_instance(ReplacementDefinition(pattern="foo", repl="x"))
    assert sub("FOO foo")[0] == "x x"


def test_regex_sub_case_sensitive():
    sub = utils.regex_sub_instance(ReplacementDefinition(pattern="foo", repl="x", case_insensitive=False))
    assert sub("FOO foo")[0] == "FOO x"


def test_regex_sub_non_regex_escapes_pattern():
    sub = utils.regex_sub_instance(ReplacementDefinition(pattern="a.b", repl="x", regex=False))
    assert sub("a.b axb")[0] == "x axb"


def test_transform_extracts_into_other_field():
    definition = ReplacementDefinition(
        pattern=r"^(\w+) bought (\w+)$", repl="", transform={"applicant_name": r"\1", "purpose": r"\2"}
    )
    text, transformed = utils.regex_sub_instance(definition)("ann bought milk")
    assert text == ""
    assert transformed == {"applicant_name": "ann", "purpose": "milk"}


def test_transform_without_match_is_empty():
    definition = ReplacementDefinition(pattern="zzz", transform={"purpose": r"\g<0>"})
    assert utils.regex_sub_instance(definition)("abc")[1] == {}


def test_replacement_definition_is_hashable():
    a = ReplacementDefinition(pattern="x", transform={"purpose": "y"})
    b = ReplacementDefinition(pattern="x", transform={"purpose": "y"})
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_cleaner_chains_replacements_in_order():
    cleaner = make({"purpose": ["a", {"pattern": "b", "repl": "c"}, ["c", "d"]]}, {"purpose": {"capitalize": False}})
    assert cleaner.clean({"purpose": "abcd"})["purpose"] == ""


def test_cleaner_applies_transformations_to_other_fields():
    cleaner = make(
        {"purpose": [{"pattern": r"^(\w+): ", "repl": "", "transform": {"applicant_name": r"\1"}}]},
        {"applicant_name": {"capitalize": False}, "purpose": {"capitalize": False}},
    )
    data = cleaner.clean({"applicant_name": "old", "purpose": "shop: item"})
    assert data == {"applicant_name": "shop", "purpose": "item"}


def test_cleaner_skips_empty_and_missing_fields():
    cleaner = make({"purpose": ["x"]})
    assert cleaner.clean({"purpose": None, "applicant_name": ""}) == {"purpose": None, "applicant_name": ""}


def test_finalizer_capitalize_and_strip():
    cleaner = make()
    assert cleaner.clean({"purpose": "  hello world "})["purpose"] == "Hello World"


def test_finalizer_can_be_disabled():
    cleaner = make(finalizer={"purpose": FinalizerDefinition(capitalize=False, strip=False)})
    assert cleaner.clean({"purpose": " hello "})["purpose"] == " hello "


def test_invalid_regex_fails_at_startup():
    with pytest.raises(re.error):
        make({"purpose": [{"pattern": "("}]})


def test_compile_single_cleaner_rejects_garbage():
    with pytest.raises(ValueError):
        FieldCleaner.compile_single_cleaner(5)
