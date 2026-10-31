from datetime import date

import pytest

from cleanab.classifier.base import Classifier, Example, Prediction
from cleanab.classifier.baselines import CharNgramClassifier, ExactPayeeClassifier
from cleanab.classifier.metrics import accuracy, coverage_at_precision, ece, evaluate
from cleanab.classifier.normalize import model_text, normalize_text

# --- normalize ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SUMUP *BAECKEREI MUELLER", "baeckerei mueller"),
        ("PayPal *SPOTIFY", "spotify"),
        ("PP.1234.PP . ZALANDO", "zalando"),
        ("Kaufumsatz EDEKA Berlin 12.03.", "edeka berlin"),
        ("Lastschrift REWE Markt GmbH", "rewe markt gmbh"),
        ("EREF+ABC123XYZ MREF+M-77 CRED+DE98ZZZ09999999999 SVWZ+Rechnung 2024", "rechnung"),
        ("Zahlung an DE89 3704 0044 0532 0130 00 Miete", "zahlung an miete"),
        ("Amzn Mktp De*AB12CD34", "ab12cd34"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_model_text_joins_payee_and_purpose():
    assert model_text("EDEKA", "Kaufumsatz 12.03. Einkauf") == "edeka | einkauf"
    assert model_text("EDEKA", "") == "edeka"


# --- baselines ---


def history():
    rows = [
        ("EDEKA Berlin", "Einkauf", "Groceries"),
        ("EDEKA Berlin", "Einkauf", "Groceries"),
        ("REWE Markt", "Kaufumsatz", "Groceries"),
        ("Netflix", "Abo", "Subscriptions"),
        ("Spotify", "Abo Premium", "Subscriptions"),
        ("Vermieter Mueller", "Miete Oktober", "Rent"),
        ("Vermieter Mueller", "Miete November", "Rent"),
        ("Shell Tankstelle", "Kraftstoff", "Transport"),
        ("Aral Tankstelle", "Kraftstoff", "Transport"),
    ]
    return [
        Example(payee, purpose, category, date(2024, 1, i + 1)) for i, (payee, purpose, category) in enumerate(rows)
    ]


CATEGORIES = ["Groceries", "Subscriptions", "Rent", "Transport"]


def test_baselines_satisfy_protocol():
    assert isinstance(ExactPayeeClassifier(), Classifier)
    assert isinstance(CharNgramClassifier(), Classifier)


def test_exact_payee_hit_and_miss():
    clf = ExactPayeeClassifier()
    clf.fit(history())
    hit = clf.predict(Example("edeka berlin", "whatever"), CATEGORIES)
    assert hit.category == "Groceries"
    assert 0.5 < hit.confidence < 1.0
    assert clf.predict(Example("Unknown Shop"), CATEGORIES).category is None


def test_exact_payee_confidence_grows_with_evidence():
    clf = ExactPayeeClassifier()
    clf.fit([Example("a", category="X")] * 1 + [Example("b", category="X")] * 10)
    assert clf.predict(Example("b"), ["X"]).confidence > clf.predict(Example("a"), ["X"]).confidence


def test_exact_payee_ignores_categories_outside_list():
    clf = ExactPayeeClassifier()
    clf.fit([Example("a", category="Gone")])
    assert clf.predict(Example("a"), ["Other"]).category is None


def test_char_ngram_generalizes_to_unseen_payee():
    clf = CharNgramClassifier()
    clf.fit(history())
    prediction = clf.predict(Example("Jet Tankstelle", "Kraftstoff"), CATEGORIES)
    assert prediction.category == "Transport"
    assert prediction.confidence == pytest.approx(max(prediction.scores.values()))


def test_char_ngram_with_single_class_does_not_crash():
    clf = CharNgramClassifier()
    clf.fit([Example("a", category="X"), Example("b", category="X")])
    assert clf.predict(Example("c"), ["X"]).category == "X"


# --- metrics ---


def preds(*pairs):
    return [Prediction(category, confidence) for category, confidence in pairs]


def test_accuracy_counts_abstentions_as_wrong():
    assert accuracy(preds(("a", 0.9), (None, 0.0)), ["a", "a"]) == 0.5


def test_coverage_at_precision_picks_largest_safe_prefix():
    predictions = preds(("a", 0.99), ("a", 0.95), ("b", 0.9), ("a", 0.6))
    truths = ["a", "a", "a", "a"]  # third is wrong
    coverage, threshold = coverage_at_precision(predictions, truths, target=0.95)
    assert coverage == 0.5
    assert threshold == 0.95


def test_coverage_at_precision_zero_when_unreachable():
    coverage, threshold = coverage_at_precision(preds(("a", 0.9)), ["b"], target=0.95)
    assert (coverage, threshold) == (0.0, None)


def test_ties_are_accepted_together():
    predictions = preds(("a", 0.9), ("b", 0.9))
    coverage, _ = coverage_at_precision(predictions, ["a", "a"], target=0.9)
    assert coverage == 0.0  # 50% precision on the tied pair


def test_ece_perfectly_calibrated_is_zero():
    predictions = preds(*[("a", 1.0)] * 4)
    assert ece(predictions, ["a"] * 4) == 0.0


def test_ece_overconfident():
    predictions = preds(*[("a", 0.95)] * 4)
    assert ece(predictions, ["a", "b", "b", "b"]) == pytest.approx(0.7)


def test_evaluate_report():
    report = evaluate(preds(("a", 0.99), (None, 0.0)), ["a", "a"])
    assert report.n == 2
    assert report.accuracy == 0.5
    assert report.coverage == 0.5
    assert report.coverage_at_precision == 0.5
