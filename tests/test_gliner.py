from cleanab.classifier.base import Example
from cleanab.classifier.gliner import GlinerClassifier, parse_result

LABELS = ["Food", "Bills"]


def test_parse_shapes():
    assert parse_result({"task": "Food"}, LABELS)[0] == "Food"
    assert parse_result({"task": {"label": "Bills", "confidence": 0.7}}, LABELS)[:2] == ("Bills", 0.7)
    assert (
        parse_result({"task": [{"label": "Food", "confidence": 0.2}, {"label": "Bills", "confidence": 0.8}]}, LABELS)[0]
        == "Bills"
    )
    assert parse_result({"task": "Other"}, LABELS)[0] is None
    assert parse_result(42, LABELS)[0] is None


def test_predict_maps_descriptions_back():
    class Fake:
        def classify_text(self, text, schema, include_confidence=False):
            assert "Bills: insurance" in schema["category"]
            return {"category": "Bills: insurance"}

    classifier = GlinerClassifier(descriptions={"Bills": "insurance"})
    classifier._extractor = Fake()
    assert classifier.predict(Example("hkk"), LABELS).category == "Bills"
