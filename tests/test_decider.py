from cleanab.classifier.base import Example
from cleanab.classifier.decider import DeciderClassifier


def test_predict_uses_choice_and_probs():
    class Fake:
        def decide(self, state, questions):
            assert "hkk" in state
            assert "Bills: insurance" in questions[0]["question"]
            assert questions[0]["options"] == ["Food", "Bills"]
            return [{"choice": "Bills", "confidence": 0.8, "probs": {"Bills": 0.8, "Food": 0.2}}]

    classifier = DeciderClassifier(descriptions={"Bills": "insurance"})
    classifier._decider = Fake()
    prediction = classifier.predict(Example("hkk"), ["Food", "Bills"])
    assert (prediction.category, prediction.confidence) == ("Bills", 0.8)


def test_unknown_choice_abstains():
    class Fake:
        def decide(self, state, questions):
            return [{"choice": "???", "confidence": 1.0, "probs": {}}]

    classifier = DeciderClassifier()
    classifier._decider = Fake()
    assert classifier.predict(Example("x"), ["Food"]).category is None
