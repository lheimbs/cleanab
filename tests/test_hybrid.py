from cleanab.classifier.base import Example, Prediction
from cleanab.classifier.hybrid import HybridClassifier


class Fixed:
    name = "fixed"

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def fit(self, history):
        pass

    def predict(self, example, categories):
        self.calls += 1
        best = max(self.scores, key=self.scores.__getitem__)
        return Prediction(best, self.scores[best], self.scores)


def test_known_payee_skips_fallbacks():
    fallback = Fixed({"A": 1.0, "B": 0.0})
    hybrid = HybridClassifier([fallback])
    hybrid.fit([Example("rewe", category="B")])
    assert hybrid.predict(Example("rewe"), ["A", "B"]).category == "B"
    assert fallback.calls == 0


def test_new_payee_averages_fallbacks():
    hybrid = HybridClassifier([Fixed({"A": 0.9, "B": 0.1}), Fixed({"A": 0.1, "B": 0.5})])
    hybrid.fit([])
    prediction = hybrid.predict(Example("new"), ["A", "B"])
    assert prediction.category == "A"
    assert round(prediction.confidence, 2) == 0.5


def test_no_answer_abstains():
    hybrid = HybridClassifier([])
    assert hybrid.predict(Example("x"), ["A"]).category is None


def test_known_classifier_answers_for_seen_payees():
    known = Fixed({"A": 0.2, "B": 0.8})
    hybrid = HybridClassifier([], known=known)
    hybrid.fit([Example("rewe", category="A")])
    assert hybrid.predict(Example("rewe"), ["A", "B"]).category == "B"
