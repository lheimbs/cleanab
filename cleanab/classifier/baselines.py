"""History-based baselines that every model has to beat."""

from __future__ import annotations

from collections import Counter, defaultdict

from .base import Example, Prediction
from .normalize import model_text, normalize_text


class ExactPayeeClassifier:
    """Majority category of past transactions with the same normalized payee."""

    name = "exact-payee"

    def __init__(self) -> None:
        self._by_payee: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, history: list[Example]) -> None:
        self._by_payee.clear()
        for example in history:
            if example.category:
                self._by_payee[normalize_text(example.payee)][example.category] += 1

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        counts = self._by_payee.get(normalize_text(example.payee))
        if not counts:
            return Prediction(None)
        total = sum(counts.values())
        scores = {category: count / total for category, count in counts.items() if category in categories}
        if not scores:
            return Prediction(None)
        best = max(scores, key=scores.__getitem__)
        # Few observations should not look as certain as many: shrink towards 0.5 for tiny samples.
        confidence = scores[best] * total / (total + 1)
        return Prediction(best, confidence, scores)


class CharNgramClassifier:
    """Character n-gram TF-IDF with logistic regression (needs scikit-learn)."""

    name = "char-ngram-lr"

    def __init__(self, ngram_range: tuple[int, int] = (2, 5), c: float = 10.0) -> None:
        self.ngram_range = ngram_range
        self.c = c
        self._vectorizer = None
        self._model = None

    def fit(self, history: list[Example]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        labeled = [e for e in history if e.category]
        classes = {e.category for e in labeled}
        if len(classes) < 2:
            self._vectorizer = self._model = None
            self._only_class = next(iter(classes), None)
            return
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range, sublinear_tf=True)
        features = self._vectorizer.fit_transform([model_text(e.payee, e.purpose) for e in labeled])
        self._model = LogisticRegression(C=self.c, max_iter=1000)
        self._model.fit(features, [e.category for e in labeled])

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        if self._model is None or self._vectorizer is None:
            only = getattr(self, "_only_class", None)
            return Prediction(only if only in categories else None, 0.5 if only else 0.0)
        features = self._vectorizer.transform([model_text(example.payee, example.purpose)])
        probabilities = self._model.predict_proba(features)[0]
        scores = {
            cls: float(p) for cls, p in zip(self._model.classes_, probabilities, strict=True) if cls in categories
        }
        if not scores:
            return Prediction(None)
        best = max(scores, key=scores.__getitem__)
        return Prediction(best, scores[best], scores)
