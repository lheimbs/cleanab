"""Staged combination: known payees from history, new payees from an ensemble of models."""

from __future__ import annotations

from .base import Classifier, Example, Prediction
from .baselines import ExactPayeeClassifier


class HybridClassifier:
    """Exact payee match first; otherwise average the probabilities of the fallback classifiers.

    The fallbacks (e.g. a char n-gram model and a zero-shot LLM) are only asked about payees the history does
    not know, so an expensive model runs once per new payee instead of once per transaction.
    """

    def __init__(self, fallbacks: list[Classifier], name: str = "hybrid") -> None:
        self.name = name
        self.exact = ExactPayeeClassifier()
        self.fallbacks = fallbacks

    def fit(self, history: list[Example]) -> None:
        self.exact.fit(history)
        for fallback in self.fallbacks:
            fallback.fit(history)

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        known = self.exact.predict(example, categories)
        if known.category is not None:
            return known
        totals = dict.fromkeys(categories, 0.0)
        answered = 0
        for fallback in self.fallbacks:
            prediction = fallback.predict(example, categories)
            if not prediction.scores:
                continue
            answered += 1
            for category, score in prediction.scores.items():
                totals[category] += score
        if not answered:
            return Prediction(None)
        scores = {category: total / answered for category, total in totals.items()}
        best = max(scores, key=scores.__getitem__)
        return Prediction(best, scores[best], scores)
