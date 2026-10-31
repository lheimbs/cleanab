"""Build the classifier pipeline selected in the config."""

from __future__ import annotations

from typing import Literal

from .base import Classifier
from .baselines import CharNgramClassifier
from .hybrid import HybridClassifier

Model = Literal["light", "full"]


def build_classifier(model: Model = "light", descriptions: dict[str, str] | None = None) -> HybridClassifier:
    """`light`: payee history plus a char n-gram model (needs the `ml-lite` extra).

    `full`: additionally asks a zero-shot decision model about payees the history does not know
    (needs the `models` dependency group; downloads a ~2.7 GB model on first use).
    """
    ngram: Classifier = CharNgramClassifier()
    fallbacks: list[Classifier] = [ngram]
    if model == "full":
        from .decider import DeciderClassifier

        fallbacks.append(DeciderClassifier(descriptions=descriptions))
    return HybridClassifier(fallbacks, name=f"hybrid-{model}", known=ngram)
