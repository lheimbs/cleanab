from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Example:
    """A transaction as seen by a classifier, optionally with its known category."""

    payee: str
    purpose: str = ""
    category: str | None = None
    date: date | None = None


@dataclass(frozen=True)
class Prediction:
    """A classifier's answer. `category` is None when the classifier abstains."""

    category: str | None
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Classifier(Protocol):
    name: str

    def fit(self, history: list[Example]) -> None:
        """Learn from confirmed transactions. Zero-shot models may only record the category list."""

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        """Pick one of `categories` for the example."""
