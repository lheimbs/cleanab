"""Evaluation metrics for classifiers that can abstain and report a confidence."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Prediction


@dataclass(frozen=True)
class Report:
    n: int
    accuracy: float  # abstentions count as wrong
    coverage: float  # fraction answered at all
    coverage_at_precision: float  # fraction auto-assigned while keeping the target precision
    ece: float  # expected calibration error over answered items
    threshold: float | None  # confidence threshold behind coverage_at_precision


def accuracy(predictions: list[Prediction], truths: list[str]) -> float:
    if not truths:
        return 0.0
    return sum(p.category == t for p, t in zip(predictions, truths, strict=True)) / len(truths)


def ece(predictions: list[Prediction], truths: list[str], bins: int = 10) -> float:
    answered = [(p, t) for p, t in zip(predictions, truths, strict=True) if p.category is not None]
    if not answered:
        return 0.0
    total = len(answered)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        bucket = [(p, t) for p, t in answered if low <= p.confidence < high or (b == bins - 1 and p.confidence == 1.0)]
        if not bucket:
            continue
        confidence = sum(p.confidence for p, _ in bucket) / len(bucket)
        hit_rate = sum(p.category == t for p, t in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - hit_rate)
    return error


def coverage_at_precision(
    predictions: list[Prediction], truths: list[str], target: float = 0.95
) -> tuple[float, float | None]:
    """Largest fraction of items that can be auto-assigned while precision stays >= target.

    Returns (coverage over all items, confidence threshold that achieves it).
    """
    answered = sorted(
        ((p.confidence, p.category == t) for p, t in zip(predictions, truths, strict=True) if p.category is not None),
        key=lambda item: -item[0],
    )
    best_count, best_threshold, correct = 0, None, 0
    for index, (confidence, hit) in enumerate(answered, start=1):
        correct += hit
        # Only cut between distinct confidence values, so ties are accepted together.
        is_boundary = index == len(answered) or answered[index][0] != confidence
        if is_boundary and correct / index >= target:
            best_count, best_threshold = index, confidence
    return (best_count / len(truths) if truths else 0.0), best_threshold


def evaluate(predictions: list[Prediction], truths: list[str], target_precision: float = 0.95) -> Report:
    coverage, threshold = coverage_at_precision(predictions, truths, target_precision)
    answered = sum(p.category is not None for p in predictions)
    return Report(
        n=len(truths),
        accuracy=accuracy(predictions, truths),
        coverage=answered / len(truths) if truths else 0.0,
        coverage_at_precision=coverage,
        ece=ece(predictions, truths),
        threshold=threshold,
    )
