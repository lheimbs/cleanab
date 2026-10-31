"""Evaluate classifiers on a labeled export of your own categorized transactions.

CSV columns: date (YYYY-MM-DD), payee, purpose, category. See experiments/README.md.

    uv run python -m experiments.evaluate data/transactions.csv --classifiers exact-payee,char-ngram-lr
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date
from pathlib import Path

from cleanab.classifier.base import Classifier, Example
from cleanab.classifier.baselines import CharNgramClassifier, ExactPayeeClassifier
from cleanab.classifier.metrics import evaluate
from cleanab.classifier.normalize import normalize_text

CLASSIFIERS = {
    "exact-payee": ExactPayeeClassifier,
    "char-ngram-lr": CharNgramClassifier,
}


def load_examples(path: Path, since: date | None = None, reconciled_only: bool = False) -> list[Example]:
    """Load labeled rows, optionally only those on/after `since` or marked reconciled in Actual."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    examples = [
        Example(
            payee=row["payee"],
            purpose=row.get("purpose", ""),
            category=row["category"] or None,
            date=date.fromisoformat(row["date"]),
        )
        for row in rows
        if row.get("category")
        and (since is None or date.fromisoformat(row["date"]) >= since)
        and (not reconciled_only or row.get("reconciled") == "1")
    ]
    return sorted(examples, key=lambda e: e.date)


def time_split(examples: list[Example], train_fraction: float) -> tuple[list[Example], list[Example]]:
    """Chronological split: a random split leaks recurring payees and inflates accuracy."""
    cut = int(len(examples) * train_fraction)
    return examples[:cut], examples[cut:]


def run(classifier: Classifier, train: list[Example], test: list[Example], categories: list[str], target: float):
    classifier.fit(train)
    seen_payees = {normalize_text(e.payee) for e in train}
    started = time.perf_counter()
    predictions = [classifier.predict(e, categories) for e in test]
    elapsed = time.perf_counter() - started

    groups = {
        "all": [True] * len(test),
        "seen payee": [normalize_text(e.payee) in seen_payees for e in test],
        "unseen payee": [normalize_text(e.payee) not in seen_payees for e in test],
    }
    for label, mask in groups.items():
        idx = [i for i, keep in enumerate(mask) if keep]
        if not idx:
            continue
        report = evaluate([predictions[i] for i in idx], [test[i].category for i in idx], target)
        threshold = f"{report.threshold:.2f}" if report.threshold is not None else "-"
        print(
            f"{classifier.name:16} {label:13} n={report.n:5} acc={report.accuracy:.3f} "
            f"answered={report.coverage:.3f} cov@{target:.0%}={report.coverage_at_precision:.3f} "
            f"(thr {threshold}) ece={report.ece:.3f}"
        )
    print(f"{classifier.name:16} {'latency':13} {elapsed / max(len(test), 1) * 1000:.1f} ms/transaction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--since", type=date.fromisoformat, help="only use transactions from this date (YYYY-MM-DD)")
    parser.add_argument("--reconciled-only", action="store_true", help="only use transactions reconciled in Actual")
    parser.add_argument("--precision", type=float, default=0.95, help="target precision for auto-assignment")
    args = parser.parse_args()

    examples = load_examples(args.data, args.since, args.reconciled_only)
    train, test = time_split(examples, args.train_fraction)
    categories = sorted({e.category for e in examples})
    print(f"{len(examples)} labeled transactions, {len(categories)} categories; train {len(train)} / test {len(test)}")

    for name in args.classifiers.split(","):
        run(CLASSIFIERS[name](), train, test, categories, args.precision)


if __name__ == "__main__":
    main()
