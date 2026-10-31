"""Evaluate classifiers on a labeled export of your own categorized transactions.

CSV columns: date (YYYY-MM-DD), payee, purpose, category. See experiments/README.md.

    uv run python -m experiments.evaluate data/transactions.csv --classifiers exact-payee,char-ngram-lr
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def load_examples(
    path: Path, since: date | None = None, until: date | None = None, reconciled_only: bool = False
) -> list[Example]:
    """Load labeled rows, optionally only those within [since, until] or marked reconciled in Actual."""
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
        and (until is None or date.fromisoformat(row["date"]) <= until)
        and (not reconciled_only or row.get("reconciled") == "1")
    ]
    return sorted(examples, key=lambda e: e.date)


def time_split(examples: list[Example], train_fraction: float) -> tuple[list[Example], list[Example]]:
    """Chronological split: a random split leaks recurring payees and inflates accuracy."""
    cut = int(len(examples) * train_fraction)
    return examples[:cut], examples[cut:]


def print_report(name: str, label: str, predictions, truths, target: float) -> None:
    report = evaluate(predictions, truths, target)
    threshold = f"{report.threshold:.2f}" if report.threshold is not None else "-"
    print(
        f"{name:16} {label:13} n={report.n:5} acc={report.accuracy:.3f} "
        f"answered={report.coverage:.3f} cov@{target:.0%}={report.coverage_at_precision:.3f} "
        f"(thr {threshold}) ece={report.ece:.3f}"
    )


def show_errors(name: str, test: list[Example], predictions, limit: int) -> None:
    """Print wrong or abstained predictions, most confident mistakes first."""
    errors = [(e, p) for e, p in zip(test, predictions, strict=True) if p.category != e.category]
    errors.sort(key=lambda item: -item[1].confidence)
    print(f"{name}: {len(errors)} errors, showing {min(limit, len(errors))}")
    for example, prediction in errors[:limit]:
        print(
            f"  {prediction.confidence:.2f} {normalize_text(example.payee)[:40]:40} "
            f"true={example.category} pred={prediction.category}"
        )


def run(
    classifier: Classifier,
    train: list[Example],
    test: list[Example],
    categories: list[str],
    target: float,
    errors: int = 0,
):
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
        if idx:
            print_report(classifier.name, label, [predictions[i] for i in idx], [test[i].category for i in idx], target)
    print(f"{classifier.name:16} {'latency':13} {elapsed / max(len(test), 1) * 1000:.1f} ms/transaction")
    if errors:
        show_errors(classifier.name, test, predictions, errors)


def group_folds(examples: list[Example], folds: int) -> list[int]:
    """Assign each example to a fold so that all transactions of one payee share a fold (stable across runs)."""

    def fold_of(example: Example) -> int:
        return int(hashlib.md5(normalize_text(example.payee).encode()).hexdigest(), 16) % folds

    return [fold_of(e) for e in examples]


def run_grouped_cv(
    factory, examples: list[Example], categories: list[str], folds: int, target: float, errors: int = 0
) -> None:
    """Cross-validate with payees held out: every transaction is predicted for a payee the model has not seen.

    This is the number that matters for a model's added value, and it uses all data instead of the small
    unseen part of a chronological split.
    """
    assignment = group_folds(examples, folds)
    all_test: list[Example] = []
    predictions = []
    name = factory().name
    started = time.perf_counter()
    for fold in range(folds):
        train = [e for e, f in zip(examples, assignment, strict=True) if f != fold]
        test = [e for e, f in zip(examples, assignment, strict=True) if f == fold]
        if not test or not train:
            continue
        classifier = factory()
        classifier.fit(train)
        predictions += [classifier.predict(e, categories) for e in test]
        all_test += test
    elapsed = time.perf_counter() - started
    print_report(name, f"unseen cv{folds}", predictions, [e.category for e in all_test], target)
    print(f"{name:16} {'latency':13} {elapsed / max(len(all_test), 1) * 1000:.1f} ms/transaction (incl. fitting)")
    if errors:
        show_errors(name, all_test, predictions, errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--since", type=date.fromisoformat, help="only use transactions from this date (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="only use transactions up to this date (YYYY-MM-DD)")
    parser.add_argument("--reconciled-only", action="store_true", help="only use transactions reconciled in Actual")
    parser.add_argument(
        "--group-cv",
        type=int,
        metavar="K",
        help="instead of the chronological split, run K-fold cross-validation with payees held out",
    )
    parser.add_argument("--errors", type=int, default=0, metavar="N", help="list the N most confident mistakes")
    parser.add_argument("--precision", type=float, default=0.95, help="target precision for auto-assignment")
    args = parser.parse_args()

    examples = load_examples(args.data, args.since, args.until, args.reconciled_only)
    categories = sorted({e.category for e in examples})
    if args.group_cv:
        print(
            f"{len(examples)} labeled transactions, {len(categories)} categories; payee-grouped {args.group_cv}-fold CV"
        )
        for name in args.classifiers.split(","):
            run_grouped_cv(CLASSIFIERS[name], examples, categories, args.group_cv, args.precision, args.errors)
        return

    train, test = time_split(examples, args.train_fraction)
    print(f"{len(examples)} labeled transactions, {len(categories)} categories; train {len(train)} / test {len(test)}")

    for name in args.classifiers.split(","):
        run(CLASSIFIERS[name](), train, test, categories, args.precision, args.errors)


if __name__ == "__main__":
    main()
