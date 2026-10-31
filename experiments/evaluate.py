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


def _gliner() -> Classifier:
    from cleanab.classifier.gliner import GlinerClassifier

    return GlinerClassifier(descriptions=DESCRIPTIONS)


def _decider() -> Classifier:
    from cleanab.classifier.decider import DeciderClassifier

    return DeciderClassifier(descriptions=DESCRIPTIONS)


def _hybrid() -> Classifier:
    from cleanab.classifier.decider import DeciderClassifier
    from cleanab.classifier.hybrid import HybridClassifier

    return HybridClassifier([CharNgramClassifier(), DeciderClassifier(descriptions=DESCRIPTIONS)], "hybrid")


def _hybrid_lite() -> Classifier:
    from cleanab.classifier.hybrid import HybridClassifier

    return HybridClassifier([CharNgramClassifier()], "hybrid-lite")


def _hybrid_full() -> Classifier:
    from cleanab.classifier.decider import DeciderClassifier
    from cleanab.classifier.hybrid import HybridClassifier

    ngram = CharNgramClassifier()
    return HybridClassifier([ngram, DeciderClassifier(descriptions=DESCRIPTIONS)], "hybrid-full", known=ngram)


DESCRIPTIONS: dict[str, str] = {}

CLASSIFIERS = {
    "exact-payee": ExactPayeeClassifier,
    "char-ngram-lr": CharNgramClassifier,
    "gliner2": _gliner,
    "decider-4b": _decider,
    "hybrid": _hybrid,
    "hybrid-lite": _hybrid_lite,
    "hybrid-full": _hybrid_full,
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


def run_online(
    factory, examples: list[Example], categories: list[str], warmup: int, target: float, errors: int = 0
) -> None:
    """Replay the transactions in date order, as a real import would see them.

    Each transaction is predicted from the history before it only, then added to the history (as if you had
    confirmed the category). The first `warmup` transactions only build history. Results are split into
    first occurrences of a payee (cold start, where a model has to earn its keep) and repeats (which the
    history already covers).
    """
    classifier = factory()
    seen: set[str] = set()
    scored: list[tuple[Example, object, bool]] = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        payee = normalize_text(example.payee)
        if index >= warmup:
            scored.append((example, classifier.predict(example, categories), payee not in seen))
        seen.add(payee)
        classifier.fit(examples[: index + 1])
    elapsed = time.perf_counter() - started

    name = classifier.name
    distinct_cold = len({normalize_text(e.payee) for e, _, cold in scored if cold})
    print(f"{name}: scored {len(scored)} transactions, {sum(c for _, _, c in scored)} first occurrences")
    for label, keep in (("all", lambda c: True), ("first occurr.", lambda c: c), ("repeat", lambda c: not c)):
        chosen = [(e, p) for e, p, cold in scored if keep(cold)]
        if chosen:
            print_report(name, label, [p for _, p in chosen], [e.category for e, _ in chosen], target)
    print(f"{name:16} {'latency':13} {elapsed / max(len(scored), 1) * 1000:.1f} ms/transaction (incl. refitting)")
    if errors:
        cold = [(e, p) for e, p, is_cold in scored if is_cold]
        show_errors(name + " (first occurrences)", [e for e, _ in cold], [p for _, p in cold], errors)
    print(f"({distinct_cold} distinct first-occurrence payees)")


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
    parser.add_argument(
        "--online",
        action="store_true",
        help="replay in date order, predicting each transaction from earlier history only; "
        "reports first occurrences of a payee separately",
    )
    parser.add_argument("--warmup", type=int, default=100, help="--online: transactions used only as initial history")
    parser.add_argument(
        "--descriptions", type=Path, help="YAML/JSON file mapping category name to a one-line description"
    )
    parser.add_argument("--errors", type=int, default=0, metavar="N", help="list the N most confident mistakes")
    parser.add_argument("--precision", type=float, default=0.95, help="target precision for auto-assignment")
    args = parser.parse_args()

    if args.descriptions:
        import yaml

        DESCRIPTIONS.update(yaml.safe_load(args.descriptions.read_text(encoding="utf-8")) or {})
    examples = load_examples(args.data, args.since, args.until, args.reconciled_only)
    categories = sorted({e.category for e in examples})
    if args.online:
        print(
            f"{len(examples)} labeled transactions, {len(categories)} categories; online replay, warmup {args.warmup}"
        )
        for name in args.classifiers.split(","):
            run_online(CLASSIFIERS[name], examples, categories, args.warmup, args.precision, args.errors)
        return
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
