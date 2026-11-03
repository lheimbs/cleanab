"""Evaluate classifiers on a labeled export of your own categorized transactions.

CSV columns: date (YYYY-MM-DD), payee, purpose, category (optional: imported_payee, reconciled).
See experiments/README.md.

    uv run python -m experiments.evaluate experiments/data/transactions.csv --classifiers exact-payee,char-ngram-lr
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import date
from functools import cache
from pathlib import Path

from cleanab.classifier.base import Classifier, Example
from cleanab.classifier.baselines import CharNgramClassifier, ExactPayeeClassifier
from cleanab.classifier.metrics import evaluate
from cleanab.classifier.normalize import normalize_text


@cache
def _gliner() -> Classifier:
    """One shared instance: GLiNER ignores history, so CV folds can reuse the loaded checkpoint."""
    from cleanab.classifier.gliner import GlinerClassifier

    return GlinerClassifier(descriptions=DESCRIPTIONS)


@cache
def _decider() -> Classifier:
    """One shared instance: the decider ignores history, so CV folds and hybrids can reuse the loaded model."""
    from cleanab.classifier.decider import DeciderClassifier

    return DeciderClassifier(descriptions=DESCRIPTIONS)


def _hybrid() -> Classifier:
    from cleanab.classifier.hybrid import HybridClassifier

    return HybridClassifier([CharNgramClassifier(), _decider()], "hybrid")


def _hybrid_lite() -> Classifier:
    from cleanab.classifier.hybrid import HybridClassifier

    return HybridClassifier([CharNgramClassifier()], "hybrid-lite")


def _hybrid_full() -> Classifier:
    from cleanab.classifier.hybrid import HybridClassifier

    ngram = CharNgramClassifier()
    return HybridClassifier([ngram, _decider()], "hybrid-full", known=ngram)


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
# Classifiers that need neither the `models` group nor a model download.
DEFAULT_CLASSIFIERS = "exact-payee,char-ngram-lr,hybrid-lite"
PAYEE_COLUMNS = ("imported_payee", "payee")


def positive_int(minimum: int):
    def parse(value: str) -> int:
        number = int(value)
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return number

    return parse


def fraction(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def precision(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be a fraction in (0, 1], e.g. 0.95")
    return number


def parse_classifiers(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in CLASSIFIERS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown classifier(s) {', '.join(unknown) or '(none)'}; choose from {', '.join(CLASSIFIERS)}"
        )
    return names


def payee_of(row: dict, column: str) -> str:
    """The payee text to classify.

    `imported_payee` is what cleanab sends at import time. Actual's `payee` is the current payee name, which
    reflects later renames and merges, so evaluating on it leaks future cleanup into past transactions.
    Rows without an imported payee (entered by hand in Actual) fall back to `payee`.
    """
    if column == "imported_payee":
        return row.get("imported_payee") or row["payee"]
    return row["payee"]


def load_examples(
    path: Path,
    since: date | None = None,
    until: date | None = None,
    reconciled_only: bool = False,
    payee_column: str = "imported_payee",
) -> list[Example]:
    """Load labeled rows, optionally only those within [since, until] or marked reconciled in Actual."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if payee_column == "imported_payee" and "imported_payee" not in (reader.fieldnames or []):
        print(
            "warning: no imported_payee column, using Actual's current payee names (renames and merges leak "
            "into the evaluation); re-export with experiments.export_actual",
            file=sys.stderr,
        )
    if payee_column == "imported_payee" and "imported_payee" in (reader.fieldnames or []):
        fallbacks = sum(1 for row in rows if row.get("category") and not row.get("imported_payee"))
        if fallbacks:
            print(
                f"note: {fallbacks} labeled rows have no imported_payee and use Actual's current payee name",
                file=sys.stderr,
            )
    examples = [
        Example(
            payee=payee_of(row, payee_column),
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


def payee_key(example: Example) -> str:
    """What counts as "the same payee" for seen/unseen splits and folds.

    Transactions without a usable payee (e.g. cleanab's "Unnamed" for fees and interest) recur with the same
    purpose text, which the text models read; keying them by purpose keeps a monthly fee from counting as a
    new payee every month. Returns "" if neither says anything; such rows are unrelated to each other.
    """
    payee = normalize_text(example.payee)
    if payee:
        return payee
    purpose = normalize_text(example.purpose)
    return f"purpose: {purpose}" if purpose else ""


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
    # An empty key says nothing about the payee, so never count it as seen.
    seen_payees = {payee_key(e) for e in train} - {""}
    started = time.perf_counter()
    predictions = [classifier.predict(e, categories) for e in test]
    elapsed = time.perf_counter() - started

    groups = {
        "all": [True] * len(test),
        "seen payee": [payee_key(e) in seen_payees for e in test],
        "unseen payee": [payee_key(e) not in seen_payees for e in test],
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

    def fold_of(index: int, example: Example) -> int:
        # Payees that normalize to "" are unrelated to each other; spread them instead of piling them in one fold.
        key = payee_key(example) or f"\0{index}"
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % folds

    return [fold_of(i, e) for i, e in enumerate(examples)]


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
    factory,
    examples: list[Example],
    categories: list[str],
    warmup: int,
    target: float,
    errors: int = 0,
    refit_every: int = 1,
) -> None:
    """Replay the transactions in date order, as a real import would see them.

    Each transaction is predicted from the history before it only, then added to the history (as if you had
    confirmed the category). The first `warmup` transactions only build history. Results are split into
    first occurrences of a payee (cold start, where a model has to earn its keep) and repeats (which the
    history already covers).

    With `refit_every` > 1 the classifier is refit only every that many transactions (and whenever a payee
    or category shows up for the first time), which is much faster for trained models; in between, it
    predicts from slightly older history.
    """
    classifier = factory()
    seen: set[str] = set()
    seen_categories: set[str] = set()
    scored: list[tuple[Example, object, bool]] = []
    started = time.perf_counter()
    for index, example in enumerate(examples):
        payee = payee_key(example)
        if index >= warmup:
            # Only categories that existed by then: a real import cannot offer one created months later.
            offered = [c for c in categories if c in seen_categories]
            scored.append((example, classifier.predict(example, offered), payee not in seen))
        new = payee not in seen or example.category not in seen_categories
        if payee:  # payees that normalize to "" are unrelated, so each one counts as a first occurrence
            seen.add(payee)
        seen_categories.add(example.category)
        if new or (index + 1) % refit_every == 0:
            classifier.fit(examples[: index + 1])
    elapsed = time.perf_counter() - started

    name = classifier.name
    cold_keys = [payee_key(e) for e, _, cold in scored if cold]
    distinct_cold = len(set(cold_keys) - {""}) + cold_keys.count("")  # empty keys are separate payees
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
    parser.add_argument(
        "--classifiers",
        type=parse_classifiers,
        default=DEFAULT_CLASSIFIERS,
        help=f"comma-separated, from {', '.join(CLASSIFIERS)} (default: {DEFAULT_CLASSIFIERS}; "
        "gliner2, decider-4b and hybrid/hybrid-full need `uv sync --group models` and download model weights)",
    )
    parser.add_argument(
        "--payee-column",
        choices=PAYEE_COLUMNS,
        default="imported_payee",
        help="imported_payee (default) is what cleanab sees at import; payee is Actual's current, possibly "
        "renamed or merged payee and leaks later cleanup into the evaluation",
    )
    parser.add_argument(
        "--train-fraction",
        type=fraction,
        help="chronological split (0 < x < 1, default 0.8); not with --online/--group-cv",
    )
    parser.add_argument("--since", type=date.fromisoformat, help="only use transactions from this date (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="only use transactions up to this date (YYYY-MM-DD)")
    parser.add_argument("--reconciled-only", action="store_true", help="only use transactions reconciled in Actual")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--group-cv",
        type=positive_int(2),
        metavar="K",
        help="instead of the chronological split, run K-fold cross-validation with payees held out",
    )
    mode.add_argument(
        "--online",
        action="store_true",
        help="replay in date order, predicting each transaction from earlier history only; "
        "reports first occurrences of a payee separately",
    )
    parser.add_argument(
        "--warmup", type=positive_int(0), help="--online: transactions used only as initial history (default 100)"
    )
    parser.add_argument(
        "--refit-every",
        type=positive_int(1),
        metavar="N",
        help="--online: refit only every N transactions (and on new payees/categories); faster, slightly stale",
    )
    parser.add_argument(
        "--descriptions", type=Path, help="YAML/JSON file mapping category name to a one-line description"
    )
    parser.add_argument(
        "--errors", type=positive_int(0), default=0, metavar="N", help="list the N most confident mistakes"
    )
    parser.add_argument("--precision", type=precision, default=0.95, help="target precision for auto-assignment")
    args = parser.parse_args()
    if not args.online and (args.warmup is not None or args.refit_every is not None):
        parser.error("--warmup and --refit-every only apply to --online")
    if (args.online or args.group_cv) and args.train_fraction is not None:
        parser.error("--train-fraction only applies to the chronological split")
    warmup = 100 if args.warmup is None else args.warmup
    refit_every = args.refit_every or 1
    train_fraction = args.train_fraction or 0.8

    if args.descriptions:
        import yaml

        descriptions = yaml.safe_load(args.descriptions.read_text(encoding="utf-8")) or {}
        if not isinstance(descriptions, dict):
            parser.error("--descriptions must map category names to descriptions")
        DESCRIPTIONS.update({str(k): str(v) for k, v in descriptions.items()})
    examples = load_examples(args.data, args.since, args.until, args.reconciled_only, args.payee_column)
    categories = sorted({e.category for e in examples})
    if args.online:
        if warmup >= len(examples):
            parser.error(f"--warmup {warmup} leaves nothing to score ({len(examples)} labeled transactions)")
        print(f"{len(examples)} labeled transactions, {len(categories)} categories; online replay, warmup {warmup}")
        for name in args.classifiers:
            run_online(CLASSIFIERS[name], examples, categories, warmup, args.precision, args.errors, refit_every)
        return
    if args.group_cv:
        print(
            f"{len(examples)} labeled transactions, {len(categories)} categories; payee-grouped {args.group_cv}-fold CV"
        )
        for name in args.classifiers:
            run_grouped_cv(CLASSIFIERS[name], examples, categories, args.group_cv, args.precision, args.errors)
        return

    train, test = time_split(examples, train_fraction)
    print(f"{len(examples)} labeled transactions, {len(categories)} categories; train {len(train)} / test {len(test)}")

    for name in args.classifiers:
        run(CLASSIFIERS[name](), train, test, categories, args.precision, args.errors)


if __name__ == "__main__":
    main()
