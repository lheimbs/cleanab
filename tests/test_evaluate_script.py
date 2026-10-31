import importlib.util
from datetime import date
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "experiments" / "evaluate.py"


@pytest.fixture(scope="module")
def evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    lines = ["date,payee,purpose,category"] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_examples_sorts_by_date_and_skips_unlabeled(evaluate_script, tmp_path):
    csv_path = tmp_path / "t.csv"
    write_csv(
        csv_path,
        [
            ("2024-02-01", "B", "", "Cat"),
            ("2024-01-01", "A", "", "Cat"),
            ("2024-03-01", "C", "", ""),
        ],
    )
    examples = evaluate_script.load_examples(csv_path)
    assert [e.payee for e in examples] == ["A", "B"]
    assert examples[0].date == date(2024, 1, 1)


def test_time_split_is_chronological(evaluate_script):
    from cleanab.classifier.base import Example

    examples = [Example(str(i), category="x", date=date(2024, 1, i + 1)) for i in range(10)]
    train, test = evaluate_script.time_split(examples, 0.8)
    assert len(train) == 8 and len(test) == 2
    assert max(e.date for e in train) < min(e.date for e in test)


def test_end_to_end_run_prints_report(evaluate_script, tmp_path, capsys):
    rows = [
        (f"2024-01-{i + 1:02d}", "Edeka" if i % 2 else "Netflix", "", "Food" if i % 2 else "Subs") for i in range(20)
    ]
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, rows)
    examples = evaluate_script.load_examples(csv_path)
    train, test = evaluate_script.time_split(examples, 0.8)
    evaluate_script.run(evaluate_script.ExactPayeeClassifier(), train, test, ["Food", "Subs"], 0.95)
    output = capsys.readouterr().out
    assert "exact-payee" in output
    assert "seen payee" in output


def test_load_examples_filters_since_and_reconciled(evaluate_script, tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "date,payee,purpose,category,reconciled\n"
        "2023-01-01,Old,,Cat,1\n"
        "2024-01-01,NewReconciled,,Cat,1\n"
        "2024-02-01,NewOpen,,Cat,0\n",
        encoding="utf-8",
    )
    since = date(2024, 1, 1)
    assert [e.payee for e in evaluate_script.load_examples(csv_path, since=since)] == ["NewReconciled", "NewOpen"]
    assert [e.payee for e in evaluate_script.load_examples(csv_path, reconciled_only=True)] == ["Old", "NewReconciled"]


def test_load_examples_filters_until(evaluate_script, tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "date,payee,purpose,category\n2023-12-31,A,,Cat\n2024-03-31,B,,Cat\n2024-04-01,C,,Cat\n",
        encoding="utf-8",
    )
    examples = evaluate_script.load_examples(csv_path, since=date(2023, 1, 1), until=date(2024, 3, 31))
    assert [e.payee for e in examples] == ["A", "B"]


def test_group_folds_keep_payees_together(evaluate_script):
    from cleanab.classifier.base import Example

    examples = [Example(payee, category="x") for payee in ["EDEKA", "edeka", "Netflix", "Spotify"] * 3]
    folds = evaluate_script.group_folds(examples, 3)
    by_payee: dict[str, set[int]] = {}
    for example, fold in zip(examples, folds, strict=True):
        by_payee.setdefault(example.payee.lower(), set()).add(fold)
    assert all(len(f) == 1 for f in by_payee.values())


def test_grouped_cv_only_predicts_unseen_payees(evaluate_script, capsys):
    from cleanab.classifier.base import Example

    examples = [Example(f"shop{i % 10}", category="A" if i % 2 else "B", date=date(2024, 1, 1)) for i in range(60)]
    evaluate_script.run_grouped_cv(evaluate_script.ExactPayeeClassifier, examples, ["A", "B"], 5, 0.95, errors=2)
    output = capsys.readouterr().out
    # exact-payee cannot know any held-out payee, so it must abstain on everything
    assert "answered=0.000" in output
    assert "unseen cv5" in output


def test_online_replay_separates_first_occurrences(evaluate_script, capsys):
    from cleanab.classifier.base import Example

    payees = ["Edeka", "Netflix", "Edeka", "Netflix", "Edeka", "Rewe", "Netflix", "Edeka"]
    category = {"Edeka": "Food", "Rewe": "Food", "Netflix": "Subs"}
    examples = [Example(p, category=category[p], date=date(2024, 1, i + 1)) for i, p in enumerate(payees)]
    evaluate_script.run_online(evaluate_script.ExactPayeeClassifier, examples, ["Food", "Subs"], warmup=2, target=0.9)
    output = capsys.readouterr().out
    assert "scored 6 transactions, 1 first occurrences" in output  # only Rewe is new after the warmup
    assert "first occurr." in output and "repeat" in output
