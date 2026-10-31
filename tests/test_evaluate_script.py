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
