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


def test_load_examples_prefers_imported_payee(evaluate_script, tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "date,payee,purpose,category,imported_payee\n"
        "2024-01-01,REWE,,Food,REWE MARKT GMBH 1234\n"
        "2024-01-02,Cash,,Food,\n",
        encoding="utf-8",
    )
    assert [e.payee for e in evaluate_script.load_examples(csv_path)] == ["REWE MARKT GMBH 1234", "Cash"]
    by_payee = evaluate_script.load_examples(csv_path, payee_column="payee")
    assert [e.payee for e in by_payee] == ["REWE", "Cash"]


def test_load_examples_without_imported_payee_column_uses_payee(evaluate_script, tmp_path):
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, [("2024-01-01", "A", "", "Cat")])
    assert [e.payee for e in evaluate_script.load_examples(csv_path)] == ["A"]


def test_parse_classifiers_rejects_unknown_names(evaluate_script):
    import argparse

    assert evaluate_script.parse_classifiers("exact-payee, char-ngram-lr") == ["exact-payee", "char-ngram-lr"]
    with pytest.raises(argparse.ArgumentTypeError, match="nope"):
        evaluate_script.parse_classifiers("exact-payee,nope")


def test_default_classifiers_need_no_model_download(evaluate_script):
    names = evaluate_script.parse_classifiers(evaluate_script.DEFAULT_CLASSIFIERS)
    assert not {"gliner2", "decider-4b", "hybrid", "hybrid-full"} & set(names)


def test_group_folds_spread_empty_payees(evaluate_script):
    from cleanab.classifier.base import Example

    examples = [Example("12345", category="x") for _ in range(30)]  # all normalize to ""
    assert len(set(evaluate_script.group_folds(examples, 5))) > 1


def test_online_refit_every_still_refits_on_new_payees(evaluate_script, capsys):
    from cleanab.classifier.base import Example

    fits = []

    class Counting(evaluate_script.ExactPayeeClassifier):
        def fit(self, history):
            fits.append(len(history))
            super().fit(history)

    payees = ["Edeka", "Edeka", "Edeka", "Netflix", "Edeka", "Netflix", "Edeka", "Edeka"]
    category = {"Edeka": "Food", "Netflix": "Subs"}
    examples = [Example(p, category=category[p], date=date(2024, 1, i + 1)) for i, p in enumerate(payees)]
    evaluate_script.run_online(Counting, examples, ["Food", "Subs"], warmup=0, target=0.9, refit_every=4)
    assert fits == [1, 4, 8]  # first Edeka, first Netflix (also the 4th), then every 4th
    assert "repeat" in capsys.readouterr().out


def test_missing_imported_payee_column_warns(evaluate_script, tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    write_csv(csv_path, [("2024-01-01", "A", "", "Cat")])
    evaluate_script.load_examples(csv_path)
    assert "no imported_payee column" in capsys.readouterr().err


def test_online_treats_empty_payees_as_first_occurrences(evaluate_script, capsys):
    from cleanab.classifier.base import Example

    examples = [Example("VISA", category="Food", date=date(2024, 1, i + 1)) for i in range(4)]
    evaluate_script.run_online(evaluate_script.ExactPayeeClassifier, examples, ["Food"], warmup=0, target=0.9)
    assert "scored 4 transactions, 4 first occurrences" in capsys.readouterr().out


def test_option_validation(evaluate_script):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        evaluate_script.positive_int(2)("1")
    assert evaluate_script.positive_int(1)("3") == 3


def test_rows_without_imported_payee_are_counted(evaluate_script, tmp_path, capsys):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(
        "date,payee,purpose,category,imported_payee\n2024-01-01,A,,Cat,A RAW\n2024-01-02,B,,Cat,\n",
        encoding="utf-8",
    )
    evaluate_script.load_examples(csv_path)
    assert "1 labeled rows have no imported_payee" in capsys.readouterr().err


def test_fraction_validation(evaluate_script):
    import argparse

    assert evaluate_script.fraction("0.8") == 0.8
    for bad in ("0", "1", "-0.2"):
        with pytest.raises(argparse.ArgumentTypeError):
            evaluate_script.fraction(bad)
