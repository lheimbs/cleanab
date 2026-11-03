import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "experiments" / "export_actual.py"


@pytest.fixture(scope="module")
def export_script():
    spec = importlib.util.spec_from_file_location("export_actual", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_maps_ids_and_skips_transfers_and_splits(export_script, monkeypatch, tmp_path, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "apps:\n  actual:\n    actual_api_url: http://actual.local/\n    actual_api_key: k\n"
        "    actual_sync_id: sync\n    actual_account_ids: [acc]\n"
    )
    responses = {
        "/categories": [{"id": "c1", "name": "Groceries"}],
        "/payees": [{"id": "p1", "name": "Edeka"}],
        "/accounts/acc/transactions": [
            {
                "date": "2024-02-01",
                "payee": "p1",
                "notes": "Einkauf",
                "category": "c1",
                "amount": -1234,
                "cleared": True,
                "reconciled": False,
                "imported_payee": "EDEKA 123",
            },
            {"date": "2024-01-01", "payee": "p1", "category": None, "amount": -500},
            {"date": "2024-01-02", "transfer_id": "t", "amount": -100},
            {"date": "2024-01-03", "subtransactions": [{}], "amount": -100},
        ],
    }

    def fake_get(url, headers, **params):
        for suffix, data in responses.items():
            if url.endswith(suffix):
                return data
        raise AssertionError(url)

    monkeypatch.setattr(export_script, "get", fake_get)
    rows = export_script.export(config, "2020-01-01", None)

    assert [r["date"] for r in rows] == ["2024-01-01", "2024-02-01"]
    assert rows[1] == {
        "date": "2024-02-01",
        "payee": "Edeka",
        "purpose": "Einkauf",
        "category": "Groceries",
        "cleared": 1,
        "reconciled": 0,
        "imported_payee": "EDEKA 123",
        "amount": -12.34,
    }
    assert rows[0]["category"] == ""
    output = capsys.readouterr().out
    assert "skipped 1 transfers" in output and "skipped 1 split transactions" in output


def test_summarize_counts_per_month(export_script, capsys):
    export_script.summarize(
        [
            {"date": "2024-01-05", "category": "A", "reconciled": 1},
            {"date": "2024-01-06", "category": "", "reconciled": 0},
            {"date": "2024-02-01", "category": "A", "reconciled": 0},
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("2024-01") and line.split()[1:] == ["2", "1", "1", "1"] for line in lines)
    assert any(line.startswith("2024-02") and line.split()[1:] == ["1", "1", "0", "0"] for line in lines)
