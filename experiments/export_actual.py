"""Export categorized transactions from Actual Budget (read-only) to a CSV for evaluate.py.

Uses the `apps.actual` section of your cleanab config (actual-http-api), so with a sops-encrypted config:

    sops exec-file ~/.config/cleanab/config.sops.yaml \\
        'uv run python -m experiments.export_actual -c {} --since 2023-01-01'

Writes experiments/data/transactions.csv and prints a per-month summary, so you can see from which
month on your transactions are categorized (and reconciled) before choosing the evaluation cutoff.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import requests
import yaml

FIELDS = ["date", "payee", "purpose", "category", "cleared", "reconciled", "imported_payee", "amount"]


def get(url: str, headers: dict, **params):
    response = requests.get(url, headers=headers, params=params, timeout=60)
    response.raise_for_status()
    return response.json()["data"]


def export(config_path: Path, since: str, until: str | None) -> list[dict]:
    config = yaml.safe_load(config_path.read_text())
    actual = config["apps"]["actual"]
    base = f"{str(actual['actual_api_url']).rstrip('/')}/budgets/{actual['actual_sync_id']}"
    headers = {"x-api-key": actual["actual_api_key"], "accept": "application/json"}
    if actual.get("actual_encryption_password"):
        headers["budget-encryption-password"] = actual["actual_encryption_password"]

    categories = {c["id"]: c["name"] for c in get(f"{base}/categories", headers)}
    payees = {p["id"]: p["name"] for p in get(f"{base}/payees", headers)}

    rows: list[dict] = []
    skipped = defaultdict(int)
    for account_id in actual["actual_account_ids"]:
        params = {"since_date": since}
        if until:
            params["until_date"] = until
        for tx in get(f"{base}/accounts/{account_id}/transactions", headers, **params):
            if tx.get("transfer_id"):
                skipped["transfers"] += 1
                continue
            if tx.get("subtransactions"):
                skipped["split transactions"] += 1
                continue
            rows.append(
                {
                    "date": tx["date"],
                    "payee": payees.get(tx.get("payee"), "") or tx.get("imported_payee") or "",
                    "purpose": tx.get("notes") or "",
                    "category": categories.get(tx.get("category"), "") if tx.get("category") else "",
                    "cleared": int(bool(tx.get("cleared"))),
                    "reconciled": int(bool(tx.get("reconciled"))),
                    "imported_payee": tx.get("imported_payee") or "",
                    "amount": tx.get("amount", 0) / 100,
                }
            )
    for reason, count in skipped.items():
        print(f"skipped {count} {reason}")
    return sorted(rows, key=lambda r: r["date"])


def summarize(rows: list[dict]) -> None:
    months: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        counts = months[row["date"][:7]]
        counts[0] += 1
        counts[1] += bool(row["category"])
        counts[2] += row["reconciled"]
    print(f"\n{'month':8} {'total':>6} {'categorized':>12} {'uncategorized':>14} {'reconciled':>11}")
    for month in sorted(months):
        total, categorized, reconciled = months[month]
        print(f"{month:8} {total:6} {categorized:12} {total - categorized:14} {reconciled:11}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", type=Path, required=True, help="cleanab config with an apps.actual section")
    parser.add_argument("-o", "--output", type=Path, default=Path("experiments/data/transactions.csv"))
    parser.add_argument("--since", default="2020-01-01", help="earliest transaction date (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="latest transaction date (YYYY-MM-DD)")
    args = parser.parse_args()

    rows = export(args.config, args.since, args.until)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} transactions to {args.output}")
    summarize(rows)


if __name__ == "__main__":
    main()
