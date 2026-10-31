# Classifier experiments

Evaluates candidate category classifiers on your own categorized transactions.

## Getting the data out of Actual

Read-only export via the actual-http-api settings in your cleanab config. With the sops-encrypted
config from the main README:

```sh
sops exec-file ~/.config/cleanab/config.sops.yaml \
  'uv run python -m experiments.export_actual -c {} --since 2023-01-01'
```

It writes `experiments/data/transactions.csv` and prints, per month, how many transactions are
categorized, uncategorized and reconciled. Use that table to decide from when on you trust the
categories, then pass `--since YYYY-MM-DD` and `--until YYYY-MM-DD` (and optionally `--reconciled-only`) to `evaluate`.
Transfers and split transactions are skipped.

## Data

A CSV with the columns `date` (YYYY-MM-DD), `payee`, `purpose`, `category` (plus optional `reconciled`),
one row per categorized transaction. Keep it out of git (`experiments/data/` is ignored).

## Run

```sh
uv run python -m experiments.evaluate experiments/data/transactions.csv
```

The data is split chronologically (default 80/20). Results are reported for all test
transactions and separately for payees seen / not seen in training, since the unseen group
is where a model has to earn its install cost. `cov@95%` is the share of transactions that
could be auto-categorized while keeping 95% precision.
