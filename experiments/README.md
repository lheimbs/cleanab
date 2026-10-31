# Classifier experiments

Evaluates candidate category classifiers on your own categorized transactions.

## Data

A CSV with the columns `date` (YYYY-MM-DD), `payee`, `purpose`, `category`, one row per
categorized transaction. Keep it out of git (`experiments/data/` is ignored).

## Run

```sh
uv run python -m experiments.evaluate experiments/data/transactions.csv
```

The data is split chronologically (default 80/20). Results are reported for all test
transactions and separately for payees seen / not seen in training, since the unseen group
is where a model has to earn its install cost. `cov@95%` is the share of transactions that
could be auto-categorized while keeping 95% precision.
