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

A CSV with the columns `date` (YYYY-MM-DD), `payee`, `purpose`, `category` (plus optional
`imported_payee` and `reconciled`), one row per categorized transaction. Keep it out of git
(`experiments/data/` is ignored).

By default the payee text is `imported_payee`, which is what cleanab sends at import time. Actual's `payee`
is the *current* payee name: renaming or merging a payee in Actual rewrites it for all past transactions,
so evaluating on it leaks later cleanup into the past and makes new payees look more familiar than they
are. `--payee-column payee` restores the old behaviour for comparison. Rows without an imported payee
(entered by hand) fall back to `payee`.

Two caveats the evaluation cannot fix: `purpose` is Actual's `notes`, which you may have edited, and some
categories may have been filled in by Actual's own payee rules rather than chosen by you.
`--reconciled-only` reduces, but does not remove, the second effect.

## Run

```sh
uv run python -m experiments.evaluate experiments/data/transactions.csv
```

This runs `exact-payee`, `char-ngram-lr` and `hybrid-lite`. The model backends (`gliner2`, `decider-4b`,
`hybrid`, `hybrid-full`) need `uv sync --group models` and download several GB of weights on first use;
pick them with `--classifiers`.

The data is split chronologically (default 80/20). Results are reported for all test
transactions and separately for payees seen / not seen in training, since the unseen group
is where a model has to earn its install cost. `cov@95%` is the share of transactions that
could be auto-categorized while keeping 95% precision. The threshold behind it is picked on the same
transactions it is scored on, so treat it (and the printed `thr`) as an upper bound, especially for small
groups; pick a production threshold on earlier data.

`char-ngram-lr` and `hybrid-lite` need scikit-learn (the `ml-lite` extra, also in the dev group).

The category list comes from the whole dataset, so zero-shot models may be offered categories created
after a transaction's date, and grouped CV trains on later transactions of other payees. Both are
deliberate: they isolate the payee effect rather than replay history (use `--online` for that).

## Unseen-payee evaluation

The chronological split leaves few transactions with payees the model has not seen. For a stable
estimate of what a model adds, hold payees out instead (all transactions of a payee end up in the same
fold) and list the most confident mistakes:

```sh
uv run python -m experiments.evaluate experiments/data/transactions.csv \
  --since 2023-01-01 --until 2024-03-31 --group-cv 5 --errors 15
```

## Replaying an import (online mode)

The most realistic test: replay transactions in date order, predicting each one from the history before
it, then adding it to the history as if you had confirmed it. First occurrences of a payee (where a model
has to help) are reported separately from repeats (which the history already covers):

```sh
uv run python -m experiments.evaluate experiments/data/transactions.csv \
  --since 2023-01-01 --until 2024-03-31 --online --warmup 100 --errors 15
```

Trained models are refit after every transaction, which gets slow for a few thousand transactions.
`--refit-every 50` refits only every 50 transactions and whenever a new payee or category appears.
