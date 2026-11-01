# CleanAB — Clean A Budget

This is re-[inventing](https://github.com/schurig/ynab-bank-importer) [the](https://bitbucket.org/ctheune/ynab-bank-imports/src/default/) [wheel](https://github.com/bank2ynab/bank2ynab). 💁‍♀️

Import FinTS/HBCI transactions (🇩🇪 👋) into multiple apps using their APIs like

- [YNAB](https://ynab.com/)
- [Actual Budget](https://actualbudget.org/)
- [Firefly III](https://www.firefly-iii.org/) (via the Data Importer)

My rationale for creating this (instead of using an existing solution), was the poor parsing/processing/cleanup of transaction data like payee and memo in other tools.
Configuration is done in YAML and can include an arbitrary amount of replacement definitions that should be applied to the transaction data.
See [config.yaml.sample](config.yaml.sample) for example use.

## Installation

```sh
uv sync
cp config.yaml.sample config.yaml  # then edit
uv run python -m cleanab --help
```

Or with Docker (mount your config into `/app`):

```sh
docker build -t cleanab .
docker run --rm -it -v "$PWD/config.yaml:/app/config.yaml:ro" cleanab --dry-run
```

`-it` is needed because the bank may ask for a TAN on the terminal.

## Usage

```sh
python -m cleanab [OPTIONS]
```

| Option | Description |
| --- | --- |
| `-n`, `--dry-run` | Fetch and clean transactions, but don't create anything in your apps. |
| `-s`, `--save` | Write the cleaned transactions to the cache directory for inspection. |
| `-t`, `--test` | Cache raw bank data on disk and reuse it on later runs; implies `--dry-run` and `--verbose`. |
| `-v`, `--verbose` | Show the replacements made to the received data. |
| `-c`, `--config` | Path to the config file (default `./config.yaml`). |

Use `--test` while tuning your `replacements`, so the bank is only queried once.

## photoTAN / QR images over SSH

If your bank asks you to scan a photoTAN or QR image, cleanab draws it into the terminal instead of opening an image
viewer, so it also works over SSH. It picks the best method for your terminal:

| `cleanab.tan_display` | Used when | Notes |
| --- | --- | --- |
| `kitty` | kitty, Ghostty | [kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/), sharpest |
| `iterm` | iTerm2, WezTerm | inline images |
| `blocks` | everything else, tmux/screen, SSH | truecolor half-blocks, shrunk to fit the terminal; give it ~60 columns and ~30 rows so QR codes stay scannable |
| `window` | only if you set it | opens the system image viewer on the machine cleanab runs on |
| `auto` (default) | | choose from the environment |

Force a mode with `cleanab: {tan_display: blocks}` in the config or `CLEANAB_TAN_DISPLAY=blocks` (an invalid value is ignored with a warning). If a mode fails, cleanab falls back to `blocks` and still asks for the TAN. Inside tmux the
graphics protocols need `allow-passthrough`, so `auto` uses `blocks` there. Zoom out the terminal font if a QR code is too
large to fit on screen, or scan it from a distance.

## Security note

`config.yaml` contains your FinTS credentials in plaintext. Keep it out of version control (it is gitignored) and restrict its file permissions.
Holding (depot) accounts are not supported yet and are skipped.

### Tip: encrypting your config with sops

To keep your credentials encrypted at rest, you can use [sops](https://github.com/getsops/sops).
Keep the encrypted config outside this repository, e.g. in `~/.config/cleanab/`, together with a `.sops.yaml` that encrypts only the secret fields (so your `replacements` stay readable):

```yaml
# ~/.config/cleanab/.sops.yaml
creation_rules:
  - path_regex: config\.sops\.yaml$
    pgp: "<your GPG key fingerprint>"  # or `age: <recipient>`
    encrypted_regex: "^(access_token|budget_id|cash_account_id|auto_import_secret|personal_access_token|actual_api_key|actual_sync_id|fints_username|fints_password|fints_product_id|iban)$"
```

```sh
cd ~/.config/cleanab
cp /path/to/config.yaml config.sops.yaml
sops encrypt -i config.sops.yaml   # encrypt in place
sops edit config.sops.yaml         # edit later
```

Run cleanab with the decrypted config passed via a temporary FIFO, so plaintext never touches the disk:

```sh
sops exec-file ~/.config/cleanab/config.sops.yaml 'uv run python -m cleanab -c {}'

# or with Docker (--no-fifo uses a temp file, which is removed afterwards)
sops exec-file --no-fifo ~/.config/cleanab/config.sops.yaml \
  'docker run --rm -it -v {}:/app/config.yaml:ro cleanab'
```

Don't pipe the decrypted config via stdin (`-c -`): the bank may need stdin to ask for a TAN.

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format .
```
