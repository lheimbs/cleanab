# CleanAB — Clean A Budget

This is re-[inventing](https://github.com/schurig/ynab-bank-importer) [the](https://bitbucket.org/ctheune/ynab-bank-imports/src/default/) [wheel](https://github.com/bank2ynab/bank2ynab). 💁‍♀️

Import FinTS/HBCI transactions (🇩🇪 👋) into multiple apps using their APIs like

- [YNAB](https://ynab.com/)
- [Actual Budget](https://actualbudget.org/)

My rationale for creating this (instead of using an existing solution), was the poor parsing/processing/cleanup of transaction data like payee and memo in other tools.
Configuration is done in YAML and can include an arbitrary amount of replacement definitions that should be applied to the transaction data.
See [config.yaml.sample](config.yaml.sample) for example use.
