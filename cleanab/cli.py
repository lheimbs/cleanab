import logging

import click
import logzero
import yaml

from .main import Cleanab
from .models.config import Config


def setup_logging(verbose: bool = False) -> None:
    """Quiet the python-fints library and set the level of cleanab's own logger."""
    logging.getLogger("fints").setLevel(logging.ERROR)
    logzero.loglevel(logging.DEBUG if verbose else logging.INFO)


class ConfigFile(click.File):
    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        value = yaml.safe_load(value)
        return Config.model_validate(value)


@click.command()
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    help=("Only fetch and clean up the transactions. None will be added in your budgeting app."),
)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    help=(
        "Save the cleaned transactions to disk. This is useful for debugging and"
        " testing. It allows you to inspect the cleaned data before it is added to your"
        " budgeting app."
    ),
)
@click.option(
    "-t",
    "--test",
    is_flag=True,
    help=(
        "Testing mode: Fetch transactions and clean them up. The raw transaction data"
        " will be written to disk, so on subsequent --test runs the local copy of the"
        " transaction data will be used. This avoids querying the bank APIs too often."
        " No transaction will be added to in your budgeting app. (implies --dry-run and"
        " --verbose)"
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show replacements made to the received data",
)
@click.option(
    "-c",
    "--config",
    "config",
    type=ConfigFile("r"),
    default="./config.yaml",
    show_default=True,
    help="Custom location of the config file.",
    metavar="configfile",
)
def cli(**kwargs):
    c = Cleanab(**kwargs)
    setup_logging(c.verbose or kwargs["config"].cleanab.debug)
    c.setup()
    c.run()
