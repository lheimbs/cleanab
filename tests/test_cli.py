import logging

import logzero
from click.testing import CliRunner

from cleanab.cli import cli, setup_logging


def test_setup_logging_levels():
    setup_logging(verbose=False)
    assert logzero.logger.level == logging.INFO
    assert logging.getLogger("fints").level == logging.ERROR
    setup_logging(verbose=True)
    assert logzero.logger.level == logging.DEBUG


def test_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
