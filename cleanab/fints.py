from __future__ import annotations

import sys
from multiprocessing import Lock

from fints.client import FinTS3PinTanClient, FinTSClientError, NeedTANResponse
from fints.hhd.flicker import terminal_flicker_unix
from logzero import logger

from .models.enums import AccountType
from .tan_display import ImageDecodeError, TanDisplayMode, show_image

lock = Lock()
# Logged-in clients per bank login, so each login (and its TAN) happens once per run. Bounded by the configured
# bank logins; failed logins are not stored.
_clients: dict[tuple, tuple] = {}


class CleanabFinTSClient(FinTS3PinTanClient):
    """A FinTS client that knows how to show TAN images, so the mode needn't be threaded through every call."""

    def __init__(self, *args, tan_display: TanDisplayMode, **kwargs):
        super().__init__(*args, **kwargs)
        self.tan_display: TanDisplayMode = tan_display


def bootstrap_fints(fints: FinTS3PinTanClient):
    # Fetch available TAN mechanisms by the bank, if we don't know it already.
    # If the client was created with cached data, the function is already set.
    if not fints.get_current_tan_mechanism():
        fints.fetch_tan_mechanisms()
        mechanisms = list(fints.get_tan_mechanisms().items())
        if len(mechanisms) > 1:
            logger.info("Multiple tan mechanisms available. Which one do you prefer?")
            for i, m in enumerate(mechanisms):
                logger.info(f"{i}: Function {m[1].security_function}: {m[1].name}")
            choice = input("Choice: ").strip()
            fints.set_tan_mechanism(mechanisms[int(choice)][0])

    if fints.is_tan_media_required() and not fints.selected_tan_medium:
        logger.info("We need the name of the TAN medium, let's fetch them from the bank")
        tan_media = fints.get_tan_media()
        if not tan_media:
            logger.error("No TAN media available")
            return
        elif len(tan_media[1]) == 1:
            fints.set_tan_medium(tan_media[1][0])
        else:
            logger.info("Multiple tan media available. Which one do you prefer?")
            for i, tan_medium in enumerate(tan_media[1]):
                logger.info(
                    f"{i}: Medium {tan_medium.tan_medium_name}: Phone no. {tan_medium.mobile_number_masked}, "
                    f"Last used {tan_medium.last_use}"
                )
            choice = input("Choice: ").strip()
            fints.set_tan_medium(tan_media[1][int(choice)])


def show_tan_image(data: bytes, mode: TanDisplayMode) -> None:
    """Show the image in the resolved ``mode``, falling back to blocks on output problems.

    Never let a display problem keep the user from entering the TAN.
    """
    if mode != "window" and not sys.stdout.isatty():
        logger.warning(
            "Not drawing the TAN image because stdout is not a terminal; "
            "run cleanab interactively or set cleanab.tan_display: window"
        )
        return
    for attempt in dict.fromkeys((mode, "blocks")):
        try:
            show_image(data, attempt)
            return
        except ImageDecodeError as e:
            # Every mode decodes the same way, so falling back can't help.
            logger.error("%s", e)
            return
        except Exception:
            logger.exception("Could not show the TAN image with tan_display=%s", attempt)
    logger.error("Could not show the TAN image; try another cleanab.tan_display mode")


def handle_tan_response(fints: CleanabFinTSClient, tan_response: NeedTANResponse) -> list:
    logger.info(f"TAN needed: {tan_response.challenge}")

    if tan_response.challenge_hhduc:
        logger.info("Please use your TAN generator to generate a TAN.")
        try:
            terminal_flicker_unix(tan_response.challenge_hhduc)
        except KeyboardInterrupt:
            pass
    elif tan_response.challenge_matrix:
        logger.info("Please use your bank's app to scan the QR code.")
        show_tan_image(tan_response.challenge_matrix[1], fints.tan_display)
    else:
        logger.info(tan_response.challenge_html)

    tan: str = input("Please enter the TAN: ")

    try:
        response = fints.send_tan(tan_response, tan)
        if isinstance(response, NeedTANResponse):
            logger.error("TAN was not accepted, please try again.")
            return handle_tan_response(fints, response)
        else:
            logger.info("TAN accepted, proceeding with the request.")
            return response
    except FinTSClientError as e:
        logger.error(f"Failed to send TAN: {e}")
        return []


def retrieve_transactions(sepa_account, fints: CleanabFinTSClient, *, start_date, end_date):
    with fints:
        bootstrap_fints(fints)
        result = fints.get_transactions(sepa_account, start_date=start_date, end_date=end_date)
        if isinstance(result, NeedTANResponse):
            result = handle_tan_response(fints, result)
    return [t.data for t in result]


def retrieve_holdings(sepa_account, fints: CleanabFinTSClient):
    with lock:
        bootstrap_fints(fints)
        holdings = fints.get_holdings(sepa_account)
        if isinstance(holdings, NeedTANResponse):
            holdings = handle_tan_response(fints, holdings)
    return [{"total_value": h.total_value} for h in holdings]


def get_fints_client(blz, username, password, endpoint, product_id, *, tan_display: TanDisplayMode):
    """The logged-in client for a bank login; ``tan_display`` is not part of the cache key."""
    key = (blz, username, password, endpoint, product_id)
    if key not in _clients:
        # Failed logins (e.g. a rejected TAN) are remembered too: logging in again for every other account at this
        # bank would repeat the PIN/TAN attempt, and banks lock online banking after a few failures.
        _clients[key] = _connect(*key, tan_display=tan_display)
    elif not _clients[key][1]:
        logger.warning("Skipping login for %s at %s, it already failed in this run", username, endpoint)
    return _clients[key]


def _connect(blz, username, password, endpoint, product_id, *, tan_display: TanDisplayMode):
    logger.info(
        "Retrieving SEPA accounts for %s from %s (product id=%s)",
        username,
        endpoint,
        product_id,
    )
    fints = CleanabFinTSClient(
        bank_identifier=blz,
        user_id=username,
        pin=password,
        server=endpoint,
        product_id=product_id,
        tan_display=tan_display,
    )
    with lock:
        with fints:
            # Bootstrap the client to set up TAN mechanisms
            bootstrap_fints(fints)

            # Handle potential TAN requirement for dialog initialization
            # fints never clears init_tan_response itself, so handle it once and reset it
            if isinstance(fints.init_tan_response, NeedTANResponse):
                handle_tan_response(fints, fints.init_tan_response)
                fints.init_tan_response = None

            # Get SEPA accounts and handle potential TAN requirement
            sepa_accounts = fints.get_sepa_accounts()
            if isinstance(sepa_accounts, NeedTANResponse):
                sepa_accounts = handle_tan_response(fints, sepa_accounts)

    return fints, sepa_accounts


def process_fints_account(account, earliest, latest, product_id, *, tan_display: TanDisplayMode) -> list:
    fints, sepa_accounts = get_fints_client(
        account.fints_blz,
        account.fints_username,
        account.fints_password,
        account.fints_endpoint,
        product_id,
        tan_display=tan_display,
    )
    accounts = [acc for acc in sepa_accounts if acc.iban == account.iban]
    if not accounts:
        logger.error(f"Account for IBAN {account.iban} not found")
        return []
    sepa_account = accounts[0]

    if account.account_type == AccountType.HOLDING:
        transactions = retrieve_holdings(sepa_account, fints)
    else:
        transactions = retrieve_transactions(sepa_account, fints, start_date=earliest, end_date=latest)

    return transactions
