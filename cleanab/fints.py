from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from multiprocessing import Lock
import time

from fints.client import FinTS3PinTanClient, NeedTANResponse
from fints.hhd.flicker import terminal_flicker_unix
from logzero import logger

from PIL import Image

from .models.enums import AccountType

lock = Lock()


def bootstrap_fints(fints: FinTS3PinTanClient):
    # Fetch available TAN mechanisms by the bank, if we don't know it already.
    # If the client was created with cached data, the function is already set.
    if not fints.get_current_tan_mechanism():
        fints.fetch_tan_mechanisms()
        mechanisms = list(fints.get_tan_mechanisms().items())
        if len(mechanisms) > 1:
            logger.info("Multiple tan mechanisms available. Which one do you prefer?")
            for i, m in enumerate(mechanisms):
                logger.info(i, "Function {p.security_function}: {p.name}".format(p=m[1]))
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
                logger.info(i,
                            "Medium {p.tan_medium_name}: Phone no. {p.mobile_number_masked}, Last used {p.last_use}".format(
                                p=tan_medium))
            choice = input("Choice: ").strip()
            fints.set_tan_medium(tan_media[1][int(choice)])


def handle_tan_response(fints: FinTS3PinTanClient, tan_response: NeedTANResponse) -> list:
    logger.info(f"TAN needed: {tan_response.challenge}")

    if tan_response.challenge_hhduc:
        logger.info("Please use your TAN generator to generate a TAN.")
        try:
            terminal_flicker_unix(tan_response.challenge_hhduc)
        except KeyboardInterrupt:
            pass
    elif tan_response.challenge_matrix:
        logger.info("Please use your bank's app to scan the QR code.")
        image_bytes = BytesIO(tan_response.challenge_matrix[1])
        Image.open(image_bytes).show()
        # Sleep a bit to give whatever application PIL uses to display the image time to start
        time.sleep(5)
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
    except Exception as e:
        logger.error(f"Failed to send TAN: {e}")
        return []


def retrieve_transactions(
    sepa_account, fints: FinTS3PinTanClient, *, start_date, end_date
):
    with fints:
        bootstrap_fints(fints)
        result = fints.get_transactions(
            sepa_account, start_date=start_date, end_date=end_date
        )
        if isinstance(result, NeedTANResponse):
            result = handle_tan_response(fints, result)
    return [t.data for t in result]


def retrieve_holdings(sepa_account, fints: FinTS3PinTanClient):
    with lock:
        bootstrap_fints(fints)
        holdings = fints.get_holdings(sepa_account)
        if isinstance(holdings, NeedTANResponse):
            holdings = handle_tan_response(fints, holdings)
    return [{"total_value": h.total_value} for h in holdings]


@lru_cache(maxsize=8)
def get_fints_client(blz, username, password, endpoint, product_id):
    logger.info("Retrieving SEPA accounts for %s from %s (product id=%s)", username, endpoint, product_id)
    fints = FinTS3PinTanClient(bank_identifier=blz, user_id=username, pin=password, server=endpoint, product_id=product_id)
    with lock:
        with fints:
            # Bootstrap the client to set up TAN mechanisms
            bootstrap_fints(fints)
            
            # Handle potential TAN requirement for dialog initialization
            while isinstance(fints.init_tan_response, NeedTANResponse):
                handle_tan_response(fints, fints.init_tan_response)
            
            # Get SEPA accounts and handle potential TAN requirement
            sepa_accounts = fints.get_sepa_accounts()
            while isinstance(sepa_accounts, NeedTANResponse):
                sepa_accounts = handle_tan_response(fints, sepa_accounts)

    return fints, sepa_accounts


def process_fints_account(account, earliest, latest, product_id) -> list:
    fints, sepa_accounts = get_fints_client(
        account.fints_blz,
        account.fints_username,
        account.fints_password,
        account.fints_endpoint,
        product_id,
    )
    accounts = [acc for acc in sepa_accounts if acc.iban == account.iban]
    if not accounts:
        logger.error(f"Account for IBAN {account.iban} not found")
        return []
    sepa_account = accounts[0]

    if account.account_type == AccountType.HOLDING:
        transactions = retrieve_holdings(sepa_account, fints)
    else:
        transactions = retrieve_transactions(
            sepa_account, fints, start_date=earliest, end_date=latest
        )

    return transactions
