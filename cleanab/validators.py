import re
import string
from uuid import UUID

LETTERS = {ord(d): str(i) for i, d in enumerate(string.digits + string.ascii_uppercase)}


def is_iban(iban_string):
    """Validate an IBAN using the ISO 13616 mod-97 check."""
    iban = iban_string.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    number_iban = (iban[4:] + iban[:4]).translate(LETTERS)
    return int(number_iban) % 97 == 1


def is_uuid(uuid_string):
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def is_https_url(url_string):
    return url_string.startswith("https://")


def is_blz(blz_string):
    return blz_string.isdigit() and len(blz_string) == 8
