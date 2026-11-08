"""Stage 0: strip bank boilerplate from German SEPA/card text so the merchant is what is left."""

from __future__ import annotations

import re

# SEPA purpose tags such as "SVWZ+", "EREF+<ref>", "MREF+<ref>", "CRED+<id>", "KREF+<ref>".
# The reference tags carry opaque ids, so the value is dropped; SVWZ is the actual purpose text and
# ABWA/ABWE (ultimate debtor/creditor) carry a name, so only their tag goes.
_REFERENCE_TAGS = re.compile(r"\b(?:EREF|MREF|CRED|KREF|BREF|OAMT|COAM)\+\S*", re.IGNORECASE)
# Only with "+"/":" so that names such as "Bicycle", "Ibanez" or "Bic GmbH" stay intact.
_PLAIN_TAGS = re.compile(r"\b(?:SVWZ|ABWA|ABWE|IBAN|BIC)[+:]\s*", re.IGNORECASE)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4}\b", re.IGNORECASE)
# The account part of real IBANs is mostly digits; this keeps "KD12 STADTWERKE MUENCHEN" from looking like one.
_IBAN_MIN_DIGITS = 10
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.(?:\d{2}|\d{4})?(?=\D|$)")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_LONG_NUMBER = re.compile(r"\b\d{4,}\b")
# Repeated, so stacked prefixes such as "Kartenzahlung girocard REWE" go as well.
_CARD_PREFIX = re.compile(
    r"^(?:(?:kaufumsatz|lastschrift|girocard|visa(?: card)?|mastercard|debitkarte|kartenzahlung|pos|poz|elv)"
    r"\b[\s:.\-]*)+",
    re.IGNORECASE,
)
# "SUMUP *BAECKEREI", "PAYPAL *SPOTIFY", "PP.1234.PP . MERCHANT"
_INTERMEDIARY = re.compile(r"^(?:sumup|paypal|zettle|sq)\s*\*\s*", re.IGNORECASE)
# "AMZN Mktp DE*AB12CD34", "Amazon.de*AB12CD": the part after "*" is a per-order reference, not a merchant.
# The whole descriptor (with or without "*ref", plus suffixes such as "AMZN.COM/BILL") becomes "amazon".
_AMAZON = re.compile(
    r"^(?:amzn mktp de|amazon\.de|amz)(?:\s*\*\s*\S+)?(?:\s+(?:amazon\.de|amzn\.com/bill|amzn mktp de))*\b",
    re.IGNORECASE,
)
# cleanab imports transactions without an applicant as "Unknown" (older versions: "Unnamed"); that is no payee,
# so it must not group them.
_PLACEHOLDER_PAYEES = {"unnamed", "unknown"}
# Fold umlauts so "Bäckerei Müller" and the transliterated "BAECKEREI MUELLER" share a key.
_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_PAYPAL_PP = re.compile(r"^pp\.\d+\.pp\s*\.\s*", re.IGNORECASE)
# Hyphens count as spaces, so "REWE-Markt" and "REWE Markt" match.
_NOISE = re.compile(r"[^\w\s&./]", re.UNICODE)


def _drop_iban(match: re.Match[str]) -> str:
    return " " if sum(ch.isdigit() for ch in match.group()) >= _IBAN_MIN_DIGITS else match.group()


def normalize_text(text: str | None) -> str:
    """Lower-case the text and strip tags, ids, dates and payment-intermediary prefixes."""
    if not text:
        return ""
    text = _IBAN.sub(_drop_iban, text)
    text = _REFERENCE_TAGS.sub(" ", text)
    text = _PLAIN_TAGS.sub(" ", text)
    # Card prefix first: "VISA PAYPAL *SPOTIFY" must end up as "spotify", like "PAYPAL *SPOTIFY".
    text = _CARD_PREFIX.sub("", text.strip())
    text = _PAYPAL_PP.sub("", text)
    text = _INTERMEDIARY.sub("", text)
    text = _AMAZON.sub("amazon ", text)
    text = _DATE.sub(" ", text)
    text = _TIME.sub(" ", text)
    text = _LONG_NUMBER.sub(" ", text)
    text = _NOISE.sub(" ", text)
    # Sentence dots and stray slashes at token edges ("Netflix.com.", "DANKE.") are not part of a name.
    tokens = (token.strip("./") for token in text.lower().translate(_UMLAUTS).split())
    text = " ".join(token for token in tokens if token)
    return "" if text in _PLACEHOLDER_PAYEES else text


def model_text(payee: str | None, purpose: str | None) -> str:
    """The single string models see: normalized payee, then normalized purpose."""
    payee_text = normalize_text(payee)
    purpose_text = normalize_text(purpose)
    return f"{payee_text} | {purpose_text}" if purpose_text else payee_text
