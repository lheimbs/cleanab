"""Stage 0: strip bank boilerplate from German SEPA/card text so the merchant is what is left."""

from __future__ import annotations

import re

# SEPA purpose tags such as "SVWZ+", "EREF+<ref>", "MREF+<ref>", "CRED+<id>", "KREF+<ref>".
# The reference tags carry opaque ids, so the value is dropped; SVWZ is the actual purpose text.
_REFERENCE_TAGS = re.compile(r"\b(?:EREF|MREF|CRED|KREF|ABWA|ABWE|BREF|OAMT|COAM)\+\S*", re.IGNORECASE)
_PLAIN_TAGS = re.compile(r"\b(?:SVWZ|IBAN|BIC)[+:]?\s*", re.IGNORECASE)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4}\b")
_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.(?:\d{2}|\d{4})?(?=\D|$)")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_LONG_NUMBER = re.compile(r"\b\d{4,}\b")
_CARD_PREFIX = re.compile(
    r"^(?:kaufumsatz|lastschrift|girocard|visa(?: card)?|mastercard|debitkarte|kartenzahlung|pos|poz|elv)\b[\s:.\-]*",
    re.IGNORECASE,
)
# "SUMUP *BAECKEREI", "PAYPAL *SPOTIFY", "PP.1234.PP . MERCHANT", "AMZN Mktp DE*AB12"
_INTERMEDIARY = re.compile(r"^(?:sumup|paypal|zettle|sq|amzn mktp de|amz)\s*\*\s*", re.IGNORECASE)
_PAYPAL_PP = re.compile(r"^pp\.\d+\.pp\s*\.\s*", re.IGNORECASE)
_NOISE = re.compile(r"[^\w\s&./-]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Lower-case the text and strip tags, ids, dates and payment-intermediary prefixes."""
    if not text:
        return ""
    text = _IBAN.sub(" ", text)
    text = _REFERENCE_TAGS.sub(" ", text)
    text = _PLAIN_TAGS.sub(" ", text)
    text = _PAYPAL_PP.sub("", text.strip())
    text = _INTERMEDIARY.sub("", text)
    text = _CARD_PREFIX.sub("", text)
    text = _DATE.sub(" ", text)
    text = _TIME.sub(" ", text)
    text = _LONG_NUMBER.sub(" ", text)
    text = _NOISE.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def model_text(payee: str | None, purpose: str | None) -> str:
    """The single string models see: normalized payee, then normalized purpose."""
    payee_text = normalize_text(payee)
    purpose_text = normalize_text(purpose)
    return f"{payee_text} | {purpose_text}" if purpose_text else payee_text
