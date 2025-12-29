import json
import pickle
from typing import Annotated

from pydantic import BaseModel, HttpUrl, StringConstraints, field_validator

from ..utils import CACHE_HOME
from ..validators import is_iban
from .enums import AccountType


class AccountConfig(BaseModel):
    iban: str
    per_app_id: str

    fints_username: str
    fints_password: str
    fints_blz: Annotated[
        str,
        StringConstraints(strip_whitespace=True, pattern=r"^\d+$"),
    ]
    fints_endpoint: HttpUrl

    friendly_name: str
    account_type: AccountType = AccountType.CHECKING

    default_cleared: bool = False
    default_approved: bool = False

    def __hash__(self):
        return hash(self.iban + self.per_app_id)

    def __eq__(self, other):
        return self.iban == other.iban and self.per_app_id == other.per_app_id

    def __str__(self):
        base = f"{self.account_type.capitalize()} Account"
        if self.friendly_name:
            base += f" '{self.friendly_name}'"
        return base + f" (…{self.iban[-4:]})"

    @property
    def _account_cache_filename(self):
        return CACHE_HOME / f"{self.iban}.pickle"

    @property
    def _cleaned_account_cache_filename(self):
        return CACHE_HOME / f"{self.iban}_cleaned.json"

    @property
    def has_account_cache(self):
        return self._account_cache_filename.is_file()

    @field_validator("iban")
    @classmethod
    def iban_valid(cls, v):
        if not is_iban(v):
            raise ValueError("Not a valid IBAN")
        return v

    def write_account_cache(self, transactions):
        CACHE_HOME.mkdir(parents=True, exist_ok=True)
        with open(self._account_cache_filename, "wb") as f:
            pickle.dump(transactions, f)

    def write_cleaned_account_cache(self, transactions):
        CACHE_HOME.mkdir(parents=True, exist_ok=True)
        with open(self._cleaned_account_cache_filename, "w") as f:
            json.dump(transactions, f)

    def read_account_cache(self):
        with open(self._account_cache_filename, "rb") as f:
            return pickle.load(f)
