from datetime import date

from pydantic import BaseModel, field_validator


class FintsTransaction(BaseModel):
    date: date
    amount: int
    applicant_name: str
    purpose: str = ""
    import_id: str = ""

    @field_validator("purpose", mode="before")
    @classmethod
    def set_purpose_empty(cls, purpose):
        return purpose or ""
