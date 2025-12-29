from enum import Enum

from pydantic import BaseModel, ConfigDict

from .. import utils


class FieldsEnum(str, Enum):
    purpose = "purpose"
    applicant_name = "applicant_name"


class ReplacementDefinition(BaseModel):
    pattern: str
    repl: str = ""
    case_insensitive: bool = True
    regex: bool = True
    transform: dict[FieldsEnum, str] = {}

    def __hash__(self):
        __dict = self.__dict__.copy()
        transform = tuple(__dict.pop("transform").items())
        return hash(self.__class__) + hash(tuple(__dict.values())) + hash(transform)

    model_config = ConfigDict(frozen=True, extra="forbid")

    def get_cleaner(self):
        return utils.regex_sub_instance(self)


class FinalizerDefinition(BaseModel):
    capitalize: bool = True
    strip: bool = True
    model_config = ConfigDict(frozen=True)
