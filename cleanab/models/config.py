from datetime import date
from typing import Annotated, Dict, List, Union

from pydantic import model_validator, ConfigDict, BaseModel, Field
from pydantic.main import create_model

from cleanab.apps.base import BaseApp, BaseAppConfig, load_app

from ..constants import FIELDS_TO_CLEAN_UP
from .account_config import AccountConfig
from .cleaner import FinalizerDefinition, ReplacementDefinition


class TimespanConfig(BaseModel):
    earliest_date: date = date(2000, 1, 1)
    maximum_days: Annotated[int, Field(ge=1)] = 30


class CleanabConfig(BaseModel):
    concurrency: Annotated[int, Field(gt=0)] = 1
    minimum_holdings_delta: Annotated[float, Field(ge=0)] = 1
    debug: bool = False
    fints_product_id: str | None = None


NestedReplacementEntry = List[Union[ReplacementDefinition, str]]
FullReplacementEntry = List[
    Union[
        NestedReplacementEntry,
        ReplacementDefinition,
        str,
    ]
]

ReplacementFields: type[BaseModel] = create_model(
    "ReplacementFields",
    **{field: (FullReplacementEntry, []) for field in FIELDS_TO_CLEAN_UP}   # type: ignore
)


FinalizerFields = create_model(
    "FinalizerFields",
    **{
        field: (FinalizerDefinition, FinalizerDefinition())
        for field in FIELDS_TO_CLEAN_UP
    }   # type: ignore
)


class Config(BaseModel):
    cleanab: CleanabConfig = CleanabConfig()
    timespan: TimespanConfig = TimespanConfig()
    accounts: Annotated[list[AccountConfig], Field(min_length=1)]
    replacements: BaseModel = ReplacementFields()
    pre_replacements: BaseModel = ReplacementFields()
    finalizer: BaseModel = FinalizerFields()
    apps: Dict[str, BaseAppConfig] = {}
    _parsed_apps: list[BaseApp] = []
    model_config: ConfigDict = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def add_type_key(cls, values):
        apps = values.get('apps', {})
        for key in apps.keys():
            apps[key]['module'] = key
        values['apps'] = apps
        return values

    def load_apps(self):
        self._parsed_apps = [
            load_app(app_name, app_config)
            for app_name, app_config in self.apps.items()
        ]

    def get_apps(self) -> list[BaseApp]:
        return self._parsed_apps
