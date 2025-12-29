from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

from logzero import logger
from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema


class BaseAppConfig(BaseModel):
    """Base class for app configurations. Does not define custom schema."""
    pass


class _AppConfigValidator:
    """Custom validator for dynamically loading app configurations."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return {"type": "object"}

    @staticmethod
    def _validate(data: Any) -> BaseAppConfig:
        """Validate and convert a dict to the appropriate app config."""
        if isinstance(data, BaseAppConfig):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        data = dict(data)  # Make a copy to avoid mutating the original
        module_name = data.pop("module", None)
        if not module_name:
            raise ValueError("App config must have a 'module' key")
        try:
            module_path = "cleanab.apps." + module_name
            module = import_module(module_path)
            return module.Config(**data)
        except ModuleNotFoundError:
            raise ValueError(f"Unknown app: {module_name}")


class BaseApp(ABC):
    @abstractmethod
    def create_transactions(self, transactions) -> tuple[list, list]:
        return [], []

    @abstractmethod
    def augment_transaction(self, transaction, account):
        pass

    @abstractmethod
    def create_intermediary(self, transactions: tuple) -> str:
        return ""


def load_app(app_name: str, config: _AppConfigValidator) -> BaseApp:
    logger.debug(f"Loading app {app_name} with config '{config}'")
    module_name = f"cleanab.apps.{app_name.lower()}"
    module = import_module(module_name)
    return module.App(config)
