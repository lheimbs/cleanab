from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, List, Tuple

from pydantic import BaseModel


class BaseAppConfig(BaseModel):
    @classmethod
    # TODO[pydantic]: We couldn't refactor `__get_validators__`, please create the `__get_pydantic_core_schema__` manually.
    # Check https://docs.pydantic.dev/latest/migration/#defining-custom-types for more information.
    def __get_validators__(cls):
        yield cls._import_app_

    @classmethod
    def _import_app_(cls, data: Dict):
        module_name = data.pop("module")
        try:
            module_path = "cleanab.apps." + module_name
            module = import_module(module_path)
            return module.Config(**data)
        except ModuleNotFoundError:
            raise ValueError(f"Unknown app: {module_name}")

    @classmethod
    def parse_obj(cls, obj):
        return cls._import_app_(obj)


class BaseApp(ABC):
    @abstractmethod
    def create_transactions(self, transactions) -> Tuple[List, List]:
        return [], []

    @abstractmethod
    def augment_transaction(self, transaction, account):
        pass

    @abstractmethod
    def create_intermediary(self, transactions: list[None]) -> str:
        return ""


def load_app(app_name: str, config: BaseAppConfig) -> BaseApp:
    module_name = f"cleanab.apps.{app_name.lower()}"
    module = import_module(module_name)
    return module.App(config)
