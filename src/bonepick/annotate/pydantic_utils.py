import dataclasses as dt

from pydantic import TypeAdapter

from .annotate_utils import DataclassType


def dataclass_to_json_schema(dt_cls: type[DataclassType]) -> dict:
    if not dt.is_dataclass(dt_cls):
        raise ValueError(f"Expected a dataclass, got {type(dt_cls)}")

    schema = TypeAdapter(dt_cls).json_schema()
    return {**{"additionalProperties": False,}, **schema}
