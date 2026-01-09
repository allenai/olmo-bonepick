import dataclasses as dt
from copy import deepcopy

from pydantic import TypeAdapter

from .annotate_utils import DataclassType


def _resolve_definition(property_: dict, definitions: dict) -> dict:
    if "$ref" not in property_:
        return deepcopy(property_)

    _, ref_key = property_["$ref"].rsplit("/", 1)
    return deepcopy(definitions[ref_key])


def _recursively_reformat_json_schema(property_: dict, definitions: dict) -> dict:
    reformatted_property_ = _resolve_definition(property_, definitions)

    if reformatted_property_.get("additionalProperties", False) is not False:
        raise ValueError("Additional properties are not supported")

    # for objects, mark all properties as required
    if reformatted_property_["type"] == "object":
        reformatted_property_["required"] = list(reformatted_property_["properties"].keys())

    for key in list(reformatted_property_["properties"].keys()):
        reformatted_value = _resolve_definition(reformatted_property_["properties"][key], definitions)

        if reformatted_value["type"] == "object":
            # format the nested properties
            reformatted_value = _recursively_reformat_json_schema(
                definitions=definitions, property_=reformatted_value
            )

        reformatted_property_["properties"][key] = reformatted_value

    return reformatted_property_


def dataclass_to_json_schema(dt_cls: type[DataclassType]) -> dict:
    if not dt.is_dataclass(dt_cls):
        raise ValueError(f"Expected a dataclass, got {type(dt_cls)}")

    schema = TypeAdapter(dt_cls).json_schema()
    reformatted_schema = _recursively_reformat_json_schema(definitions=schema.pop("$defs", {}), property_=schema)
    return reformatted_schema
