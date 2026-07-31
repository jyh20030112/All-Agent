from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]
MutableJsonValue: TypeAlias = (
    JsonScalar | list["MutableJsonValue"] | dict[str, "MutableJsonValue"]
)


def freeze_json_value(value: object, *, label: str = "value") -> JsonValue:
    """Validate and recursively freeze one JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings")
            frozen[key] = freeze_json_value(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(
            freeze_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must be JSON-compatible")


def freeze_json_object(
    value: Mapping[str, object],
    *,
    label: str = "value",
) -> JsonObject:
    """Validate and recursively freeze one JSON object."""

    frozen = freeze_json_value(value, label=label)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return frozen


def thaw_json_value(value: JsonValue) -> MutableJsonValue:
    """Return a detached mutable representation of one frozen JSON value."""

    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value
