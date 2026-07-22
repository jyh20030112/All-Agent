from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias

ToolSchema: TypeAlias = dict[str, Any]
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _mutable_json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json_copy(item) for item in value]
    return deepcopy(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class ToolEffect(StrEnum):
    """Declared side-effect class used by the Core tool scheduler."""

    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"


class ToolDefinitionError(ValueError):
    """Raised when a handler exposes an invalid tool definition."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Canonical OpenAI-first function tool definition used inside Core."""

    name: str
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    effect: ToolEffect = ToolEffect.SIDE_EFFECTING
    strict: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TOOL_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ToolDefinitionError(
                "tool name must contain 1-64 letters, digits, underscores, or dashes"
            )
        if self.description is not None and not isinstance(self.description, str):
            raise ToolDefinitionError("tool description must be a string or None")
        if self.parameters is not None:
            if not isinstance(self.parameters, Mapping):
                raise ToolDefinitionError("tool parameters must be a mapping or None")
            parameters = _mutable_json_copy(self.parameters)
            try:
                json.dumps(parameters, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ToolDefinitionError(
                    "tool parameters must be JSON serializable"
                ) from exc
            object.__setattr__(self, "parameters", _freeze_json(parameters))
        if not isinstance(self.effect, ToolEffect):
            raise ToolDefinitionError("tool effect must be a ToolEffect")
        if self.strict is not None and not isinstance(self.strict, bool):
            raise ToolDefinitionError("tool strict must be a boolean or None")

    @classmethod
    def from_openai_tool(
        cls,
        tool: Mapping[str, Any],
        *,
        effect: ToolEffect = ToolEffect.SIDE_EFFECTING,
    ) -> ToolDefinition:
        """Normalize one legacy OpenAI function-calling dictionary."""

        if not isinstance(tool, Mapping):
            raise ToolDefinitionError("tool must be a mapping")
        if tool.get("type") != "function":
            raise ToolDefinitionError("only named function tools are supported")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise ToolDefinitionError("tool must contain a function mapping")
        if "name" not in function:
            raise ToolDefinitionError("tool must contain function.name")
        return cls(
            name=function["name"],
            description=function.get("description"),
            parameters=function.get("parameters"),
            effect=effect,
            strict=function.get("strict"),
        )

    def with_effect(self, effect: ToolEffect) -> ToolDefinition:
        """Return a copy with one compatibility effect override applied."""

        if not isinstance(effect, ToolEffect):
            raise ToolDefinitionError("tool effect must be a ToolEffect")
        return replace(self, effect=effect)

    def to_openai_tool(self) -> ToolSchema:
        """Serialize this definition to the OpenAI function-calling shape."""

        function: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            function["description"] = self.description
        if self.parameters is not None:
            function["parameters"] = _mutable_json_copy(self.parameters)
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}


ToolDefinitionInput: TypeAlias = ToolDefinition | Mapping[str, Any]


def normalize_tool_definition(
    tool: ToolDefinitionInput,
    *,
    effect: ToolEffect | None = None,
) -> ToolDefinition:
    """Return one detached canonical definition from either public input form."""

    if isinstance(tool, ToolDefinition):
        return tool if effect is None else tool.with_effect(effect)
    return ToolDefinition.from_openai_tool(
        tool,
        effect=effect or ToolEffect.SIDE_EFFECTING,
    )


def normalize_tool_definitions(
    tools: Sequence[ToolDefinitionInput],
) -> tuple[ToolDefinition, ...]:
    """Normalize a tool collection without retaining legacy dictionaries."""

    return tuple(normalize_tool_definition(tool) for tool in tools)
