from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import islice
from typing import Any

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from ejagent.agent.types import StepOutcome
from ejagent.handlers.definition import ToolDefinition
from ejagent.middleware.base import ToolCallContext, ToolMiddleware, ToolNext


class ToolSchemaConfigurationError(ValueError):
    """Raised when a registered tool contains an invalid JSON Schema."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"tool {tool_name!r} contains an invalid parameters schema")


class ToolSchemaValidationMiddleware(ToolMiddleware):
    """Validate normalized tool arguments before policy and handler execution."""

    def __init__(
        self,
        *,
        max_errors: int = 8,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        if isinstance(max_errors, bool) or not isinstance(max_errors, int):
            raise TypeError("max_errors must be an integer")
        if max_errors <= 0:
            raise ValueError("max_errors must be greater than zero")
        super().__init__(name=name, enabled=enabled)
        self.max_errors = max_errors
        self._validators: dict[str, Validator] = {}

    def configure_tools(
        self,
        tool_definitions: Sequence[ToolDefinition],
    ) -> None:
        """Compile every static schema before middleware resources start."""

        validators: dict[str, Validator] = {}
        for definition in tool_definitions:
            if definition.parameters is None:
                continue
            validators[definition.name] = _compile_validator(definition)
        self._validators = validators

    async def __call__(
        self,
        context: ToolCallContext,
        call_next: ToolNext,
    ) -> StepOutcome:
        definition = context.tool_definition
        if definition is None or definition.parameters is None:
            return await call_next(context)
        validator = self._validators.get(definition.name)
        if validator is None:
            validator = _compile_validator(definition)
            self._validators[definition.name] = validator

        try:
            errors = list(
                islice(
                    validator.iter_errors(context.arguments),
                    self.max_errors + 1,
                )
            )
        except Unresolvable as exc:
            raise ToolSchemaConfigurationError(definition.name) from exc
        if not errors:
            return await call_next(context)

        truncated = len(errors) > self.max_errors
        visible_errors = errors[: self.max_errors]
        visible_errors.sort(key=_error_sort_key)
        data: dict[str, Any] = {
            "status": "error",
            "tool": context.tool_name,
            "code": "invalid_tool_arguments",
            "errors": [_serialize_error(error) for error in visible_errors],
        }
        if truncated:
            data["truncated"] = True
        return StepOutcome(data)


def _compile_validator(definition: ToolDefinition) -> Validator:
    assert definition.parameters is not None
    schema = _mutable_json_value(definition.parameters)
    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except SchemaError as exc:
        raise ToolSchemaConfigurationError(definition.name) from exc
    return validator_class(schema, registry=Registry())


def _serialize_error(error: ValidationError) -> dict[str, str]:
    keyword = str(error.validator or "schema")
    path = list(error.absolute_path)
    if keyword == "required":
        missing = _first_missing_property(error)
        if missing is not None:
            path.append(missing)
    return {
        "path": _json_pointer(path),
        "keyword": keyword,
        "message": _safe_error_message(keyword),
    }


def _first_missing_property(error: ValidationError) -> str | None:
    required = error.validator_value
    if not isinstance(required, Sequence) or isinstance(required, str):
        return None
    return next(
        (
            name
            for name in required
            if isinstance(name, str)
            and error.message == f"{name!r} is a required property"
        ),
        None,
    )


def _safe_error_message(keyword: str) -> str:
    messages = {
        "additionalProperties": "unexpected properties are not allowed",
        "anyOf": "value does not match any allowed schema",
        "const": "value does not match the required constant",
        "contains": "array does not contain a required item",
        "dependentRequired": "dependent properties are missing",
        "enum": "value is not in the allowed set",
        "exclusiveMaximum": "number is above the allowed exclusive maximum",
        "exclusiveMinimum": "number is below the allowed exclusive minimum",
        "format": "value does not match the required format",
        "maxItems": "array contains too many items",
        "maxLength": "string is longer than allowed",
        "maxProperties": "object contains too many properties",
        "maximum": "number is above the allowed maximum",
        "minItems": "array contains too few items",
        "minLength": "string is shorter than allowed",
        "minProperties": "object contains too few properties",
        "minimum": "number is below the allowed minimum",
        "multipleOf": "number is not an allowed multiple",
        "not": "value matches a forbidden schema",
        "oneOf": "value does not match exactly one allowed schema",
        "pattern": "string does not match the required pattern",
        "patternProperties": "property does not match the required schema",
        "prefixItems": "array item does not match the required schema",
        "propertyNames": "property name is not allowed",
        "required": "required property is missing",
        "type": "value does not match the required type",
        "uniqueItems": "array items must be unique",
    }
    return messages.get(keyword, "value does not match the tool schema")


def _error_sort_key(error: ValidationError) -> tuple[tuple[str, ...], str]:
    return (
        tuple(str(part) for part in error.absolute_path),
        str(error.validator or "schema"),
    )


def _json_pointer(path: Sequence[object]) -> str:
    if not path:
        return ""
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in path
    )


def _mutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json_value(item) for item in value]
    return value
