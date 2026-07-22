"""Composable local and external tool handlers."""

from ejagent.handlers.base import (
    BaseHandler,
    MethodToolHandler,
    UnknownToolError,
)
from ejagent.handlers.definition import (
    ToolDefinition,
    ToolDefinitionError,
    ToolEffect,
)
from ejagent.handlers.mcp import McpToolHandler

__all__ = [
    "BaseHandler",
    "MethodToolHandler",
    "ToolDefinition",
    "ToolDefinitionError",
    "ToolEffect",
    "UnknownToolError",
    "McpToolHandler",
]
