"""Composable ToolExecutor adapters for EJAgent Core."""

from ejagent.tools.executor import (
    CompositeToolExecutor,
    FunctionTool,
    FunctionToolExecutor,
    ToolFunction,
)
from ejagent.tools.mcp import McpManager, McpToolExecutor

__all__ = [
    "CompositeToolExecutor",
    "FunctionTool",
    "FunctionToolExecutor",
    "McpManager",
    "McpToolExecutor",
    "ToolFunction",
]
