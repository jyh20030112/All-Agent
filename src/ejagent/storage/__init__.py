"""Durable adapters for EJAgent Core contracts."""

from ejagent.storage.jsonl import STORE_SCHEMA_VERSION, JsonlSessionStore

__all__ = [
    "JsonlSessionStore",
    "STORE_SCHEMA_VERSION",
]
