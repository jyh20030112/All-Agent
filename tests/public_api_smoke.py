"""Smoke-test the installed package without relying on the source tree."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from importlib.metadata import version
from importlib.resources import files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-no-mcp", action="store_true")
    args = parser.parse_args()

    if args.expect_no_mcp and importlib.util.find_spec("fastmcp") is not None:
        raise AssertionError("fastmcp must not be installed by the core package")

    import simagentplg

    missing_attributes = [
        name for name in simagentplg.__all__ if not hasattr(simagentplg, name)
    ]
    if missing_attributes:
        raise AssertionError(
            f"public exports do not resolve: {', '.join(missing_attributes)}"
        )

    required_exports = {
        "BaseAgent",
        "BehaviorAction",
        "BehaviorDecision",
        "BehaviorHook",
        "BehaviorHookError",
        "TurnSnapshot",
        "AgentOrchestrator",
        "AgentRunResult",
        "ModelAdapter",
        "OpenAIModelAdapter",
        "ToolMiddleware",
        "ToolEffect",
        "McpToolHandler",
        "McpServerManager",
        "SkillManager",
        "SessionStorage",
        "Compactor",
        "AutoCompactionPolicy",
        "ContextOverflowError",
        "ControlInput",
        "ControlInputKind",
        "ControlReceipt",
        "ControlStatus",
        "ContinueRejectedReason",
        "ContinueRejectedError",
        "FollowUpFailurePolicy",
        "FollowUpDiscardReason",
        "FollowUpHandle",
        "FollowUpError",
        "FollowUpRejectedError",
        "FollowUpDiscardedError",
        "CompactionTrigger",
        "ModelCompactor",
        "JsonlSessionStorage",
        "SessionTreeStorage",
        "SessionLockTimeoutError",
        "SessionRecord",
        "SessionBranch",
        "SessionCheckout",
        "SessionRetry",
        "SteeringApplied",
        "SteeringDiscarded",
        "AgentContinued",
        "SessionRunIntent",
        "session_to_dict",
        "session_from_dict",
    }
    missing_exports = required_exports.difference(simagentplg.__all__)
    if missing_exports:
        raise AssertionError(
            f"required public exports are missing: {', '.join(sorted(missing_exports))}"
        )

    if not files("simagentplg").joinpath("py.typed").is_file():
        raise AssertionError("installed package is missing the py.typed marker")
    if not version("SimAgentPlg"):
        raise AssertionError("installed distribution has no version")

    if args.expect_no_mcp:

        async def check_missing_mcp_message() -> None:
            manager = simagentplg.McpServerManager("unused.json")
            try:
                await manager.startup()
            except RuntimeError as exc:
                if "SimAgentPlg[mcp]" not in str(exc):
                    raise AssertionError(
                        "missing MCP dependencies produced no install guidance"
                    ) from exc
            else:
                raise AssertionError("MCP startup unexpectedly succeeded")

        asyncio.run(check_missing_mcp_message())


if __name__ == "__main__":
    main()
