import runpy
import unittest
from pathlib import Path

from ejagent import OpenAIModelAdapter, SkillManager
from ejagent.context import SkillsContextPipeline
from ejagent.contracts import CancellationSource, ToolCall
from ejagent.providers import OpenAIModelPort
from ejagent.tools import FunctionTool, FunctionToolExecutor

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


class ExampleTests(unittest.IsolatedAsyncioTestCase):
    def test_examples_can_be_imported_without_running_main(self) -> None:
        for path in sorted(EXAMPLES_DIR.glob("*.py")):
            with self.subTest(example=path.name):
                namespace = runpy.run_path(path, run_name="example_test")
                self.assertIn("main", namespace)

    async def test_custom_tool_example_dispatches(self) -> None:
        namespace = runpy.run_path(
            EXAMPLES_DIR / "02_custom_tool.py",
            run_name="example_test",
        )
        executor = FunctionToolExecutor(
            (FunctionTool(namespace["ADD_TOOL"], namespace["add"]),)
        )

        outcome = await executor.execute(
            ToolCall("test-call", "add", {"left": 19.5, "right": 22.5}),
            cancellation=CancellationSource().token,
        )
        self.assertEqual(
            outcome.result,
            {"status": "success", "value": 42.0},
        )

    async def test_skill_example_discovers_local_skill(self) -> None:
        namespace = runpy.run_path(
            EXAMPLES_DIR / "06_skill.py",
            run_name="example_test",
        )
        pipeline = SkillsContextPipeline(namespace["SKILLS_DIR"])

        await pipeline.start()
        manager = pipeline.catalog

        self.assertIn("release_notes", manager._skills)
        skill = manager._skills["release_notes"]
        self.assertIsNotNone(skill.template_md)
        self.assertIsNotNone(skill.sample_md)
        await pipeline.shutdown()

    def test_harness_examples_use_real_provider_adapter(self) -> None:
        for filename in (
            "07_event_observers.py",
            "09_runtime_control.py",
            "10_composed_harness.py",
            "11_streaming_events.py",
            "12_tool_progress.py",
            "13_usage_budget.py",
            "14_context_pressure.py",
            "15_explicit_compaction.py",
        ):
            with self.subTest(example=filename):
                namespace = runpy.run_path(
                    EXAMPLES_DIR / filename,
                    run_name="example_test",
                )
                if filename == "09_runtime_control.py":
                    adapter = namespace["ObservableOpenAIModelAdapter"]
                    self.assertTrue(issubclass(adapter, OpenAIModelAdapter))
                else:
                    self.assertIs(
                        namespace["OpenAIModelAdapter"],
                        OpenAIModelAdapter,
                    )

        for filename in (
            "01_stateful_chat.py",
            "02_custom_tool.py",
            "04_mcp_tools.py",
            "06_skill.py",
            "08_session_resume.py",
            "16_durable_session.py",
        ):
            with self.subTest(example=filename):
                namespace = runpy.run_path(
                    EXAMPLES_DIR / filename,
                    run_name="example_test",
                )
                self.assertIs(namespace["OpenAIModelPort"], OpenAIModelPort)

    def test_skill_manager_requires_an_explicit_root(self) -> None:
        with self.assertRaisesRegex(TypeError, "skills_root"):
            SkillManager()  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
