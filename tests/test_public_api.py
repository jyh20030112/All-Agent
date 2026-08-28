from __future__ import annotations

import unittest

import ejagent


class PublicApiTests(unittest.TestCase):
    def test_top_level_exports_only_new_composition_surface(self) -> None:
        expected = {
            "AgentHarness",
            "AnthropicConfig",
            "AnthropicModelPort",
            "CompositeToolExecutor",
            "DerivedCompactionPipeline",
            "FunctionTool",
            "FunctionToolExecutor",
            "IdentityContextPipeline",
            "JsonlSessionStore",
            "McpToolExecutor",
            "ModelConfig",
            "OpenAIModelPort",
            "RuntimeKernel",
            "Skill",
            "SkillCatalog",
            "SkillsContextPipeline",
        }

        self.assertEqual(set(ejagent.__all__), expected)
        self.assertTrue(all(hasattr(ejagent, name) for name in expected))

    def test_legacy_entry_points_are_absent(self) -> None:
        for name in (
            "BaseAgent",
            "AgentOrchestrator",
            "ModelAdapter",
            "OpenAIModelAdapter",
            "BaseHandler",
            "ToolMiddleware",
            "SessionStorage",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(ejagent, name))


if __name__ == "__main__":
    unittest.main()
