from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "ejagent"
LOW_LEVEL_PACKAGES = (
    SOURCE_ROOT / "contracts",
    SOURCE_ROOT / "kernel",
)
FORBIDDEN_IMPORTS = (
    "ejagent.agent",
    "ejagent.handlers",
    "ejagent.middleware",
    "ejagent.plugins",
    "ejagent.providers",
    "ejagent.session",
)


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_contracts_and_kernel_do_not_import_legacy_layers(self) -> None:
        violations: list[str] = []

        for package in LOW_LEVEL_PACKAGES:
            for path in sorted(package.rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    modules: tuple[str, ...]
                    if isinstance(node, ast.ImportFrom):
                        modules = (node.module or "",)
                    elif isinstance(node, ast.Import):
                        modules = tuple(alias.name for alias in node.names)
                    else:
                        continue
                    for module in modules:
                        if module.startswith(FORBIDDEN_IMPORTS):
                            relative = path.relative_to(PROJECT_ROOT)
                            violations.append(f"{relative}:{node.lineno}: {module}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
