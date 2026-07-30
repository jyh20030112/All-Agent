"""Validate wheel and sdist contents produced by the packaging configuration."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def only_match(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {pattern!r} artifact in {directory}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_artifact_smoke.py DIST_DIRECTORY")

    dist_dir = Path(sys.argv[1])
    wheel = only_match(dist_dir, "*.whl")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if "ejagent/py.typed" not in names:
            raise AssertionError("wheel is missing ejagent/py.typed")
        if any(name.startswith("simagentplg/") for name in names):
            raise AssertionError("wheel unexpectedly contains the legacy package")
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode()

    if "Name: ejagent-core\n" not in metadata:
        raise AssertionError("wheel metadata has the wrong distribution name")
    if "Version: 0.6.0\n" not in metadata:
        raise AssertionError("wheel metadata has the wrong distribution version")
    if "Provides-Extra: mcp" not in metadata:
        raise AssertionError("wheel metadata does not declare the mcp extra")
    fastmcp_requirements = [
        line
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: fastmcp")
    ]
    if not fastmcp_requirements or not all(
        "extra == 'mcp'" in line for line in fastmcp_requirements
    ):
        raise AssertionError("fastmcp must only be required by the mcp extra")
    for dependency in ("jsonschema", "referencing"):
        requirements = [
            line
            for line in metadata.splitlines()
            if line.startswith(f"Requires-Dist: {dependency}")
        ]
        if not requirements or any("extra ==" in line for line in requirements):
            raise AssertionError(
                f"{dependency} must be declared as a core wheel dependency"
            )


if __name__ == "__main__":
    main()
