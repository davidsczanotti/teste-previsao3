from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib as toml
else:  # pragma: no cover - CI/envs <3.11
    import tomli as toml  # type: ignore


def test_pyproject_moves_heavy_deps_to_extras():
    root = Path(__file__).resolve().parents[4]
    data = toml.loads((root / "pyproject.toml").read_text())

    # PEP 621 style
    project = data.get("project", {})
    deps = project.get("dependencies", [])
    assert all("tensorflow" not in d for d in deps), "tensorflow must not be in default dependencies"
    assert all("flask" not in d for d in deps), "flask must not be in default dependencies"

    opt = project.get("optional-dependencies", {})
    # Extras must exist and include tensorflow/flask
    tf = opt.get("tf", [])
    web = opt.get("web", [])
    assert any("tensorflow" in s for s in tf), "tensorflow should be listed under [project.optional-dependencies].tf"
    assert any("flask" in s for s in web), "flask should be listed under [project.optional-dependencies].web"
