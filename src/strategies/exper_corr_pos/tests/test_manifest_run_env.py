from __future__ import annotations

import json
from pathlib import Path

from src.strategies.exper_corr_pos import train


def test_record_manifest_includes_run_env(tmp_path: Path):
    # Redirect manifest path to a temporary file to avoid polluting real reports
    manifest_path = tmp_path / "run_manifest.json"
    train.MANIFEST_PATH = manifest_path  # type: ignore[attr-defined]

    cfg_path = tmp_path / "config.json"
    cfg = {"env": {"accounting_mode": "mtm"}, "train": {}}
    cfg_path.write_text(json.dumps(cfg))

    # Should create the manifest with a run_env block
    train._record_manifest(cfg, cfg_path, seed=123)

    data = json.loads(manifest_path.read_text())
    assert isinstance(data, list) and data, "manifest should be a non-empty list"
    last = data[-1]
    run_env = last.get("run_env")
    assert isinstance(run_env, dict), "run_env must be a dict in manifest entry"
    assert "python" in run_env and isinstance(run_env["python"], str)
    # lib_versions should include at least numpy key (best-effort presence)
    libs = run_env.get("lib_versions")
    assert isinstance(libs, dict) and "numpy" in libs
