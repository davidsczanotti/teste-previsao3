from __future__ import annotations

import json
import shutil
from pathlib import Path


def load_config():
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def purge_artifacts(artifacts_dir: Path, keep_last_wfo: int = 1, remove_other_runs: bool = True) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # WFO dirs
    wfo_dirs = sorted([p for p in artifacts_dir.iterdir() if p.is_dir() and p.name.startswith("wfo-")])
    if keep_last_wfo < 0:
        keep_last_wfo = 0
    to_delete = wfo_dirs[:-keep_last_wfo] if keep_last_wfo > 0 else wfo_dirs
    for d in to_delete:
        shutil.rmtree(d, ignore_errors=True)

    if remove_other_runs:
        for p in artifacts_dir.iterdir():
            if p.is_dir() and not p.name.startswith("wfo-"):
                shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    cfg = load_config()
    keep = int(cfg.get("cleanup", {}).get("keep_last_wfo", 1))
    remove_others = bool(cfg.get("cleanup", {}).get("remove_other_runs", True))
    artifacts_dir = Path(cfg["storage"]["artifacts_dir"]).resolve()
    purge_artifacts(artifacts_dir, keep_last_wfo=keep, remove_other_runs=remove_others)
    print(f"Artifacts cleaned. Kept last {keep} WFO dirs. Removed other runs={remove_others}.")


if __name__ == "__main__":
    main()

