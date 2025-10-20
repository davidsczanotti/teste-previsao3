from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    if not cfg_path.exists():
        print("config_active.json não encontrado.")
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg["storage"]["artifacts_dir"]) / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"config_snapshot_{ts}.json"
    out_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Snapshot salvo em:", out_file)


if __name__ == "__main__":
    main()

