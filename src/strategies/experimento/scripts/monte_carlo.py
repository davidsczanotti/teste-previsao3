from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..analysis.monte_carlo import MonteCarloConfig, run_monte_carlo, save_artifact
from ..storage.db import insert_metrics


def load_config():
    cfg_path = Path("src/strategies/experimento/config/config_active.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_config()
    storage = cfg["storage"]
    analysis = cfg.get("analysis", {})
    mc_cfg = analysis.get("monte_carlo", {})
    sims = int(mc_cfg.get("sims", 1000))
    steps = mc_cfg.get("steps")
    if steps is not None:
        steps = int(steps)
    seed = int(mc_cfg.get("seed", 42))

    mc = MonteCarloConfig(sims=sims, steps=steps, seed=seed)

    # None => last run
    result = run_monte_carlo(storage["results_db"], run_id=None, cfg=mc)

    # Persist as metrics on last run and save artifact
    with sqlite3.connect(storage["results_db"]) as cx:
        cur = cx.execute(
            "SELECT run_id FROM runs WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            rid = row[0]
            insert_metrics(cx, rid, {k: float(v) for k, v in result.items()})
            cx.commit()
            save_artifact(storage["artifacts_dir"], rid, result)
            print(f"Monte Carlo saved for {rid}: {result}")
        else:
            print("No finished runs to attach Monte Carlo results.")


if __name__ == "__main__":
    main()

