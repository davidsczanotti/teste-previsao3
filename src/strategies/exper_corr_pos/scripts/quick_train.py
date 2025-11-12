from __future__ import annotations

"""
Quick training runner for exper_corr_pos to produce a minimal checkpoint
for audits/backtests without changing the main config.

Usage:

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.quick_train

This reads src/strategies/exper_corr_pos/config.json and applies tiny
overrides to run a short training and save moe_policy_final.pt under
reports/train/quickrun/.
"""

import json
from pathlib import Path
from typing import Dict, Any

from ..train import train_agent, DEFAULT_CONFIG


def main() -> None:
    cfg_path = Path(DEFAULT_CONFIG)
    cfg: Dict[str, Any] = json.loads(cfg_path.read_text())

    overrides: Dict[str, Any] = {
        "train": {
            "episodes": 2,
            "rollout_steps": 64,
            "eval_every": 0,
            "plot_every": 0,
            "ckpt_every": 0,
            "final_every": 1,
            "log_every": 1,
            "resume": False,
            "outdir": "src/strategies/exper_corr_pos/reports/train/quickrun",
        }
    }

    print("[quick_train] Running short training to materialize a checkpoint…")
    result = train_agent(
        cfg,
        cfg_path=cfg_path,
        overrides=overrides,
        record_manifest=False,
        enable_plots=False,
        trial_id=None,
        disable_wandb=True,
    )
    print("[quick_train] Done:", result)


if __name__ == "__main__":
    main()

