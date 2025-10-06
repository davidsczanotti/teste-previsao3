from __future__ import annotations

import os
from typing import Dict, Any

from .config import DeepTripleRsiConfig
from .pretrain import pretrain
from .train import train


def run_curriculum() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    cfg = DeepTripleRsiConfig()
    model_path = os.path.join("reports", "agents", "triple_rsi_deep", f"{cfg.ticker}_{cfg.interval}.npz")

    # Step 1: behavior cloning from heuristic
    pre = pretrain(cfg, epochs=10, lr=1e-3, out_path=model_path)
    results["pretrain"] = pre

    # Step 2: RL without costs (exploration)
    cfgA = DeepTripleRsiConfig(
        ticker=cfg.ticker, interval=cfg.interval, days=cfg.days,
        episodes=20, learning_rate=5e-4,
        lot_size=cfg.lot_size, fee_rate=0.0, slippage_bps=0,
        min_hold_bars=8, reopen_cooldown_bars=8,
        action_cost_open=0.0, action_cost_close=0.0,
        stoch_period=cfg.stoch_period, stoch_upper=cfg.stoch_upper, stoch_lower=cfg.stoch_lower,
        long_only=True,
        epsilon_start=0.2, epsilon_end=0.05,
        bc_weight=0.05,
    )
    resA = train(cfgA, model_path=model_path)
    results["phase_A"] = resA["eval"]

    # Step 3: RL with mild costs
    cfgB = DeepTripleRsiConfig(
        ticker=cfg.ticker, interval=cfg.interval, days=cfg.days,
        episodes=40, learning_rate=5e-4,
        lot_size=cfg.lot_size, fee_rate=0.0002, slippage_bps=1,
        min_hold_bars=6, reopen_cooldown_bars=6,
        action_cost_open=0.05, action_cost_close=0.05,
        stoch_period=cfg.stoch_period, stoch_upper=cfg.stoch_upper, stoch_lower=cfg.stoch_lower,
        long_only=True,
        epsilon_start=0.1, epsilon_end=0.02,
        bc_weight=0.02,
    )
    resB = train(cfgB, model_path=model_path)
    results["phase_B"] = resB["eval"]

    # Step 4: RL with realistic costs
    cfgC = DeepTripleRsiConfig(
        ticker=cfg.ticker, interval=cfg.interval, days=cfg.days,
        episodes=30, learning_rate=3e-4,
        lot_size=cfg.lot_size, fee_rate=0.001, slippage_bps=5,
        min_hold_bars=6, reopen_cooldown_bars=6,
        action_cost_open=1.0, action_cost_close=1.0,
        stoch_period=cfg.stoch_period, stoch_upper=cfg.stoch_upper, stoch_lower=cfg.stoch_lower,
        long_only=True,
    )
    resC = train(cfgC, model_path=model_path)
    results["phase_C"] = resC["eval"]

    return results


if __name__ == "__main__":
    out = run_curriculum()
    print(out)
