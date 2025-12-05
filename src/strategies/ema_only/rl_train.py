"""
Script para criar ambiente RL EMA-only a partir de config.json.
Se chamado diretamente, roda apenas um rollout aleatório para sanity check.
Use este módulo para plugar em PPO (stable-baselines3) ou outro agente.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .backtest import compute_ema
from .rl_env import EmaEnv, RLConfig
from .rl_features import build_features
from .regime_sampler import attach_regime, concat_blocks, label_regime_daily, make_blocks, sample_blocks
from ...utils.data_loader import load_data_range  # type: ignore

CFG_PATH = Path("src/strategies/ema_only/config.json")


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def make_env_from_cfg(
    cfg: dict,
    start: str,
    end: str,
    *,
    use_regime_sampling: bool | None = None,
    seed: int | None = None,
) -> Tuple[EmaEnv, pd.Series, pd.Series]:
    data_cfg = cfg.get("data", {})
    strat_cfg = cfg.get("strategy", {})
    rl_cfg = cfg.get("rl", {})
    reward_cfg = rl_cfg.get("reward", {})
    backtest_cfg = cfg.get("backtest", {})

    symbol = data_cfg.get("symbol", "BTCUSDT")
    base_tf = data_cfg.get("timeframe", "4h")
    ref_tf = data_cfg.get("ref_timeframe", "1d")
    intraday_tf = data_cfg.get("intraday_timeframe", "1h")
    intr_window_hours = int(data_cfg.get("intraday_window_hours", 12))
    intr_min_align = float(data_cfg.get("intraday_min_alignment", 0.6))
    fast = strat_cfg.get("ema_period", 34)
    slow = strat_cfg.get("slow_ema_period", 89)
    ref_ema_period = strat_cfg.get("ref_ema_period", 200)

    base = (
        load_data_range(symbol, base_tf, start, end, use_cache_only=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    ref = (
        load_data_range(symbol, ref_tf, "2018-01-01 00:00:00", end, use_cache_only=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    # Série intraday (1h) para o especialista adicional; alinharemos por Date via merge_asof.
    intraday = (
        load_data_range(symbol, intraday_tf, "2018-01-01 00:00:00", end, use_cache_only=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )
    feats = build_features(base, ref, fast=fast, slow=slow, ref_ema_period=ref_ema_period)

    # Especialista intraday: exige que a tendência 1h esteja alinhada com a 4h
    # em uma fração mínima dos candles da janela (rolling).
    try:
        intr = intraday.copy()
        intr["ema_fast_1h"] = compute_ema(intr["close"].astype(float), fast)
        intr["ema_slow_1h"] = compute_ema(intr["close"].astype(float), slow)
        intr["sign_1h"] = np.sign(intr["ema_fast_1h"] - intr["ema_slow_1h"]).replace(0.0, np.nan)

        # Tendência 4h em função do tempo; depois projetada para 1h via merge_asof.
        base_trend = base.copy()
        base_trend["ema_fast_4h"] = compute_ema(base_trend["close"].astype(float), fast)
        base_trend["ema_slow_4h"] = compute_ema(base_trend["close"].astype(float), slow)
        base_trend["sign_4h"] = np.sign(base_trend["ema_fast_4h"] - base_trend["ema_slow_4h"]).replace(0.0, np.nan)
        trend_map = base_trend[["Date", "sign_4h"]].sort_values("Date").reset_index(drop=True)

        intr = intr.sort_values("Date").reset_index(drop=True)
        intr = pd.merge_asof(intr, trend_map, on="Date", direction="backward")

        # Flag de alinhamento por candle 1h: +1 se mesma direção e não nulos, 0 caso contrário.
        align_flag = (
            (intr["sign_1h"].notna())
            & (intr["sign_4h"].notna())
            & (np.sign(intr["sign_1h"]) == np.sign(intr["sign_4h"]))
        ).astype(float)
        # Janela em barras 1h (>=1)
        win_bars = max(1, int(intr_window_hours))
        intr["align_ratio"] = align_flag.rolling(window=win_bars, min_periods=1).mean()

        # Projeta o align_ratio para o índice 4h via merge_asof.
        align_map = intr[["Date", "align_ratio"]].sort_values("Date").reset_index(drop=True)
        merged = pd.merge_asof(
            base[["Date"]].copy().reset_index(drop=True),
            align_map,
            on="Date",
            direction="backward",
        )

        feats = feats.copy()
        feats["intraday_align_ratio"] = merged["align_ratio"].fillna(0.0)
        feats["exp_intraday_trend"] = (feats["intraday_align_ratio"] >= intr_min_align).astype(float)
    except Exception:
        # Se algo der errado, mantemos o expert intraday em 0.0 (não quebra o treino).
        feats = feats.copy()
        if "intraday_align_ratio" not in feats.columns:
            feats["intraday_align_ratio"] = 0.0
        if "exp_intraday_trend" not in feats.columns:
            feats["exp_intraday_trend"] = 0.0
    # --- Amostragem por regime (opcional) -----------------------------------
    rs_cfg = rl_cfg.get("train", {}).get("regime_sampling") or {}
    if use_regime_sampling is None:
        use_regime_sampling = bool(rs_cfg.get("enabled", False))
    if use_regime_sampling:
        block_months = int(rs_cfg.get("block_months", 6))
        num_blocks = int(rs_cfg.get("num_blocks", 4))
        thresholds = rs_cfg.get("regime_thresholds", {}) or {}
        bull_thr = float(thresholds.get("bull", 0.01))
        bear_thr = float(thresholds.get("bear", -0.01))
        rs_seed = rs_cfg.get("seed", seed)

        ref_reg = label_regime_daily(ref.copy(), bull=bull_thr, bear=bear_thr, lookback=30)
        base_with_reg = attach_regime(base, ref_reg)
        blocks = make_blocks(base_with_reg, block_months=block_months)
        sampled = sample_blocks(blocks, num_blocks=num_blocks, seed=rs_seed)
        base, feats = concat_blocks(base_with_reg, feats, sampled)

    # Garante coluna block_reset no vetor de features
    if "block_reset" not in feats.columns:
        feats = feats.copy()
        feats["block_reset"] = 0.0

    norm_mean, norm_std = feats.mean(), feats.std().replace(0.0, 1.0)

    # Custos e tamanho de posição: por padrão herdam da estratégia/backtest,
    # mas podem ser sobrescritos em rl.reward.
    fee_pct = float(reward_cfg.get("fee_pct", strat_cfg.get("fee_pct", 0.0004)))
    slippage_pct = float(reward_cfg.get("slippage_pct", 0.0005))
    lot_size = float(reward_cfg.get("lot_size", strat_cfg.get("lot_size", 0.001)))
    init_equity = float(reward_cfg.get("init_equity", backtest_cfg.get("initial_capital", 1000.0)))

    rlconf = RLConfig(
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        lot_size=lot_size,
        init_equity=init_equity,
        trade_penalty=float(reward_cfg.get("trade_penalty", 0.0)),
        dd_threshold_pct=float(reward_cfg.get("dd_threshold_pct", 0.02)),
        dd_penalty=float(reward_cfg.get("dd_penalty", 0.0)),
        min_hold_bars=int(reward_cfg.get("min_hold_bars", 0)),
        churn_penalty=float(reward_cfg.get("churn_penalty", 0.0)),
        align_bonus=float(reward_cfg.get("align_bonus", 0.0)),
        realized_bonus_coef=float(reward_cfg.get("realized_bonus_coef", 0.0)),
        atr_risk_scale=float(reward_cfg.get("atr_risk_scale", 0.0)),
        enforce_ref_bias=bool(reward_cfg.get("enforce_ref_bias", True)),
        reward_scale=float(reward_cfg.get("reward_scale", 1.0)),
        consensus_bonus=float(reward_cfg.get("consensus_bonus", 0.0)),
        consensus_threshold=float(reward_cfg.get("consensus_threshold", 0.66)),
        vol_max_atr_rel=float(reward_cfg.get("vol_max_atr_rel", 0.0)),
        vol_penalty=float(reward_cfg.get("vol_penalty", 0.0)),
        gating_penalty=float(reward_cfg.get("gating_penalty", 0.0)),
        turnover_penalty=float(reward_cfg.get("turnover_penalty", 0.0)),
        living_cost_per_episode=float(reward_cfg.get("living_cost_per_episode", 0.0)),
        use_monthly_reward=bool(reward_cfg.get("use_monthly_reward", False)),
        allow_short=True,
        trend_entry_bonus=float(reward_cfg.get("trend_entry_bonus", 0.0)),
        trend_entry_penalty=float(reward_cfg.get("trend_entry_penalty", 0.0)),
        trend_flip_penalty=float(reward_cfg.get("trend_flip_penalty", 0.0)),
        atr_stop_mult=float(reward_cfg.get("atr_stop_mult", 0.0)),
        atr_trail_mult=float(reward_cfg.get("atr_trail_mult", 0.0)),
        max_long_entry_dist_fast_pct=float(reward_cfg.get("max_long_entry_dist_fast_pct", 0.0)),
        max_short_entry_dist_fast_pct=float(reward_cfg.get("max_short_entry_dist_fast_pct", 0.0)),
        pullback_entry_bonus=float(reward_cfg.get("pullback_entry_bonus", 0.0)),
        trend_exit_penalty=float(reward_cfg.get("trend_exit_penalty", 0.0)),
    )
    env = EmaEnv(df=base, features=feats, cfg=rlconf, norm_mean=norm_mean, norm_std=norm_std)
    return env, norm_mean, norm_std


class MetricsCallback(BaseCallback):
    """
    Callback simples que registra métricas em CSV para uso pelo rl_visualize.

    Colunas:
      - step: timesteps globais do PPO
      - reward_mean: média do reward bruto no último passo (aprox.)
      - pnl: equity atual - equity inicial
      - trades: número acumulado de trades
    """

    def __init__(self, metrics_path: Path, log_every: int = 1_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.metrics_path = metrics_path
        self.log_every = log_every
        self._last_logged_step = 0
        self._init_file()

    def _init_file(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        import csv

        # Sempre recria o arquivo a cada treino para evitar sobreposição
        # de múltiplas execuções no mesmo metrics.csv.
        with self.metrics_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward_mean", "pnl", "trades"])

    def _on_step(self) -> bool:
        # Registra no máximo a cada log_every timesteps para evitar arquivos gigantes.
        if self.num_timesteps - self._last_logged_step < self.log_every:
            return True

        try:
            import csv

            rewards = self.locals.get("rewards")
            reward_mean = float(np.mean(rewards)) if rewards is not None else 0.0

            # Assume DummyVecEnv com um único env
            env = getattr(self.training_env, "envs", [None])[0]
            pnl = 0.0
            trades = 0
            if env is not None:
                init_equity = float(getattr(env.cfg, "init_equity", 0.0))
                equity_now = float(getattr(env, "last_equity", getattr(env, "equity", init_equity)))
                pnl = equity_now - init_equity
                trades = int(getattr(env, "trades", 0))

            with self.metrics_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([int(self.num_timesteps), reward_mean, pnl, trades])
            self._last_logged_step = int(self.num_timesteps)
        except Exception:
            # Nunca quebra o treino por causa do callback.
            return True

        return True


def train_from_config(
    cfg: dict,
    *,
    start_override: str | None = None,
    end_override: str | None = None,
    use_regime_sampling: bool | None = None,
    run_name: str | None = None,
) -> None:
    rl_cfg = cfg.get("rl", {})
    train_cfg = rl_cfg.get("train", {})
    start = str(start_override or train_cfg.get("start", "2019-01-01 00:00:00"))
    end = str(end_override or train_cfg.get("end", "2025-01-01 00:00:00"))
    total_timesteps = int(train_cfg.get("total_timesteps", 800_000))
    n_steps = int(train_cfg.get("n_steps", 256))
    batch_size = int(train_cfg.get("batch_size", 256))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    gamma = float(train_cfg.get("gamma", 0.99))
    n_epochs = int(train_cfg.get("n_epochs", 5))

    def _make_env():
        env, _, _ = make_env_from_cfg(
            cfg,
            start=start,
            end=end,
            use_regime_sampling=use_regime_sampling,
            seed=int(train_cfg.get("seed", 42)),
        )
        return env

    vec_env = DummyVecEnv([_make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        n_epochs=n_epochs,
        verbose=1,
    )

    metrics_dir = Path("src/strategies/ema_only/reports/rl")
    if run_name:
        metrics_dir = metrics_dir / run_name
    metrics_cb = MetricsCallback(metrics_dir / "metrics.csv")
    model.learn(total_timesteps=total_timesteps, callback=metrics_cb)

    model_path = metrics_dir / "ppo_ema_only.zip"
    model.save(model_path)
    print(f"Modelo PPO salvo em {model_path}")


def main() -> None:
    cfg = load_cfg()
    train_from_config(cfg)


if __name__ == "__main__":
    main()
