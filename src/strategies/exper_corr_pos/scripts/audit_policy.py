"""Auditoria rápida de ações e trades do agente MoE.

Uso:

  BINANCE_OFFLINE=1 poetry run python -m src.strategies.exper_corr_pos.scripts.audit_policy \\
      --days 365 --checkpoint src/strategies/exper_corr_pos/reports/train/moe_policy_final.pt

Sem argumentos o script usa os valores do config.json (visualize.prefer) e
carrega 180 dias do cache local.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..data import load_primary_series, load_confirm_series, prepare_dataset
from ..env import BTCMixtureEnv, EnvConfig
from ..models import MoEPolicy
from ..visualize import _find_checkpoint
from ..utils_cfg import enabled_expert_names, bars_for_days, hours_per_bar, build_policy


plt.switch_backend("Agg")


CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")


def _expert_labels(cfg: dict, count: int) -> List[str]:
    names = enabled_expert_names(cfg)
    if len(names) == count:
        return names
    return [f"e{i}" for i in range(count)]


def _vol_bucket(value: float, q1: float, q2: float) -> str:
    if np.isnan(value) or np.isnan(q1) or np.isnan(q2):
        return "unknown"
    if value <= q1:
        return "low"
    if value <= q2:
        return "medium"
    return "high"


def _expected_penalty_components(
    *,
    env_cfg: EnvConfig,
    size: float,
    entry_price: float,
    exit_price: float | None = None,
    side: int,
    reason: str,
    trend_state: float,
    trend_strength: float,
) -> Dict[str, float]:
    notional = abs(size * entry_price)
    if notional <= 0.0:
        return {"turnover": 0.0, "trend": 0.0, "flip": 0.0}

    turnover_pct = max(0.0, float(getattr(env_cfg, "turnover_penalty_pct", 0.0)))
    if turnover_pct > 0.0:
        turnover = notional * turnover_pct
    else:
        turnover = max(0.0, float(env_cfg.turnover_penalty))

    trend = 0.0
    trend_coef_pct = max(0.0, float(getattr(env_cfg, "trend_penalty_coef_pct", 0.0)))
    base_trend = max(0.0, float(env_cfg.trend_penalty_coef))
    if side != 0 and np.isfinite(trend_state) and trend_state != 0.0 and np.sign(trend_state) != np.sign(side):
        if trend_coef_pct > 0.0:
            base = notional * trend_coef_pct
        else:
            base = base_trend
        if base > 0.0:
            strength = trend_strength
            if not np.isfinite(strength) or strength == 0.0:
                strength = 1.0
            mult = max(1.0, float(getattr(env_cfg, "trend_penalty_entry_mult", 1.0)))
            trend = base * abs(strength) * mult

    flip = 0.0
    if reason == "flip":
        flip_pct = max(0.0, float(getattr(env_cfg, "flip_exit_penalty_pct", 0.0)))
        if flip_pct > 0.0:
            use_price = float(exit_price) if exit_price is not None else entry_price
            flip = abs(size * use_price) * flip_pct
        else:
            flip = max(0.0, float(getattr(env_cfg, "flip_exit_penalty", 0.0)))

    return {"turnover": turnover, "trend": trend, "flip": flip}


def _expected_bonus(
    env_cfg: EnvConfig,
    *,
    pnl_gross: float,
    duration_bars: int,
    size: float | None = None,
    entry_price: float | None = None,
    side: int | None = None,
    trend_state: float | None = None,
    trend_strength: float | None = None,
) -> float:
    # Hold bonus
    alpha = float(getattr(env_cfg, "hold_bonus_alpha", 0.0))
    total = 0.0
    if alpha > 0.0:
        duration = max(1, int(duration_bars))
        raw_bonus = alpha * duration * pnl_gross
        if bool(getattr(env_cfg, "hold_bonus_positive_only", False)) and raw_bonus < 0.0:
            pass
        else:
            total += raw_bonus

    # Entry alignment bonus (aplicado apenas se duração >= trend_bonus_min_bars barras)
    if (
        size is not None
        and entry_price is not None
        and side is not None
        and trend_state is not None
        and trend_strength is not None
    ):
        min_bars = int(getattr(env_cfg, "trend_bonus_min_bars", 2))
        if (
            side != 0
            and np.isfinite(trend_state)
            and trend_state != 0.0
            and np.sign(trend_state) == np.sign(side)
            and int(duration_bars) >= max(1, min_bars)
        ):
            notional = abs(size * entry_price)
            coef_pct = max(0.0, float(getattr(env_cfg, "trend_bonus_coef_pct", 0.0)))
            base = notional * coef_pct if coef_pct > 0.0 else max(0.0, float(getattr(env_cfg, "trend_bonus_coef", 0.0)))
            if base > 0.0:
                strength = trend_strength
                if not np.isfinite(strength) or strength == 0.0:
                    strength = 1.0
                mult = float(getattr(env_cfg, "trend_bonus_entry_mult", getattr(env_cfg, "trend_penalty_entry_mult", 1.0)))
                total += base * abs(float(strength)) * max(1.0, mult)
    return float(total)


def _metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "total_pnl": 0.0,
            "avg_duration_bars": 0.0,
            "avg_duration_hours": 0.0,
        }

    pnls = df["pnl_net"].astype(float)
    wins = (pnls > 0).sum()
    losses = (pnls < 0).sum()
    total = len(df)
    pos_sum = pnls[pnls > 0].sum()
    neg_sum = pnls[pnls < 0].sum()
    if neg_sum < 0:
        profit_factor = float(pos_sum / abs(neg_sum)) if pos_sum > 0 else 0.0
    else:
        profit_factor = None

    return {
        "trades": int(total),
        "win_rate": float(wins / total) if total else 0.0,
        "profit_factor": profit_factor,
        "avg_pnl": float(pnls.mean()),
        "median_pnl": float(pnls.median()),
        "total_pnl": float(pnls.sum()),
        "avg_duration_bars": float(df["duration_bars"].mean()) if "duration_bars" in df else 0.0,
        "avg_duration_hours": float(df["duration_hours"].mean()) if "duration_hours" in df else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita ações e PnL do policy atual")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint específico (.pt). Quando omitido usa visualize.prefer",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Quantidade de dias para carregar (default: visualize.days ou 180)",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Mantém random_start do config (default: força False para avaliação determinística)",
    )
    return parser.parse_args()


def load_policy(cfg: dict, checkpoint: Path | None) -> tuple[MoEPolicy, Path]:
    prefer = cfg.get("visualize", {}).get("prefer", "latest")
    chosen = Path(checkpoint) if checkpoint else _find_checkpoint(prefer)

    # Monta política com o mesmo input_dim das features
    dummy_base = load_primary_series(cfg)
    dummy_confirm = load_confirm_series(cfg)
    dummy_dataset = prepare_dataset(dummy_base.tail(600), config=cfg, confirm_df=dummy_confirm)
    price_cols = ["open", "high", "low", "close", "volume"]
    input_dim = dummy_dataset.drop(columns=price_cols).shape[1]

    policy = build_policy(input_dim, cfg)
    state = torch.load(chosen, map_location="cpu")
    model_state = policy.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    policy.load_state_dict(filtered, strict=False)
    policy.eval()
    return policy, chosen


def main() -> None:
    args = parse_args()
    cfg = json.loads(CFG_PATH.read_text())
    report_cfg = cfg.get("reports", {})
    trade_cfg = report_cfg.get("trade_ledger", {})
    gating_cfg = report_cfg.get("gating_attribution", {})
    regime_cfg = report_cfg.get("regime_summary", {})

    data_cfg = cfg.get("data", {})
    timeframe = str(data_cfg.get("timeframe") or "").strip()
    if not timeframe:
        raise ValueError("Parâmetro obrigatório ausente: data.timeframe no config.json")
    days = args.days or int(cfg.get("visualize", {}).get("days", 180))
    base_df = load_primary_series(cfg)
    confirm_df = load_confirm_series(cfg)
    spread_window = int(cfg.get("data", {}).get("spread_window", 240))
    min_bars = max(spread_window + 20, bars_for_days(timeframe, days))
    base_trimmed = base_df.tail(min_bars)
    dataset = prepare_dataset(base_trimmed, config=cfg, confirm_df=confirm_df)
    if dataset.empty:
        dataset = prepare_dataset(base_df, config=cfg, confirm_df=confirm_df)
        if dataset.empty:
            raise RuntimeError(
                "Dataset vazio após aplicar filtros para auditoria. Aumente visualize.days/--days ou reduza spread_window."
            )
    price_cols = ["open", "high", "low", "close", "volume"]
    timestamps = dataset.index.to_list()
    price_df = dataset[price_cols].reset_index(drop=True)
    feat_df = dataset.drop(columns=price_cols).reset_index(drop=True)

    policy, checkpoint = load_policy(cfg, args.checkpoint)
    expert_labels = _expert_labels(cfg, policy.num_experts)

    env_cfg = EnvConfig(**cfg.get("env", {}))
    if not args.random_start:
        env_cfg.random_start = False
        env_cfg.window_bars = 0
    env = BTCMixtureEnv(price_df, feat_df, env_cfg, timestamps=timestamps)

    obs = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    action_counts = {0: 0, 1: 0, 2: 0}
    trade_pnls: List[float] = []
    trade_bars: List[int] = []

    bar_hours = hours_per_bar(timeframe)
    if "ret_vol_24" in feat_df.columns:
        vol_series = feat_df["ret_vol_24"].dropna()
        if vol_series.empty:
            q1 = q2 = np.nan
        else:
            q1 = float(vol_series.quantile(1 / 3))
            q2 = float(vol_series.quantile(2 / 3))
    else:
        q1 = q2 = np.nan

    ledger_rows: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    trade_id = 1
    current_idx = 0
    last_info: Dict[str, Any] = {"equity": float(env_cfg.init_equity)}

    done = False
    while not done:
        with torch.no_grad():
            dist, _, _ = policy(obs)
            action = torch.argmax(dist.probs, dim=-1).item()
            weights, mask = policy.gating(obs, top_k=policy.top_k)
        weights_np = weights.squeeze(0).detach().cpu().numpy()
        mask_np = mask.squeeze(0).detach().cpu().numpy()
        step_records.append({"idx": current_idx, "weights": weights_np.copy(), "mask": mask_np.copy()})

        next_obs, _, done, info = env.step(action)
        last_info = info
        action_counts[action] += 1

        if info.get("trade_closed"):
            entry_idx = int(info.get("trade_entry_idx", -1))
            exit_idx = int(info.get("trade_exit_idx", -1))
            entry_ts = str(info.get("trade_entry_ts", ""))
            exit_ts = str(info.get("trade_exit_ts", ""))
            entry_price = float(info.get("trade_entry_price", 0.0))
            exit_price = float(info.get("trade_exit_price", 0.0))
            pnl_net = float(info.get("trade_pnl", 0.0))
            pnl_gross = float(info.get("trade_gross", 0.0))
            cost = float(info.get("trade_cost", 0.0))
            bonus = float(info.get("trade_bonus", 0.0))
            penalty = float(info.get("trade_penalty", 0.0))
            duration_bars = int(info.get("trade_bars", 0))
            duration_hours = float(duration_bars) * bar_hours
            reason = str(info.get("trade_reason", ""))
            side_int = int(info.get("trade_side", 0))
            size = float(info.get("trade_size", 0.0))
            side_label = {1: "long", -1: "short"}.get(side_int, "flat")
            win_flag = "win" if pnl_net > 0 else ("loss" if pnl_net < 0 else "flat")

            trade_steps = [rec for rec in step_records if entry_idx <= rec["idx"] <= exit_idx]
            if trade_steps:
                weights_stack = np.stack([rec["weights"] for rec in trade_steps], axis=0)
                avg_weights = weights_stack.mean(axis=0)
                entry_weights = trade_steps[0]["weights"]
            else:
                avg_weights = np.zeros(policy.num_experts, dtype=float)
                entry_weights = avg_weights

            if 0 <= entry_idx < len(feat_df):
                entry_feat = feat_df.iloc[entry_idx]
            else:
                entry_feat = pd.Series(dtype=float)

            trend_state = float(entry_feat.get("htf_trend_state", np.nan)) if not entry_feat.empty else np.nan
            ret_vol = float(entry_feat.get("ret_vol_24", np.nan)) if not entry_feat.empty else np.nan
            vol_bucket = _vol_bucket(ret_vol, q1, q2)
            ml_prob = float(entry_feat.get("ml_prob_up", np.nan)) if not entry_feat.empty else np.nan
            ml_conf = float(entry_feat.get("ml_confidence", np.nan)) if not entry_feat.empty else np.nan

            penalty_components = _expected_penalty_components(
                env_cfg=env_cfg,
                size=size,
                entry_price=entry_price,
                exit_price=exit_price,
                side=side_int,
                reason=reason,
                trend_state=trend_state,
                trend_strength=float(entry_feat.get("htf_trend_strength", np.nan)) if not entry_feat.empty else np.nan,
            )
            penalty_expected = sum(penalty_components.values())
            penalty_delta = penalty - penalty_expected
            bonus_expected = _expected_bonus(
                env_cfg,
                pnl_gross=pnl_gross,
                duration_bars=duration_bars,
                size=size,
                entry_price=entry_price,
                side=side_int,
                trend_state=trend_state,
                trend_strength=float(entry_feat.get("htf_trend_strength", np.nan)) if not entry_feat.empty else np.nan,
            )
            bonus_delta = bonus - bonus_expected

            row: Dict[str, Any] = {
                "trade_id": trade_id,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "side": side_label,
                "size": size,
                "reason": reason,
                "duration_bars": duration_bars,
                "duration_hours": duration_hours,
                "pnl_net": pnl_net,
                "pnl_gross": pnl_gross,
                "cost": cost,
                "bonus": bonus,
                "trade_penalty": penalty,
                "penalty_expected": penalty_expected,
                "penalty_delta": penalty_delta,
                "penalty_turnover": penalty_components["turnover"],
                "penalty_trend": penalty_components["trend"],
                "penalty_flip": penalty_components["flip"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "win_flag": win_flag,
                "htf_trend_state": trend_state,
                "ret_vol_24": ret_vol,
                "vol_bucket": vol_bucket,
                "ml_prob_up": ml_prob,
                "ml_confidence": ml_conf,
                "bonus_expected": bonus_expected,
                "bonus_delta": bonus_delta,
            }

            for idx_expert, label in enumerate(expert_labels):
                row[f"avg_weight_{label}"] = float(avg_weights[idx_expert])
                row[f"entry_weight_{label}"] = float(entry_weights[idx_expert])

            ledger_rows.append(row)
            trade_id += 1
            trade_pnls.append(pnl_net)
            trade_bars.append(duration_bars)
            step_records = [rec for rec in step_records if rec["idx"] > exit_idx]

        obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        current_idx = min(current_idx + 1, len(price_df) - 1)

    steps = sum(action_counts.values())
    print("=== Auditoria do policy ===")
    print(f"checkpoint: {checkpoint}")
    print(f"dias avaliados: {days}")
    print("contagem de ações:")
    for k, v in action_counts.items():
        ratio = v / steps if steps else 0.0
        label = {0: "short", 1: "flat", 2: "long"}.get(k, str(k))
        print(f"  {label:5s}: {v:6d} ({ratio:.2%})")

    if trade_pnls:
        winners = sum(p > 0 for p in trade_pnls)
        print(f"trades fechados: {len(trade_pnls)} | win rate: {winners/len(trade_pnls):.2%}")
        print(f"PnL médio: {mean(trade_pnls):.5f} | mediana: {median(trade_pnls):.5f}")
        print(f"Duração média (barras): {mean(trade_bars):.2f} | mediana: {median(trade_bars)}")
    else:
        print("Nenhum trade fechado na janela avaliada.")

    print(f"Equity final: {float(last_info.get('equity', float('nan'))):.2f}")

    ledger_df = pd.DataFrame(ledger_rows)

    trade_enabled = bool(trade_cfg.get("enabled", True))
    gating_enabled = bool(gating_cfg.get("enabled", True))
    regime_enabled = bool(regime_cfg.get("enabled", True))

    penalty_tol = 1e-5
    bonus_tol = 1e-5
    penalty_alert_path = Path("src/strategies/exper_corr_pos/reports/train/penalty_alerts.csv")
    bonus_alert_path = Path("src/strategies/exper_corr_pos/reports/train/bonus_alerts.csv")
    _emit_delta_alert(
        ledger_df,
        column="penalty_delta",
        tolerance=penalty_tol,
        path=penalty_alert_path,
        label="penalidades",
        include_cols=[
            "trade_id",
            "entry_ts",
            "exit_ts",
            "side",
            "reason",
            "trade_penalty",
            "penalty_expected",
            "penalty_delta",
            "penalty_turnover",
            "penalty_trend",
            "penalty_flip",
        ],
    )
    _emit_delta_alert(
        ledger_df,
        column="bonus_delta",
        tolerance=bonus_tol,
        path=bonus_alert_path,
        label="bônus",
        include_cols=[
            "trade_id",
            "entry_ts",
            "exit_ts",
            "side",
            "reason",
            "trade_bonus",
            "bonus_expected",
            "bonus_delta",
            "trade_bars",
            "trade_gross",
        ],
    )

    if trade_enabled and not ledger_df.empty:
        trade_path = Path(trade_cfg.get("path", "src/strategies/exper_corr_pos/reports/train/trade_ledger.csv"))
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_df.to_csv(trade_path, index=False)
        print(f"Trade ledger salvo em {trade_path}")

    if gating_enabled and not ledger_df.empty:
        weight_cols = [c for c in ledger_df.columns if c.startswith("avg_weight_")]
        gating_path = Path(gating_cfg.get("path", "src/strategies/exper_corr_pos/reports/train/gating_attribution.csv"))
        gating_path.parent.mkdir(parents=True, exist_ok=True)
        cols_to_save = ["trade_id", "entry_ts", "exit_ts", "side", "pnl_net", "duration_bars", "duration_hours", "win_flag"] + weight_cols
        gating_df = ledger_df[cols_to_save]
        gating_df.to_csv(gating_path, index=False)

        summary = gating_df.groupby("win_flag")[weight_cols].mean().fillna(0.0)
        if not summary.empty:
            order = [flag for flag in ["win", "loss", "flat"] if flag in summary.index]
            summary = summary.loc[order]
            plot_df = summary.T
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_df.plot(kind="bar", ax=ax)
            ax.set_title("Peso médio por expert — agrupado por resultado")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Peso médio")
            ax.legend(title="resultado")
            fig.tight_layout()
            plot_path = Path(gating_cfg.get("plot_path", "src/strategies/exper_corr_pos/reports/train/gating_attribution.png"))
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            print(f"Atribuição de experts salva em {gating_path} (gráfico: {plot_path})")
        else:
            print("Atribuição de experts indisponível (dados insuficientes).")

    if regime_enabled and not ledger_df.empty:
        regime_path = Path(regime_cfg.get("path", "src/strategies/exper_corr_pos/reports/train/regime_summary.json"))
        regime_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload: Dict[str, Any] = {"overall": _metrics(ledger_df)}
        if "htf_trend_state" in ledger_df.columns:
            summary_payload["trend_state"] = {
                str(int(k)) if not pd.isna(k) else "nan": _metrics(df)
                for k, df in ledger_df.groupby("htf_trend_state")
            }
        if "vol_bucket" in ledger_df.columns:
            summary_payload["vol_bucket"] = {str(k): _metrics(df) for k, df in ledger_df.groupby("vol_bucket")}
        if "htf_trend_state" in ledger_df.columns and "vol_bucket" in ledger_df.columns:
            nested: Dict[str, Dict[str, Any]] = {}
            for (trend, bucket), df_subset in ledger_df.groupby(["htf_trend_state", "vol_bucket"]):
                trend_key = str(int(trend)) if not pd.isna(trend) else "nan"
                nested.setdefault(trend_key, {})[str(bucket)] = _metrics(df_subset)
            summary_payload["trend_vol"] = nested
        regime_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"Resumo de regimes salvo em {regime_path}")


def _emit_delta_alert(
    df: pd.DataFrame,
    *,
    column: str,
    tolerance: float,
    path: Path,
    label: str,
    include_cols: List[str],
) -> None:
    if column not in df.columns:
        return
    alerts = df[df[column].abs() > tolerance]
    if alerts.empty:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        print(f"[auditor] {label.capitalize()}: nenhum desvio acima de {tolerance:.1e}.")
        return
    subset_cols = [c for c in include_cols if c in alerts.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    alerts.sort_values(by="trade_id")[subset_cols].to_csv(path, index=False)
    worst = alerts[column].abs().max()
    print(
        f"[auditor] {label.capitalize()}: {len(alerts)} desvios > {tolerance:.1e} (máx={worst:.4f}). Detalhes em {path}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
