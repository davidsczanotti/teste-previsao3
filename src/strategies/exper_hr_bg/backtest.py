from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from .data import build_dataset
from ...utils.metrics import calculate_metrics


CFG_PATH = Path("src/strategies/exper_hr_bg/config.json")


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
  ds = build_dataset(config)
  X = ds.features
  y = ds.labels
  close = ds.close
  idx = ds.index
  n = len(X)
  if n < 500:
    raise ValueError(f"Dataset pequeno demais para backtest (n={n}).")

  train_cfg = config.get("train", {}) or {}
  test_frac = float(train_cfg.get("test_fraction", 0.3))
  split = int(n * (1.0 - test_frac))

  X_test = X[split:]
  y_test = y[split:]
  close_full = close
  index_full = idx

  model_path = Path(train_cfg.get("outdir", "src/strategies/exper_hr_bg/reports/train")) / "model.pkl"
  if not model_path.exists():
    raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Rode o train.py antes.")

  with model_path.open("rb") as f:
    model = pickle.load(f)

  probs = model.predict_proba(X_test)
  preds = probs.argmax(axis=1)
  # prob+margin para decisões
  sorted_probs = np.sort(probs, axis=1)[:, ::-1]
  max_probs = sorted_probs[:, 0]
  margins = sorted_probs[:, 0] - sorted_probs[:, 1]

  data_cfg = config.get("data", {})
  label_cfg = data_cfg.get("label", {}) or {}
  horizon = int(label_cfg.get("horizon_bars", 6))
  fee_pct = float(config.get("env", {}).get("fee_pct", 0.001))
  slippage_pct = float(config.get("env", {}).get("slippage_pct", 0.0003))
  cost_per_side = fee_pct + slippage_pct

  bt_cfg = config.get("backtest", {}) or {}
  min_prob = float(bt_cfg.get("min_prob", 0.0))
  min_margin = float(bt_cfg.get("min_margin", 0.0))

  trades: List[Dict[str, Any]] = []
  test_positions = np.arange(split, n)
  for cls, prob, margin, ds_idx in zip(preds, max_probs, margins, test_positions):
    # classes: 0=none,1=up,2=down
    if cls == 0:
      continue
    if prob < min_prob or margin < min_margin:
      continue
    entry_idx = ds_idx + 1
    exit_idx = ds_idx + horizon
    if exit_idx >= len(close_full):
      break
    entry_price = float(close_full[entry_idx])
    exit_price = float(close_full[exit_idx])
    side = 1 if cls == 1 else -1
    ret = side * (exit_price / entry_price - 1.0)
    cost = 2.0 * cost_per_side
    pnl = ret - cost
    trades.append(
      {
        "pnl": pnl,
        "entry_idx": int(entry_idx),
        "exit_idx": int(exit_idx),
        "entry_ts": str(index_full[entry_idx]),
        "exit_ts": str(index_full[exit_idx]),
        "side": int(side),
        "entry_price": entry_price,
        "exit_price": exit_price,
      }
    )

  metrics = calculate_metrics(trades)
  result = {
    "strategy": "exper_hr_bg",
    "symbol": data_cfg.get("base_symbol", "BTCUSDT"),
    "interval": data_cfg.get("timeframe", "1h"),
    "n_samples": int(n),
    "n_trades": int(metrics.get("total_trades", 0)),
    "metrics": metrics,
  }
  return result


def main() -> None:
  cfg = json.loads(CFG_PATH.read_text())
  outdir = Path(cfg.get("backtest", {}).get("outdir", "src/strategies/exper_hr_bg/reports/backtest"))
  outdir.mkdir(parents=True, exist_ok=True)
  res = run_backtest(cfg)
  out_json = outdir / f"exper_hr_bg_{res['symbol']}_{res['interval']}.json"
  out_json.write_text(json.dumps(res, indent=2))
  print(f"[exper_hr_bg.backtest] Resultado salvo em {out_json}")


if __name__ == "__main__":
  main()

