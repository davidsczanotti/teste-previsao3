from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from .data import build_dataset
from .models import CEClassifier
from ...utils.metrics import calculate_metrics


CFG_PATH = Path("src/strategies/exper_ce_ce/config.json")


def run_backtest(config: Dict[str, Any]) -> Dict[str, Any]:
  data = build_dataset(config)
  n = len(data.features)
  if n < 100:
    raise ValueError(f"Dataset pequeno demais para backtest (n={n}).")

  test_frac = float(config.get("backtest", {}).get("test_fraction", 0.3))
  split = int(n * (1.0 - test_frac))
  X = data.features
  idx = data.index
  close = data.close

  X_train, X_test = X[:split], X[split:]
  idx_test = idx[split:]
  close_test = close[split:]

  model_cfg = config.get("model", {})
  hidden_sizes = model_cfg.get("hidden_sizes", [64, 32])
  dropout = float(model_cfg.get("dropout", 0.0))

  # carregar normalização e modelo treinado
  train_cfg = config.get("train", {})
  outdir = Path(train_cfg.get("outdir", "src/strategies/exper_ce_ce/reports/train"))
  best_path = outdir / "model_best.pt"
  norm_path = outdir / "normalization.npz"
  if not best_path.exists() or not norm_path.exists():
    raise FileNotFoundError("Modelo treinado/normalização não encontrados. Rode o train.py antes.")

  norm = np.load(norm_path)
  mean = norm["mean"]
  std = norm["std"]
  X_test_n = (X_test - mean) / std

  device = torch.device(train_cfg.get("device", "cpu"))
  model = CEClassifier(input_dim=X.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(device)
  state = torch.load(best_path, map_location=device)
  model.load_state_dict(state)
  model.eval()

  with torch.no_grad():
    logits = model(torch.tensor(X_test_n, dtype=torch.float32, device=device))
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1).cpu().numpy()
    max_probs, top_idx = probs.max(dim=-1)
    # margem = prob_top1 - prob_top2
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    margins = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu().numpy()
    max_probs_np = max_probs.cpu().numpy()

  # Monta trades simples: se pred=1 → long, pred=2 → short, pred=0 → nenhum trade
  # Entrada em t+1, saída em t+H (horizon configurado)
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
  closes_full = data.close
  index_full = data.index

  # índices relativos ao dataset completo
  test_positions = np.arange(split, n)
  for pred, prob, margin, ds_idx in zip(preds, max_probs_np, margins, test_positions):
    # 0 = "none" (não opera)
    if pred == 0:
      continue
    # filtro por confiança mínima e margem entre classes
    if prob < min_prob or margin < min_margin:
      continue
    entry_idx = ds_idx + 1
    exit_idx = ds_idx + horizon
    if exit_idx >= len(closes_full):
      break
    entry_price = float(closes_full[entry_idx])
    exit_price = float(closes_full[exit_idx])
    side = 1 if pred == 1 else -1
    ret = side * (exit_price / entry_price - 1.0)
    # custo round-trip aproximado
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
    "strategy": "exper_ce_ce",
    "symbol": data_cfg.get("base_symbol", "BTCUSDT"),
    "interval": data_cfg.get("timeframe", "1h"),
    "n_samples": int(n),
    "n_trades": int(metrics.get("total_trades", 0)),
    "metrics": metrics,
  }
  return result


def main() -> None:
  cfg = json.loads(CFG_PATH.read_text())
  outdir = Path(cfg.get("backtest", {}).get("outdir", "src/strategies/exper_ce_ce/reports/backtest"))
  outdir.mkdir(parents=True, exist_ok=True)
  res = run_backtest(cfg)
  out_json = outdir / f"exper_ce_ce_{res['symbol']}_{res['interval']}.json"
  out_json.write_text(json.dumps(res, indent=2))
  print(f"[exper_ce_ce.backtest] Resultado salvo em {out_json}")


if __name__ == "__main__":
  main()
