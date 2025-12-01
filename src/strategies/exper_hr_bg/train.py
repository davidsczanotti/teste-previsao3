from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import build_dataset


CFG_PATH = Path("src/strategies/exper_hr_bg/config.json")


def train_main(config: Dict[str, Any]) -> Dict[str, Any]:
  ds = build_dataset(config)
  X = ds.features
  y = ds.labels
  n = len(X)
  if n < 500:
    raise ValueError(f"Dataset muito pequeno para treino (n={n}). Ajuste janela/horizonte.")

  train_cfg = config.get("train", {}) or {}
  max_samples = int(train_cfg.get("max_samples", 100000))
  if n > max_samples:
    idx = np.linspace(0, n - 1, max_samples, dtype=int)
    X = X[idx]
    y = y[idx]
    n = max_samples

  test_frac = float(train_cfg.get("test_fraction", 0.3))
  split = int(n * (1.0 - test_frac))
  X_train, y_train = X[:split], y[:split]

  # sample weights para lidar com desbalanceamento e dar mais peso a up/down que "none"
  class_counts = np.bincount(y_train, minlength=3).astype(np.float64)
  class_counts[class_counts == 0.0] = 1.0
  inv = 1.0 / class_counts
  weights = inv / inv.mean()
  sample_weight = weights[y_train]

  model_cfg = config.get("model", {}) or {}
  model = HistGradientBoostingClassifier(
    learning_rate=float(model_cfg.get("learning_rate", 0.05)),
    max_depth=int(model_cfg.get("max_depth", 5)),
    max_leaf_nodes=int(model_cfg.get("max_leaf_nodes", 31)),
    min_samples_leaf=int(model_cfg.get("min_samples_leaf", 50)),
    max_iter=int(model_cfg.get("max_iter", 300)),
    random_state=int(train_cfg.get("random_state", 42)),
  )
  model.fit(X_train, y_train, sample_weight=sample_weight)

  # métricas simples no treino
  train_pred = model.predict(X_train)
  train_acc = float((train_pred == y_train).mean())

  outdir = Path(train_cfg.get("outdir", "src/strategies/exper_hr_bg/reports/train"))
  outdir.mkdir(parents=True, exist_ok=True)

  model_path = outdir / "model.pkl"
  with model_path.open("wb") as f:
    pickle.dump(model, f)

  metrics_path = outdir / "metrics.csv"
  with metrics_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_samples", "train_size", "train_acc"])
    writer.writerow([n, split, train_acc])

  return {
    "n_samples": int(n),
    "train_size": int(split),
    "train_acc": train_acc,
    "model_path": str(model_path),
  }


def main() -> None:
  cfg = json.loads(CFG_PATH.read_text())
  res = train_main(cfg)
  print("[exper_hr_bg.train] resumo:", res)


if __name__ == "__main__":
  main()

