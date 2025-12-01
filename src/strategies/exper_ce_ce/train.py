from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import build_dataset
from .models import CEClassifier, TrainConfig


CFG_PATH = Path("src/strategies/exper_ce_ce/config.json")


def _set_seeds(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def train_main(config: Dict[str, Any]) -> Dict[str, Any]:
  train_cfg = config.get("train", {})
  seed = int(train_cfg.get("seed", 42))
  _set_seeds(seed)

  dataset = build_dataset(config)
  X = dataset.features
  y = dataset.labels

  n = len(X)
  if n < 100:
    raise ValueError(f"Dataset muito pequeno para treino supervisionado (n={n}). Ajuste squeezes/thresholds.")

  max_samples = int(train_cfg.get("max_samples", 50000))
  if n > max_samples:
    idx = np.linspace(0, n - 1, max_samples, dtype=int)
    X = X[idx]
    y = y[idx]
    n = max_samples

  val_frac = float(train_cfg.get("val_fraction", 0.2))
  split = int(n * (1.0 - val_frac))
  X_train, X_val = X[:split], X[split:]
  y_train, y_val = y[:split], y[split:]

  device = torch.device(train_cfg.get("device", "cpu"))

  # normalização simples por média/std do treino
  mean = X_train.mean(axis=0, keepdims=True)
  std = X_train.std(axis=0, keepdims=True)
  std[std == 0.0] = 1.0
  X_train_n = (X_train - mean) / std
  X_val_n = (X_val - mean) / std

  train_ds = TensorDataset(
    torch.tensor(X_train_n, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long),
  )
  val_ds = TensorDataset(
    torch.tensor(X_val_n, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long),
  )

  batch_size = int(train_cfg.get("batch_size", 512))
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

  hidden_sizes = config.get("model", {}).get("hidden_sizes", [64, 32])
  dropout = float(config.get("model", {}).get("dropout", 0.0))

  model = CEClassifier(input_dim=X.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(device)

  # pesos de classe para contornar desbalanceamento (0=none,1=up,2=down)
  class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
  class_counts[class_counts == 0.0] = 1.0
  inv = 1.0 / class_counts
  weights = inv / inv.mean()
  class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

  cfg_t = TrainConfig(
    epochs=int(train_cfg.get("epochs", 30)),
    batch_size=batch_size,
    learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
    weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    device=device,
  )

  optim = torch.optim.AdamW(model.parameters(), lr=cfg_t.learning_rate, weight_decay=cfg_t.weight_decay)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  outdir = Path(train_cfg.get("outdir", "src/strategies/exper_ce_ce/reports/train"))
  outdir.mkdir(parents=True, exist_ok=True)
  metrics_path = outdir / "metrics.csv"
  with metrics_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

  best_val_acc = -1.0
  best_path = outdir / "model_best.pt"
  norm_path = outdir / "normalization.npz"

  for epoch in range(1, cfg_t.epochs + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in train_loader:
      xb = xb.to(device)
      yb = yb.to(device)
      logits = model(xb)
      loss = criterion(logits, yb)
      optim.zero_grad()
      loss.backward()
      optim.step()
      train_loss += loss.item() * yb.size(0)
      preds = logits.argmax(dim=-1)
      correct += (preds == yb).sum().item()
      total += yb.size(0)
    train_loss /= max(1, total)
    train_acc = correct / max(1, total)

    model.eval()
    val_loss = 0.0
    v_correct = 0
    v_total = 0
    with torch.no_grad():
      for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        val_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=-1)
        v_correct += (preds == yb).sum().item()
        v_total += yb.size(0)
    val_loss /= max(1, v_total)
    val_acc = v_correct / max(1, v_total)

    with metrics_path.open("a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      torch.save(model.state_dict(), best_path)
      np.savez(norm_path, mean=mean, std=std)

  return {
    "n_samples": int(len(dataset.features)),
    "train_size": int(len(y_train)),
    "val_size": int(len(y_val)),
    "best_val_acc": float(best_val_acc),
    "best_model_path": str(best_path),
    "norm_path": str(norm_path),
  }


def main() -> None:
  cfg = json.loads(CFG_PATH.read_text())
  res = train_main(cfg)
  print("[exper_ce_ce.train] resumo:", res)


if __name__ == "__main__":
  main()

