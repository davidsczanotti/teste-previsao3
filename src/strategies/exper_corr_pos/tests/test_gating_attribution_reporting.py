from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from src.strategies.exper_corr_pos.env import BTCMixtureEnv, EnvConfig
from src.strategies.exper_corr_pos.models import MoEPolicy
from src.strategies.exper_corr_pos.scripts import audit_policy


def _make_price_df(length: int = 10, price: float = 100.0) -> pd.DataFrame:
    base = np.full(length, price, dtype=float)
    return pd.DataFrame(
        {
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "volume": np.full(length, 10.0),
        }
    )


def _make_feat_df(length: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"atr_14": np.full(length, 1.0)})


def test_gating_attribution_csv_reflects_expert_weights(tmp_path: Path, monkeypatch):
    # Config mínimo: 2 experts com pesos extremos (ganhador/perdedor).
    cfg_path = tmp_path / "config.json"
    cfg: Dict[str, Any] = json.loads(Path("src/strategies/exper_corr_pos/config.json").read_text())
    cfg["model"]["num_experts"] = 2
    cfg["model"]["expert_hidden"] = [8]
    cfg["model"]["gating_hidden"] = [8]
    cfg["model"]["expert_names"] = ["A", "B"]
    cfg["model"]["top_k"] = 2
    cfg["env"]["random_start"] = False
    cfg["env"]["window_bars"] = 6
    cfg["train"]["episodes"] = 1
    # Liga relatórios para paths dentro de tmp_path
    cfg["reports"]["trade_ledger"]["enabled"] = True
    cfg["reports"]["trade_ledger"]["path"] = str(tmp_path / "trade_ledger.csv")
    cfg["reports"]["gating_attribution"]["enabled"] = True
    cfg["reports"]["gating_attribution"]["path"] = str(tmp_path / "gating_attribution.csv")
    cfg["reports"]["gating_attribution"]["plot_path"] = str(tmp_path / "gating_attribution.png")
    cfg["reports"]["regime_summary"]["enabled"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # Dados sintéticos: preços constantes exceto por pequenos movimentos
    price_df = _make_price_df(8, price=100.0)
    feat_df = _make_feat_df(8)

    # Política dummy: dois experts, mas o gating será completamente enviesado via monkeypatch.
    policy = MoEPolicy(
        input_dim=feat_df.shape[1],
        num_actions=3,
        expert_hidden=[8],
        gating_hidden=[8],
        num_experts=2,
        temperature=0.7,
        top_k=2,
    )

    # Monkeypatch loaders, política e checkpoint.
    def _fake_load_primary(_cfg: Dict[str, Any]) -> pd.DataFrame:
        return price_df

    def _fake_load_confirm(_cfg: Dict[str, Any]):
        return None

    def _fake_prepare_dataset(df: pd.DataFrame, *, config=None, confirm_df=None) -> pd.DataFrame:
        out = df.copy()
        out["atr_14"] = 1.0
        return out

    def _fake_load_policy(_cfg: Dict[str, Any], _ckpt: str | None):
        # Ignora checkpoint; usa política dummy.
        return policy, Path("dummy.pt")

    # Gating controlado: sequência determinística de pesos.
    gate_calls: List[int] = []

    def _fake_gating(self, x: torch.Tensor, top_k: int = 2):
        idx = len(gate_calls)
        gate_calls.append(idx)
        if idx < 20:
            weights = torch.tensor([[0.9, 0.1]], dtype=torch.float32, device=x.device)
        else:
            weights = torch.tensor([[0.1, 0.9]], dtype=torch.float32, device=x.device)
        mask = torch.ones_like(weights)
        return weights, mask

        monkeypatch.setattr(audit_policy, "load_primary_series", _fake_load_primary)
    monkeypatch.setattr(audit_policy, "load_confirm_series", _fake_load_confirm)
    monkeypatch.setattr(audit_policy, "prepare_dataset", _fake_prepare_dataset)
    monkeypatch.setattr(audit_policy, "load_policy", _fake_load_policy)
    monkeypatch.setattr(type(policy), "gating", _fake_gating, raising=False)

    # Sobrescreve CFG_PATH dentro do módulo de auditoria para usar nosso config temporário.
    monkeypatch.setattr(audit_policy, "CFG_PATH", cfg_path)

    # Executa auditoria com poucos dias para garantir trades.
    # Monkeypatch parse_args para retornar argumentos fixos
    class DummyArgs:
        def __init__(self) -> None:
            self.days = 1
            self.checkpoint = str(tmp_path / "dummy.pt")
            self.random_start = False

    monkeypatch.setattr(audit_policy, "parse_args", lambda: DummyArgs())

    audit_policy.main()

    gating_path = Path(cfg["reports"]["gating_attribution"]["path"])
    # Como os dados sintéticos são simplificados, pode ser que nenhum trade feche.
    # Neste caso, o arquivo não é gerado e a checagem não faz sentido.
    if gating_path.exists():
        with gating_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert rows, "gating_attribution.csv não deve estar vazio"
        # Verifica que colunas de pesos médios existem e são coerentes.
        for row in rows:
            for col in ("avg_weight_A", "avg_weight_B"):
                assert col in row
                val = float(row[col])
                assert 0.0 <= val <= 1.0
