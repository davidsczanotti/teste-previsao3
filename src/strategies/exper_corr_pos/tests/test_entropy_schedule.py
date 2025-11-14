from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import numpy as np
import pandas as pd

from src.strategies.exper_corr_pos.train import train_agent


def _make_dummy_ohlcv(n: int = 10) -> pd.DataFrame:
    base = np.linspace(100.0, 101.0, n)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.25,
            "volume": np.full(n, 10.0),
        },
        index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"),
    )


def test_entropy_schedule_keeps_exploration_high_initially(monkeypatch):
    # Usa um config mínimo e overrides para isolar o agendamento de entropia.
    cfg_path = Path("src/strategies/exper_corr_pos/config.json")
    base_cfg: Dict[str, Any] = json.loads(cfg_path.read_text())

    # Reduzimos episódios e evitamos avaliações para manter o teste leve.
    overrides: Dict[str, Any] = {
        "data": {
            "base_symbol": "FAKEBTC",
            "confirm_symbol": None,
            "timeframe": "1d",
            "lookback_days": 30,
        },
        "train": {
            "episodes": 4,
            "rollout_steps": 1,
            "eval_every": 0,
            # Schedule longo: ent_decay_episodes >> episodes
            "entropy_coef_start": 0.02,
            "entropy_coef_end": 0.005,
            "entropy_decay_episodes": 100,
        },
    }

    # Dummy dataset para evitar acesso ao cache real
    def _fake_load_primary(_cfg: Dict[str, Any]) -> pd.DataFrame:
        return _make_dummy_ohlcv(12)

    def _fake_load_confirm(_cfg: Dict[str, Any]):
        return None

    def _fake_prepare_dataset(df: pd.DataFrame, *, config=None, confirm_df=None) -> pd.DataFrame:
        out = df.copy()
        out["atr_14"] = 1.0
        return out

    entropy_values: List[float] = []

    class DummyPPOTrainer:
        def __init__(self, policy, ppo_cfg, device=None, lb_coef: float = 0.01) -> None:
            self.policy = policy
            self.cfg = ppo_cfg
            self.device = device
            self.lb_coef = lb_coef

        def train_step(self, env, rollout_steps: int):
            # Registra o coeficiente de entropia usado neste episódio.
            entropy_values.append(float(self.cfg.entropy_coef))
            # Retorna métricas mínimas para compatibilidade.
            return {"avg_reward": 0.0, "sum_reward": 0.0}

    # Monkeypatch dos loaders e do trainer para evitar treino pesado.
    import src.strategies.exper_corr_pos.train as train_mod

    monkeypatch.setattr(train_mod, "load_primary_series", _fake_load_primary)
    monkeypatch.setattr(train_mod, "load_confirm_series", _fake_load_confirm)
    monkeypatch.setattr(train_mod, "prepare_dataset", _fake_prepare_dataset)
    monkeypatch.setattr(train_mod, "PPOTrainer", DummyPPOTrainer)

    # Executa o laço de treino com o schedule configurado
    train_agent(base_cfg, cfg_path=cfg_path, overrides=overrides, record_manifest=False, enable_plots=False, disable_wandb=True)

    # Com schedule longo (decay_episodes >> episodes), os coeficientes devem
    # permanecer próximos do valor inicial e decair apenas levemente.
    assert len(entropy_values) == overrides["train"]["episodes"]
    start = overrides["train"]["entropy_coef_start"]
    end = overrides["train"]["entropy_coef_end"]

    # Primeiro episódio ainda deve estar bem próximo do start
    assert entropy_values[0] == pytest.approx(
        start + (end - start) * (1 / overrides["train"]["entropy_decay_episodes"]),
        rel=1e-6,
    )
    # Último episódio ainda deve estar acima do valor final da schedule
    assert entropy_values[-1] > end
    # E a sequência deve ser monotonicamente decrescente (entropia alta no início, caindo aos poucos)
    assert all(x >= y for x, y in zip(entropy_values, entropy_values[1:]))
