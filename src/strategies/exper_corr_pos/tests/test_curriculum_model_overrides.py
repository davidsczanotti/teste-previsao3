import numpy as np
import pandas as pd

from src.strategies.exper_corr_pos.env import BTCMixtureEnv, EnvConfig
from src.strategies.exper_corr_pos.models import MoEPolicy
from src.strategies.exper_corr_pos.train import _apply_curriculum_phase
from src.strategies.exper_corr_pos.utils_cfg import build_policy
import json
from pathlib import Path


def _make_env(n: int = 10):
    price_df = pd.DataFrame(
        {
            "open": np.linspace(100.0, 101.0, n),
            "high": np.linspace(100.5, 101.5, n),
            "low": np.linspace(99.5, 100.5, n),
            "close": np.linspace(100.25, 101.25, n),
            "volume": np.full(n, 10.0),
        }
    )
    feat_df = pd.DataFrame({"atr_14": np.full(n, 1.0)})
    env = BTCMixtureEnv(price_df, feat_df, EnvConfig())
    return env


def test_apply_curriculum_phase_updates_model_temperature_and_topk():
    env = _make_env()
    policy = MoEPolicy(input_dim=1, num_actions=3, expert_hidden=[8], gating_hidden=[8], num_experts=4, top_k=2)

    curriculum = {
        "phases": [
            {
                "until_episode": 10,
                "model": {"temperature": 1.1, "top_k": 2},
            },
            {
                "until_episode": 20,
                "model": {"temperature": 0.7, "top_k": 1},
            },
        ]
    }

    # Phase 1
    _ = _apply_curriculum_phase(curriculum, env, episode=5, default_rollout=64, policy=policy)
    assert abs(policy.gating.temperature - 1.1) < 1e-6
    assert policy.top_k == 2

    # Phase 2
    _ = _apply_curriculum_phase(curriculum, env, episode=15, default_rollout=64, policy=policy)
    assert abs(policy.gating.temperature - 0.7) < 1e-6
    assert policy.top_k == 1


def test_config_curriculum_final_uses_topk_two():
    # Garante que o config.json está configurado para usar top_k=2
    # na fase final (Mario combinando pelo menos dois experts).
    cfg_path = Path("src/strategies/exper_corr_pos/config.json")
    cfg = json.loads(cfg_path.read_text())

    # Config global do modelo deve ter top_k=2
    assert cfg["model"]["top_k"] == 2

    # build_policy deve respeitar esse top_k inicial
    dummy_env = _make_env()
    input_dim = 1  # _make_env usa features com 1 coluna
    policy = build_policy(input_dim, cfg)
    assert policy.top_k == 2

    # Curriculum final também deve aplicar top_k=2 quando episódio > último until_episode
    curriculum_cfg = cfg["train"]["curriculum"]
    last_until = max(phase["until_episode"] for phase in curriculum_cfg["phases"])
    _ = _apply_curriculum_phase(curriculum_cfg, dummy_env, episode=last_until + 10, default_rollout=64, policy=policy)
    assert policy.top_k == 2
