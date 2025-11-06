import numpy as np
import pandas as pd

from src.strategies.exper_corr_pos.env import BTCMixtureEnv, EnvConfig


def _make_price_df(length: int) -> pd.DataFrame:
    base = np.linspace(100.0, 102.0, length)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.full(length, 10.0),
        }
    )


def _make_features_df(length: int) -> pd.DataFrame:
    return pd.DataFrame({"atr_14": np.full(length, 1.0)})


def test_env_reset_and_step_shapes():
    length = 20
    price_df = _make_price_df(length)
    feat_df = _make_features_df(length)
    env = BTCMixtureEnv(price_df, feat_df, EnvConfig())

    obs = env.reset()
    assert obs.shape[0] == feat_df.shape[1]

    next_obs, reward, done, info = env.step(2)
    assert next_obs.shape[0] == feat_df.shape[1]
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "equity" in info
    assert "position" in info


def test_env_handles_position_flip():
    length = 30
    price_df = _make_price_df(length)
    feat_df = _make_features_df(length)
    cfg = EnvConfig(turnover_penalty=0.1, init_equity=1000.0)
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    # Enter long
    env.step(2)
    # Flip to short
    _, _, _, info = env.step(0)
    assert "trade_pnl" in info


def test_env_triggers_ruin_on_equity_floor():
    length = 15
    price_df = _make_price_df(length)
    price_df["close"] = np.linspace(100.0, 50.0, length)
    price_df["open"] = price_df["close"]
    feat_df = _make_features_df(length)
    cfg = EnvConfig(
        init_equity=1000.0,
        equity_floor_pct=0.5,
        max_drawdown_pct=0.6,
        drawdown_kill_bars=1,
        position_size=1.0,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(2)
        if done:
            assert info.get("ruined") is True
            break


def test_env_turnover_penalty_applies_on_flip():
    length = 20
    price_df = _make_price_df(length)
    feat_df = _make_features_df(length)
    cfg = EnvConfig(turnover_penalty=1.0, init_equity=1000.0)
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)
    _, reward, _, _ = env.step(0)
    assert reward <= 0
