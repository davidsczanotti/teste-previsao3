import numpy as np
import pandas as pd
import pytest

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


def _make_constant_env_inputs(length: int, price: float = 100.0, *, trend_state: float = 0.0, trend_strength: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_df = pd.DataFrame(
        {
            "open": np.full(length, price),
            "high": np.full(length, price),
            "low": np.full(length, price),
            "close": np.full(length, price),
            "volume": np.full(length, 10.0),
        }
    )
    feat_df = pd.DataFrame(
        {
            "atr_14": np.full(length, 1.0),
            "htf_trend_state": np.full(length, trend_state),
            "htf_trend_strength": np.full(length, trend_strength),
        }
    )
    return price_df, feat_df


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
        equity_floor_pct=0.95,
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


def test_trend_penalty_penalizes_contra_trend_positions():
    length = 5
    base = np.full(length, 100.0)
    price_df = pd.DataFrame(
        {
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "volume": np.ones(length),
        }
    )
    feat_df = pd.DataFrame(
        {
            "atr_14": np.ones(length),
            "htf_trend_state": np.ones(length),  # tendencia positiva
        }
    )
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.5,
        stop_atr_mult=1.0,
        trail_atr_mult=1.0,
        trend_penalty_coef=0.25,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=length,
        idle_penalty_factor=0.0,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    # Long segue a tendência -> não deveria penalizar
    _, reward_long, _, _ = env.step(2)
    env.reset()
    # Short vai contra a tendência -> recebe penalidade negativa
    _, reward_short, _, _ = env.step(0)
    assert reward_long == pytest.approx(0.0, abs=1e-6)
    assert reward_short < reward_long
    assert reward_short == pytest.approx(-cfg.trend_penalty_coef, abs=1e-6)


def test_turnover_penalty_pct_scales_with_notional():
    price_df, feat_df = _make_constant_env_inputs(10, price=100.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.5,
        fee_pct=0.0,
        slippage_pct=0.0,
        turnover_penalty=0.0,
        turnover_penalty_pct=0.05,
        random_start=False,
        window_bars=10,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)
    _, _, _, info = env.step(1)
    assert info["trade_closed"] is True
    expected_notional = 100.0 * cfg.position_size
    expected_penalty = expected_notional * cfg.turnover_penalty_pct
    assert info["trade_penalty"] == pytest.approx(expected_penalty, rel=1e-6)


def test_flip_exit_penalty_pct_applies_only_on_flip():
    price_df, feat_df = _make_constant_env_inputs(10, price=80.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.4,
        fee_pct=0.0,
        slippage_pct=0.0,
        flip_exit_penalty=0.0,
        flip_exit_penalty_pct=0.05,
        random_start=False,
        window_bars=10,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)
    _, _, _, info = env.step(0)
    assert info["trade_closed"] is True
    expected_notional = 80.0 * cfg.position_size
    expected_penalty = expected_notional * cfg.flip_exit_penalty_pct
    assert info["trade_penalty"] == pytest.approx(expected_penalty, rel=1e-6)


def test_trend_penalty_pct_scales_with_strength_and_notional():
    price_df, feat_df = _make_constant_env_inputs(10, price=120.0, trend_state=1.0, trend_strength=2.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.25,
        fee_pct=0.0,
        slippage_pct=0.0,
        trend_penalty_coef=0.0,
        trend_penalty_coef_pct=0.02,
        random_start=False,
        window_bars=10,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(0)  # abre short contra tendencia positiva
    _, _, _, info = env.step(1)  # fecha
    assert info["trade_closed"] is True
    notional = 120.0 * cfg.position_size
    expected_penalty = notional * cfg.trend_penalty_coef_pct * abs(feat_df.loc[0, "htf_trend_strength"])
    assert info["trade_penalty"] == pytest.approx(expected_penalty, rel=1e-6)


def test_hold_bonus_matches_alpha_formula():
    price_df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [100.0, 101.0, 102.0, 103.0],
            "low": [100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": np.full(4, 10.0),
        }
    )
    feat_df = pd.DataFrame(
        {
            "atr_14": np.ones(4),
            "htf_trend_state": np.zeros(4),
            "htf_trend_strength": np.zeros(4),
        }
    )
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.5,
        fee_pct=0.0,
        slippage_pct=0.0,
        hold_bonus_alpha=0.1,
        hold_bonus_positive_only=False,
        random_start=False,
        window_bars=4,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)
    env.step(2)
    _, _, _, info = env.step(1)
    assert info["trade_closed"] is True
    expected_bonus = cfg.hold_bonus_alpha * info["trade_bars"] * info["trade_gross"]
    assert info["trade_bonus"] == pytest.approx(expected_bonus, rel=1e-6)
