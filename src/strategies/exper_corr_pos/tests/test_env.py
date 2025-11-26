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


def test_min_hold_rule_blocks_early_close_and_flip():
    length = 10
    price_df = _make_price_df(length)
    feat_df = _make_features_df(length)
    cfg = EnvConfig(
        init_equity=1000.0,
        min_hold_bars_enabled=True,
        min_hold_bars=3,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    # Open long at t=0
    _, _, _, info = env.step(2)
    assert info["position"] == 1
    # t=1: try to go flat (should be blocked)
    _, _, _, info = env.step(1)
    assert info["position"] == 1
    assert info["trade_closed"] is False
    # t=2: try to flip short (should be blocked)
    _, _, _, info = env.step(0)
    assert info["position"] == 1
    assert info["trade_closed"] is False
    # t=3: now minimum reached; try to go flat (should allow close)
    _, _, _, info = env.step(1)
    assert info["position"] == 0
    assert info["trade_closed"] is True


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


def test_short_trade_penalty_applies_for_fast_closes():
    price_df, feat_df = _make_constant_env_inputs(5, price=100.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=5,
        short_trade_penalty=5.0,
        short_trade_min_bars=3,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)  # abre long
    _, _, _, info = env.step(1)  # fecha antes do mínimo
    assert info["trade_short_penalty"] == pytest.approx(cfg.short_trade_penalty)
    assert info["trade_penalty"] >= cfg.short_trade_penalty


def test_giveback_penalty_triggers_after_large_peak():
    prices = [100.0, 120.0, 130.0, 101.0, 101.0]
    price_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(prices)),
        }
    )
    feat_df = pd.DataFrame({"atr_14": np.ones(len(prices))})
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=len(prices),
        giveback_threshold_pct=0.5,
        giveback_penalty_pct=0.1,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    info = {}
    for action in (2, 2, 2, 1):
        _, _, _, info = env.step(action)
        if info.get("trade_closed"):
            break
    peak_pnl = (130.0 - 100.0) * cfg.position_size
    expected_giveback = peak_pnl * cfg.giveback_penalty_pct
    assert info["trade_giveback_penalty"] == pytest.approx(expected_giveback)
    assert info["trade_penalty"] == pytest.approx(info["trade_giveback_penalty"])


def test_profit_tax_applies_on_winning_trade():
    prices = [100.0, 110.0, 110.0]
    price_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(prices)),
        }
    )
    feat_df = pd.DataFrame({"atr_14": np.ones(len(prices))})
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        profit_tax_pct=0.15,
        reward_scale_divisor=1.0,
        random_start=False,
        window_bars=len(prices),
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)  # open long
    env.step(2)  # hold
    _, _, _, info = env.step(1)  # close to flat
    assert info["trade_closed"] is True
    expected_pnl = (prices[2] - prices[0]) * cfg.position_size
    expected_tax = expected_pnl * cfg.profit_tax_pct
    expected_net = expected_pnl - expected_tax
    assert info["trade_tax"] == pytest.approx(expected_tax, rel=1e-6)
    assert info["trade_pnl"] == pytest.approx(expected_net, rel=1e-6)
    assert info["trade_penalty"] >= expected_tax
    assert info["equity"] == pytest.approx(cfg.init_equity + expected_net, rel=1e-6)


def test_living_cost_penalty_reduces_equity_over_episode():
    length = 4
    living_cost = 40.0
    price_df, feat_df = _make_constant_env_inputs(length, price=100.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        living_cost_per_episode=living_cost,
        reward_scale_divisor=1.0,
        random_start=False,
        window_bars=length,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, info = env.step(1)  # sempre flat
        total_reward += reward
    expected_penalty = -living_cost
    assert total_reward == pytest.approx(expected_penalty, rel=1e-6)
    assert info["equity"] == pytest.approx(cfg.init_equity + expected_penalty, rel=1e-6)


def test_tier_bonus_increases_with_profit():
    price_df, feat_df = _make_constant_env_inputs(3, price=100.0)
    cfg = EnvConfig(
        init_equity=100.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        reward_scale_divisor=1.0,
        tier_bonus_step_pct=0.1,
        tier_bonus_max_pct=0.5,
        tier_bonus_cap_pnl_pct=0.3,
        random_start=False,
        window_bars=3,
    )
    # lucro de 20% -> tier 0.2 -> bônus = delta * 0.2
    price_df_bonus = pd.DataFrame(
        {
            "open": [100.0, 120.0, 120.0],
            "high": [100.0, 120.0, 120.0],
            "low": [100.0, 120.0, 120.0],
            "close": [100.0, 120.0, 120.0],
            "volume": np.ones(3),
        }
    )
    env = BTCMixtureEnv(price_df_bonus, feat_df, cfg)
    env.reset()
    env.step(2)
    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _ = env.step(1)
        total_reward += reward
    delta = (120.0 - 100.0) * cfg.position_size
    expected_bonus = delta * 0.2
    cap = delta * cfg.tier_bonus_cap_pnl_pct
    expected_bonus = min(expected_bonus, cap)
    assert total_reward >= expected_bonus - 1e-6


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


def test_trend_bonus_pct_awards_on_alignment():
    # Trend up, open long: deve ganhar bônus na entrada proporcional ao notional
    price_df, feat_df = _make_constant_env_inputs(5, price=200.0, trend_state=1.0, trend_strength=1.5)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.2,
        fee_pct=0.0,
        slippage_pct=0.0,
        trend_bonus_coef=0.0,
        trend_bonus_coef_pct=0.01,
        trend_bonus_entry_mult=1.0,
        trend_penalty_coef=0.0,
        trend_penalty_coef_pct=0.0,
        random_start=False,
        window_bars=5,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()
    env.step(2)
    env.step(2)  # garante duração >= 2 barras
    _, _, _, info = env.step(1)
    assert info["trade_closed"] is True
    notional = 200.0 * cfg.position_size
    expected_entry_bonus = notional * cfg.trend_bonus_coef_pct * abs(feat_df.loc[0, "htf_trend_strength"]) * cfg.trend_bonus_entry_mult
    assert info["trade_bonus"] == pytest.approx(expected_entry_bonus, rel=1e-6)


def test_idle_penalty_applies_when_flat_and_scales_with_factor():
    # Ambiente sem custos nem posição; apenas penalidade de ociosidade.
    length = 5
    price_df, feat_df = _make_constant_env_inputs(length, price=100.0)
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=0.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        idle_penalty_factor=0.01,
        reward_scale_divisor=1.0,
        random_start=False,
        window_bars=length,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    env.reset()

    expected_episode_length = length
    expected_penalty_per_step = (cfg.init_equity * cfg.idle_penalty_factor) / float(expected_episode_length)

    total_reward = 0.0
    steps_taken = 0
    done = False
    # Mantém sempre flat (ação 1) para acionar apenas a penalidade de ociosidade.
    while not done:
        _, reward, done, info = env.step(1)
        total_reward += reward
        steps_taken += 1
        assert info["position"] == 0
        assert reward == pytest.approx(-expected_penalty_per_step, rel=1e-6)

    # Deve ter percorrido toda a janela
    assert steps_taken == expected_episode_length
    assert total_reward == pytest.approx(-expected_penalty_per_step * expected_episode_length, rel=1e-6)


def test_trend_alignment_bonus_and_penalty_affect_equity():
    # Verifica que seguir a tendência HTF rende bônus e ir contra gera penalidade.
    length = 4
    price_df_up, feat_df_up = _make_constant_env_inputs(length, price=100.0, trend_state=1.0, trend_strength=1.5)
    price_df_down, feat_df_down = _make_constant_env_inputs(length, price=100.0, trend_state=-1.0, trend_strength=1.5)

    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=length,
        trend_bonus_coef=0.0,
        trend_bonus_coef_pct=0.01,
        trend_bonus_entry_mult=1.0,
        trend_penalty_coef=0.0,
        trend_penalty_coef_pct=0.01,
        trend_penalty_entry_mult=1.0,
        hold_bonus_alpha=0.0,
    )

    # Caso alinhado: long em tendência de alta
    env_aligned = BTCMixtureEnv(price_df_up, feat_df_up, cfg)
    env_aligned.reset()
    info_aligned = {}
    for action in (2, 2, 1):  # abre long, mantém, fecha
        _, _, _, info_aligned = env_aligned.step(action)
    aligned_equity = info_aligned["equity"]

    # Caso contra tendência: long em tendência de baixa
    env_contra = BTCMixtureEnv(price_df_down, feat_df_down, cfg)
    env_contra.reset()
    info_contra = {}
    for action in (2, 2, 1):
        _, _, _, info_contra = env_contra.step(action)
    contra_equity = info_contra["equity"]

    # Mesmo preço (sem PnL), diferença deve vir apenas de bônus/penalidade de tendência
    assert aligned_equity > cfg.init_equity
    assert contra_equity < cfg.init_equity
    assert aligned_equity > contra_equity


def test_risk_atr_scale_reduces_reward_magnitude():
    # Quando risk_atr_scale > 0, a recompensa em períodos voláteis deve ser menor
    # do que a mesma recompensa sem ajuste de risco.
    prices = [100.0, 102.0]
    price_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(prices)),
        }
    )
    feat_df = pd.DataFrame({"atr_14": np.full(len(prices), 1.0)})

    base_cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=len(prices),
        reward_scale_divisor=1.0,
        idle_penalty_factor=0.0,
        profit_trail_pct=0.0,
        risk_atr_scale=0.0,
    )
    env_base = BTCMixtureEnv(price_df, feat_df, base_cfg)
    env_base.reset()
    _, reward_base, _, _ = env_base.step(2)  # abre long e anda uma barra

    risk_cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=len(prices),
        reward_scale_divisor=1.0,
        idle_penalty_factor=0.0,
        profit_trail_pct=0.0,
        risk_atr_scale=10.0,
    )
    env_risk = BTCMixtureEnv(price_df, feat_df, risk_cfg)
    env_risk.reset()
    _, reward_risk, _, _ = env_risk.step(2)

    assert reward_base > 0.0
    assert reward_risk > 0.0
    # Ajuste de risco deve reduzir a magnitude da recompensa
    assert reward_risk < reward_base


def test_mtm_does_not_double_count_pnl_on_close():
    # Ambiente sem custos nem bônus, posição fixa; no modo "mtm" o PnL deve ser
    # contabilizado apenas via mark-to-market, e o fechamento não pode adicionar PnL de novo.
    prices = [100.0, 101.0, 102.0]
    price_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(prices)),
        }
    )
    feat_df = pd.DataFrame({"atr_14": np.ones(len(prices))})
    cfg = EnvConfig(
        init_equity=1000.0,
        position_size=1.0,
        fee_pct=0.0,
        slippage_pct=0.0,
        random_start=False,
        window_bars=len(prices),
        accounting_mode="mtm",
        hold_bonus_alpha=0.0,
        giveback_threshold_pct=0.0,
        giveback_penalty_pct=0.0,
        short_trade_penalty=0.0,
    )
    env = BTCMixtureEnv(price_df, feat_df, cfg)
    obs = env.reset()
    assert obs is not None

    total_reward = 0.0
    done = False

    # t=0: abre long
    _, reward, done, info = env.step(2)
    total_reward += reward
    assert not done
    assert info["position"] == 1

    # t=1: mantém long
    _, reward, done, info = env.step(2)
    total_reward += reward
    assert not done
    assert info["position"] == 1

    # t=2: fecha posição (vai para flat)
    _, reward, done, info = env.step(1)
    total_reward += reward
    # A posição deve estar fechada neste ponto
    assert info["trade_closed"] is True
    assert info["position"] == 0

    # PnL esperado: posição de 1 unidade, entrada em 100, saída em 102
    expected_pnl = (prices[2] - prices[0]) * cfg.position_size
    # No modo mtm, equity final - inicial deve coincidir com PnL
    assert info["equity"] == pytest.approx(cfg.init_equity + expected_pnl, rel=1e-6)
    # E a soma dos rewards também não pode exceder esse valor (sem PnL em dobro)
    assert total_reward == pytest.approx(expected_pnl, rel=1e-6)
