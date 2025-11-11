import pytest

from src.strategies.exper_corr_pos.env import EnvConfig
from src.strategies.exper_corr_pos.scripts.audit_policy import (
    _expected_bonus,
    _expected_penalty_components,
)


def test_expected_penalty_components_pct_breakdown():
    cfg = EnvConfig(
        turnover_penalty=0.0,
        turnover_penalty_pct=0.02,
        flip_exit_penalty=0.0,
        flip_exit_penalty_pct=0.03,
        trend_penalty_coef=0.0,
        trend_penalty_coef_pct=0.01,
        trend_penalty_entry_mult=1.5,
    )
    comps = _expected_penalty_components(
        env_cfg=cfg,
        size=0.5,
        entry_price=100.0,
        side=-1,
        reason="flip",
        trend_state=1.0,
        trend_strength=2.0,
    )
    assert comps["turnover"] == pytest.approx(1.0)
    assert comps["trend"] == pytest.approx(1.5)
    assert comps["flip"] == pytest.approx(1.5)
    assert sum(comps.values()) == pytest.approx(4.0)


def test_expected_penalty_components_fallbacks():
    cfg = EnvConfig(
        turnover_penalty=0.8,
        turnover_penalty_pct=0.0,
        flip_exit_penalty=0.4,
        flip_exit_penalty_pct=0.0,
        trend_penalty_coef=0.25,
        trend_penalty_coef_pct=0.0,
        trend_penalty_entry_mult=0.5,
    )
    comps = _expected_penalty_components(
        env_cfg=cfg,
        size=1.0,
        entry_price=50.0,
        side=1,
        reason="close",
        trend_state=-1.0,
        trend_strength=0.0,
    )
    # strength zero -> defaults to 1.0 inside helper
    assert comps["turnover"] == pytest.approx(cfg.turnover_penalty)
    assert comps["trend"] == pytest.approx(cfg.trend_penalty_coef * max(1.0, cfg.trend_penalty_entry_mult))
    assert comps["flip"] == 0.0


def test_expected_bonus_positive_only():
    cfg = EnvConfig(hold_bonus_alpha=0.1, hold_bonus_positive_only=True)
    bonus_win = _expected_bonus(cfg, pnl_gross=20.0, duration_bars=3)
    bonus_loss = _expected_bonus(cfg, pnl_gross=-5.0, duration_bars=3)
    assert bonus_win == pytest.approx(0.1 * 3 * 20.0)
    assert bonus_loss == 0.0
