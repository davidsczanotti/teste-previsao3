"""Compatibility wrapper for legacy imports.

The implementation lives in `src.core.backtest`.
"""

from __future__ import annotations

from src.core.backtest import backtest_ema_only, run_backtest

__all__ = ["backtest_ema_only", "run_backtest"]

