import numpy as np
import optuna

from .backtest import backtest_al_brooks_book
from .config import AlBrooksBookConfig, save_active_config
from ...utils.metrics import calculate_metrics
from ...utils.optimizer import run_optimization_cli


def make_objective(df_train, lot_size: float, min_trade_threshold: int = 20):
    threshold = max(1, int(min_trade_threshold))
    FEE_PCT = 0.0004
    SLIPPAGE_PCT = 0.0005

    def objective(trial: optuna.Trial) -> float:
        ema_fast = trial.suggest_int("ema_fast_period", 15, 25)
        ema_med = trial.suggest_int("ema_medium_period", 40, 80)
        ema_slow = trial.suggest_int("ema_slow_period", 150, 250)
        slope_lookback = trial.suggest_int("slope_lookback", 3, 8)
        swing_lookback = trial.suggest_int("swing_lookback", 2, 5)
        bar_body_min_pct = trial.suggest_float("bar_body_min_pct", 50.0, 80.0, step=2.5)
        near_extreme_frac = trial.suggest_float("near_extreme_frac", 0.15, 0.35, step=0.05)
        atr_period = trial.suggest_int("atr_period", 10, 20)

        enable_inside_trend = trial.suggest_categorical("enable_inside_trend", [True, False])
        enable_h2_l2 = trial.suggest_categorical("enable_h2_l2", [True, False])
        enable_bo_pb = trial.suggest_categorical("enable_bo_pb", [True, False])
        bo_lookback = trial.suggest_int("bo_lookback", 10, 40)
        max_ema_distance_atr = trial.suggest_float("max_ema_distance_atr", 0.5, 1.5, step=0.1)
        use_trend_slope = trial.suggest_categorical("use_trend_slope", [True, False])
        min_ema_slope = trial.suggest_float("min_ema_slope", 0.0, 10.0, step=0.5)

        risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.0, 2.5, step=0.1)
        atr_stop_multiplier = trial.suggest_float("atr_stop_multiplier", 0.0, 2.5, step=0.25)
        atr_trail_multiplier = trial.suggest_float("atr_trail_multiplier", 0.0, 2.5, step=0.25)
        min_atr = trial.suggest_float("min_atr", 0.0, 50.0, step=0.5)

        try:
            trades, pnl, _ = backtest_al_brooks_book(
                df_train.copy(),
                ema_fast_period=ema_fast,
                ema_medium_period=ema_med,
                ema_slow_period=ema_slow,
                slope_lookback=slope_lookback,
                swing_lookback=swing_lookback,
                bar_body_min_pct=bar_body_min_pct,
                near_extreme_frac=near_extreme_frac,
                atr_period=atr_period,
                enable_inside_trend=enable_inside_trend,
                enable_h2_l2=enable_h2_l2,
                enable_bo_pb=enable_bo_pb,
                bo_lookback=bo_lookback,
                max_ema_distance_atr=max_ema_distance_atr,
                use_trend_slope=use_trend_slope,
                min_ema_slope=min_ema_slope,
                risk_reward_ratio=risk_reward_ratio,
                atr_stop_multiplier=atr_stop_multiplier,
                atr_trail_multiplier=atr_trail_multiplier,
                min_atr=min_atr,
                lot_size=lot_size,
                taker_fee_pct=FEE_PCT,
                slippage_pct=SLIPPAGE_PCT,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e9

        m = calculate_metrics(trades)
        trades_n = m.get("total_trades", 0)
        pf = m.get("profit_factor", 0.0)
        total_pnl = m.get("total_pnl", 0.0)

        if trades_n == 0:
            return -1.0
        if not np.isfinite(pf):
            pf = 10.0

        trade_factor = min(1.0, trades_n / float(threshold))
        if total_pnl <= 0:
            return float(total_pnl) * trade_factor
        return float(pf) * trade_factor + (float(total_pnl) / 200.0)

    return objective


def main():
    run_optimization_cli(
        strategy_name="ALBROOKS_BOOK",
        default_symbol="BTCUSDT",
        default_timeframe="1m",
        objective_func_creator=make_objective,
        backtest_func=backtest_al_brooks_book,
        plot_func=None,
        config_model=AlBrooksBookConfig,
        save_config_func=save_active_config,
    )


if __name__ == "__main__":
    main()

