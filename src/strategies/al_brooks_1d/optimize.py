import numpy as np
import optuna

from .backtest import backtest_al_brooks_inside_bar, plot_backtest
from .config import AlBrooksConfig, save_active_config
from ...utils.metrics import calculate_metrics
from ...utils.optimizer import run_optimization_cli


def make_objective(df_train, lot_size: float, min_trade_threshold: int = 10):
    """Creates the objective function for Optuna."""
    threshold = max(1, min_trade_threshold)

    def objective(trial: optuna.Trial) -> float:
        # Definir o espaço de busca para os parâmetros
        ema_fast = trial.suggest_int("ema_fast_period", 5, 20)
        ema_medium = trial.suggest_int("ema_medium_period", ema_fast + 3, ema_fast + 25)
        ema_slow = trial.suggest_int("ema_slow_period", ema_medium + 5, ema_medium + 60)

        risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.2, 4.0, step=0.1)
        max_avg_deviation_pct = trial.suggest_float("max_avg_deviation_pct", 0.1, 3.0, step=0.05)
        adx_threshold = trial.suggest_float("adx_threshold", 18.0, 35.0, step=1.0)
        atr_stop_multiplier = trial.suggest_float("atr_stop_multiplier", 1.0, 3.0, step=0.1)
        atr_trail_multiplier = trial.suggest_float("atr_trail_multiplier", 0.0, 3.0, step=0.1)
        htf_lookback = trial.suggest_int("htf_lookback", 10, 40)
        min_atr = trial.suggest_float("min_atr", 0.0, 50.0, step=0.5)

        # Roda o backtest com os parâmetros sugeridos
        try:
            trades, total_pnl, _ = backtest_al_brooks_inside_bar(
                df_train.copy(),
                ema_fast_period=ema_fast,
                ema_medium_period=ema_medium,
                ema_slow_period=ema_slow,
                risk_reward_ratio=risk_reward_ratio,
                max_avg_deviation_pct=max_avg_deviation_pct,
                lot_size=lot_size,
                adx_threshold=adx_threshold,
                atr_stop_multiplier=atr_stop_multiplier,
                atr_trail_multiplier=atr_trail_multiplier,
                htf_lookback=htf_lookback,
                min_atr=min_atr,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e9  # Penaliza configurações que causam erro

        metrics = calculate_metrics(trades)
        trade_count = metrics["total_trades"]
        total_pnl = metrics["total_pnl"]
        profit_factor = metrics["profit_factor"]

        if trade_count == 0:
            return -1.0

        if not np.isfinite(profit_factor):
            profit_factor = 10.0

        trade_factor = min(1.0, trade_count / threshold)

        if total_pnl <= 0:
            return total_pnl * trade_factor

        score = (profit_factor * trade_factor) + (total_pnl / 200.0)
        return score

    return objective


def main():
    """Main function to run the CLI for optimization."""
    run_optimization_cli(
        strategy_name="ALBROOKS",
        default_symbol="BTCUSDT",
        default_timeframe="1d",
        objective_func_creator=make_objective,
        backtest_func=backtest_al_brooks_inside_bar,
        plot_func=plot_backtest,
        config_model=AlBrooksConfig,
        save_config_func=save_active_config,
    )


if __name__ == "__main__":
    main()
