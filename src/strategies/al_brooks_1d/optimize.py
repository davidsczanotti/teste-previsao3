import argparse
from datetime import datetime, timedelta, UTC

import numpy as np
import optuna
import pandas as pd

from ...binance_client import get_historical_klines
from .backtest import backtest_al_brooks_inside_bar, plot_backtest
from .config import AlBrooksConfig, save_active_config
from ...utils.metrics import calculate_metrics


def load_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(ticker, interval, start_str)
    if df.empty:
        raise RuntimeError("Nenhum dado retornado da Binance.")
    return df.sort_values("Date").reset_index(drop=True)


MIN_TRADE_THRESHOLD = 20


def make_objective(df_train: pd.DataFrame, lot_size: float):
    def objective(trial: optuna.Trial) -> float:
        # Definir o espaço de busca para os parâmetros
        ema_fast = trial.suggest_int("ema_fast_period", 5, 20)
        ema_medium = trial.suggest_int("ema_medium_period", ema_fast + 3, ema_fast + 25)
        ema_slow = trial.suggest_int("ema_slow_period", ema_medium + 5, ema_medium + 60)

        risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.2, 3.0, step=0.1)
        max_avg_deviation_pct = trial.suggest_float("max_avg_deviation_pct", 0.1, 1.5, step=0.05)
        adx_threshold = trial.suggest_float("adx_threshold", 18.0, 35.0, step=1.0)
        atr_stop_multiplier = trial.suggest_float("atr_stop_multiplier", 1.0, 3.0, step=0.1)
        atr_trail_multiplier = trial.suggest_float("atr_trail_multiplier", 0.0, 3.0, step=0.1)
        htf_lookback = trial.suggest_int("htf_lookback", 10, 40)
        min_atr = trial.suggest_float("min_atr", 0.0, 50.0, step=0.5)

        # Roda o backtest com os parâmetros sugeridos
        try:
            trades, pnl, _ = backtest_al_brooks_inside_bar(
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

        # Métrica de otimização: Profit Factor
        # Queremos maximizar o Profit Factor, mas também garantir que seja lucrativo
        metrics = calculate_metrics(trades)
        trade_count = metrics["total_trades"]
        total_pnl = metrics["total_pnl"]
        profit_factor = metrics["profit_factor"]

        if trade_count == 0:
            return -1.0

        if not np.isfinite(profit_factor):
            profit_factor = 10.0

        trade_factor = min(1.0, trade_count / MIN_TRADE_THRESHOLD)

        if total_pnl <= 0:
            return total_pnl * trade_factor

        score = (profit_factor * trade_factor) + (total_pnl / 200.0)
        return score

    return objective


def print_summary(title: str, trades: list, pnl: float):
    """Imprime um resumo detalhado do backtest."""
    print(f"\n--- {title} ---")

    closed_trades = [t for t in trades if "pnl" in t]
    num_trades = len(closed_trades)

    if num_trades == 0:
        print("Nenhuma operação foi fechada no período.")
        print(f"P&L Final: $ {pnl:.2f}")
        return

    wins = [t for t in closed_trades if t["pnl"] > 0]
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] < 0))

    win_rate = (len(wins) / num_trades) * 100
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    print(
        f"P&L Final: $ {pnl:.2f} | Trades: {num_trades} | Win Rate: {win_rate:.2f}% | Profit Factor: {profit_factor:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Otimizar a estratégia de Al Brooks com Optuna.")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--interval", default="1d", help="Intervalo das velas")
    parser.add_argument("--days", type=int, default=365, help="Dias de dados históricos")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fração de dados para treino (ex: 0.8 para 80%)")
    parser.add_argument("--lot_size", type=float, default=0.1, help="Tamanho do lote")
    parser.add_argument("--trials", type=int, default=200, help="Número de trials do Optuna")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    args = parser.parse_args()

    print(f"Carregando dados: {args.ticker} @ {args.interval} por {args.days} dias...")
    df = load_data(args.ticker, args.interval, args.days)

    # Divide os dados em treino (in-sample) e validação (out-of-sample)
    n = len(df)
    split_idx = int(n * args.train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_valid = df.iloc[split_idx:].copy()

    print(f"Total de candles: {n} | Treino: {len(df_train)} | Validação: {len(df_valid)}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # Queremos maximizar o Profit Factor
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(df_train, args.lot_size), n_trials=args.trials, show_progress_bar=True, gc_after_trial=True
    )

    print("\n--- Otimização Concluída ---")
    print(f"Melhor valor (Profit Factor): {study.best_value:.2f}")
    print("Melhores parâmetros encontrados:")
    print(study.best_params)

    # Salvar a configuração ativa
    best_config = AlBrooksConfig(
        ticker=args.ticker,
        interval=args.interval,
        days=args.days,
        lot_size=args.lot_size,
        **study.best_params,
    )
    active_path = save_active_config(best_config)
    print(f"\nConfiguração ativa salva em: {active_path}")

    # Executa o backtest com os melhores parâmetros nos dados de TREINO
    trades_train, pnl_train, _ = backtest_al_brooks_inside_bar(
        df_train.copy(),
        **study.best_params,
        lot_size=best_config.lot_size,
    )
    print_summary("Resultado em Amostra (In-Sample / Treino)", trades_train, pnl_train)

    # Executa o backtest com os mesmos parâmetros nos dados de VALIDAÇÃO
    trades_valid, pnl_valid, df_valid_indicators = backtest_al_brooks_inside_bar(
        df_valid.copy(),
        **study.best_params,
        lot_size=best_config.lot_size,
    )
    print_summary("Resultado Fora da Amostra (Out-of-Sample / Validação)", trades_valid, pnl_valid)

    # Plotar o gráfico do período de validação
    if not df_valid.empty:
        print("\nGerando gráfico do período de validação...")
        plot_backtest(df_valid_indicators, trades_valid, f"{args.ticker}_validation")


if __name__ == "__main__":
    main()
