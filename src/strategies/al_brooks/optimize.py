import argparse
from datetime import datetime, timedelta, UTC

import optuna
import pandas as pd

from ...binance_client import get_historical_klines
from .backtest import backtest_al_brooks_inside_bar
from .config import AlBrooksConfig, save_active_config


def load_data(ticker: str, interval: str, days: int) -> pd.DataFrame:
    start_dt = datetime.now(UTC) - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    df = get_historical_klines(ticker, interval, start_str)
    if df.empty:
        raise RuntimeError("Nenhum dado retornado da Binance.")
    return df.sort_values("Date").reset_index(drop=True)


def make_objective(df_train: pd.DataFrame, lot_size: float):
    def objective(trial: optuna.Trial) -> float:
        # Definir o espaço de busca para os parâmetros
        ema_fast = trial.suggest_int("ema_fast_period", 5, 15)
        ema_medium = trial.suggest_int("ema_medium_period", ema_fast + 5, 30)
        ema_slow = trial.suggest_int("ema_slow_period", ema_medium + 10, 60)

        risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.5, 3.5, step=0.1)
        max_avg_deviation_pct = trial.suggest_float("max_avg_deviation_pct", 0.1, 1.0, step=0.05)

        # Roda o backtest com os parâmetros sugeridos
        try:
            trades, pnl = backtest_al_brooks_inside_bar(
                df_train.copy(),
                ema_fast_period=ema_fast,
                ema_medium_period=ema_medium,
                ema_slow_period=ema_slow,
                risk_reward_ratio=risk_reward_ratio,
                max_avg_deviation_pct=max_avg_deviation_pct,
                lot_size=lot_size,
            )
        except Exception as e:
            trial.set_user_attr("error", str(e))
            return -1e9  # Penaliza configurações que causam erro

        # Métrica de otimização: Profit Factor
        # Queremos maximizar o Profit Factor, mas também garantir que seja lucrativo
        closed_trades = [t for t in trades if "pnl" in t]
        if not closed_trades:
            return 0.0  # Sem trades, sem pontuação

        total_profit = sum(t["pnl"] for t in closed_trades if t["pnl"] > 0)
        total_loss = abs(sum(t["pnl"] for t in closed_trades if t["pnl"] < 0))

        if total_loss == 0:
            profit_factor = 100.0  # Evita divisão por zero, valor alto para 100% de acerto
        else:
            profit_factor = total_profit / total_loss

        # Condição: Apenas considere o Profit Factor se o P&L for positivo
        if pnl <= 0:
            return pnl  # Retorna o P&L negativo para penalizar estratégias não lucrativas

        return profit_factor

    return objective


def main():
    parser = argparse.ArgumentParser(description="Otimizar a estratégia de Al Brooks com Optuna.")
    parser.add_argument("--ticker", default="BTCUSDT", help="Símbolo do ativo")
    parser.add_argument("--interval", default="15m", help="Intervalo das velas")
    parser.add_argument("--days", type=int, default=365, help="Dias de dados históricos")
    parser.add_argument("--lot_size", type=float, default=0.1, help="Tamanho do lote")
    parser.add_argument("--trials", type=int, default=200, help="Número de trials do Optuna")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    args = parser.parse_args()

    print(f"Carregando dados: {args.ticker} @ {args.interval} por {args.days} dias...")
    df = load_data(args.ticker, args.interval, args.days)

    # Otimização será feita no conjunto de dados inteiro por simplicidade,
    # mas uma divisão treino/validação seria ideal para um teste mais robusto.
    df_train = df

    print(f"Total de {len(df_train)} candles para otimização.")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # Queremos maximizar o Profit Factor
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(df_train, args.lot_size), n_trials=args.trials)

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

    # Validar os melhores parâmetros no mesmo conjunto de dados para ver o P&L
    print("\n--- Validando melhores parâmetros ---")
    trades, pnl = backtest_al_brooks_inside_bar(
        df_train.copy(),
        **study.best_params,
        lot_size=best_config.lot_size,
    )

    closed_trades = [t for t in trades if "pnl" in t]
    num_trades = len(closed_trades)
    wins = len([t for t in closed_trades if t["pnl"] > 0])
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0

    print(f"Resultado com parâmetros otimizados:")
    print(f"P&L Final: $ {pnl:.2f}")
    print(f"Total de Operações: {num_trades}")
    print(f"Taxa de Acerto: {win_rate:.2f}%")


if __name__ == "__main__":
    main()
