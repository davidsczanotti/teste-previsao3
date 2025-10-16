import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna

from . import config, strategy
from ...utils.data_loader import load_data
from ...utils.report import save_active_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def objective(trial: optuna.Trial, price_data: pd.DataFrame, train_split_idx: int) -> float:
    """
    Função objetivo para a otimização com Optuna.
    """
    # Definir o espaço de busca dos hiperparâmetros
    params = {
        "rsi_slow_period": trial.suggest_int("rsi_slow_period", 100, 300),
        "rsi_medium_period": trial.suggest_int("rsi_medium_period", 20, 100),
        "rsi_fast_period": trial.suggest_int("rsi_fast_period", 5, 20),
        "rsi_pullback_level_long": trial.suggest_int("rsi_pullback_level_long", 30, 45),
        "rsi_pullback_level_short": trial.suggest_int("rsi_pullback_level_short", 55, 70),
        "adx_period": trial.suggest_int("adx_period", 10, 30),
        "adx_threshold": trial.suggest_float("adx_threshold", 15, 30),
        "atr_period": trial.suggest_int("atr_period", 10, 30),
        "min_atr_pct": trial.suggest_float("min_atr_pct", 0.01, 0.1),
        "rr_ratio": trial.suggest_float("rr_ratio", 1.0, 3.0),
        "stop_loss_multiplier": trial.suggest_float("stop_loss_multiplier", 1.0, 5.0),
    }

    # Dividir os dados em treino
    train_data = price_data.iloc[:train_split_idx]

    try:
        pf = strategy.run_backtest(
            price_data=train_data,
            **params,
            fee=config.FEE,
            initial_capital=config.INITIAL_CAPITAL,
            size=config.POSITION_SIZE_PCT,
        )

        stats = pf.stats()
        # Otimizar pelo Sortino Ratio, que foca no risco de queda
        sortino = stats["Sortino Ratio"]

        # Penalizar se houver poucos trades
        if stats["Total Trades"] < 10:
            return -1.0

        return sortino if not np.isnan(sortino) else -1.0

    except Exception as e:
        logging.warning(f"Trial falhou com erro: {e}")
        return -1.0  # Retorna um valor ruim se o backtest falhar


def main():
    parser = argparse.ArgumentParser(description="Otimização da estratégia Triple RSI.")
    parser.add_argument("--days", type=int, default=30, help="Número de dias de dados para usar.")
    parser.add_argument("--trials", type=int, default=100, help="Número de tentativas de otimização.")
    parser.add_argument("--train-pct", type=float, default=0.8, help="Percentual de dados para treino.")
    args = parser.parse_args()

    logging.info(f"Carregando dados para {config.SYMBOL}@{config.TIMEFRAME}...")
    df = load_data(symbol=config.SYMBOL, timeframe=config.TIMEFRAME, days=args.days)
    logging.info(f"Total de {len(df)} candles carregados.")

    # Dividir dados em treino e validação
    train_split_idx = int(len(df) * args.train_pct)
    train_df = df.iloc[:train_split_idx]
    validation_df = df.iloc[train_split_idx:]

    logging.info(f"Período de treino: {train_df['Date'].min()} a {train_df['Date'].max()}")
    logging.info(f"Período de validação: {validation_df['Date'].min()} a {validation_df['Date'].max()}")

    # Configurar e rodar a otimização
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, train_split_idx), n_trials=args.trials, n_jobs=-1)

    best_params = study.best_params
    logging.info(f"Otimização concluída! Melhor score (Sortino): {study.best_value:.2f}")
    logging.info(f"Melhores parâmetros: {json.dumps(best_params, indent=2)}")

    # Salvar a configuração ativa
    save_active_config("TRIPLE_RSI_DEEP", config.SYMBOL, config.TIMEFRAME, best_params)

    # --- Validação Fora da Amostra (Out-of-Sample) ---
    logging.info("--- Executando validação em dados não vistos (Out-of-Sample) ---")
    pf_validation = strategy.run_backtest(
        price_data=validation_df,
        **best_params,
        fee=config.FEE,
        initial_capital=config.INITIAL_CAPITAL,
        size=config.POSITION_SIZE_PCT,
    )

    stats = pf_validation.stats()
    logging.info("\n--- Resultados da Validação ---")
    logging.info(f"P&L Final: ${stats['Total Return [%]'] * config.INITIAL_CAPITAL / 100:.2f}")
    logging.info(f"Total de Trades: {stats['Total Trades']}")
    logging.info(f"Taxa de Acerto: {stats['Win Rate [%]']:.2f}%")
    logging.info(f"Profit Factor: {stats['Profit Factor']:.2f}")
    logging.info(f"Sortino Ratio: {stats['Sortino Ratio']:.2f}")
    logging.info(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")


if __name__ == "__main__":
    main()
