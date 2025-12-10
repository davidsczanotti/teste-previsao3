#!/usr/bin/env python3
"""
Otimização de parâmetros para EMA-only usando Optuna
"""

import json
import sys
from pathlib import Path
import optuna
import pandas as pd

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .backtest import backtest_ema_only, load_data_with_ref

def objective(trial, df, config):
    """Função objetivo para Optuna."""
    # Sugerir parâmetros
    config['strategy']['ema_fast_period'] = trial.suggest_int('ema_fast_period', 5, 21)
    config['strategy']['ema_mid_period'] = trial.suggest_int('ema_mid_period', 10, 55)
    config['strategy']['ema_slow_period'] = trial.suggest_int('ema_slow_period', 55, 200)
    config['strategy']['sma_fast_period'] = trial.suggest_int('sma_fast_period', 5, 21)
    config['strategy']['sma_mid_period'] = trial.suggest_int('sma_mid_period', 10, 55)
    config['strategy']['sma_slow_period'] = trial.suggest_int('sma_slow_period', 55, 200)
    config['strategy']['lot_size'] = trial.suggest_float('lot_size', 0.0005, 0.005)
    config['strategy']['percent_trailing_pct'] = trial.suggest_float('percent_trailing_pct', 0.01, 0.05)

    # Backtest
    result = backtest_ema_only(df, config)

    # Objetivo: maximizar Sharpe ratio diário
    returns = pd.Series(result['equity']).pct_change().dropna()
    if len(returns) > 1:
        sharpe = returns.mean() / returns.std() * (365 ** 0.5) if returns.std() > 0 else 0
    else:
        sharpe = 0

    return sharpe

def run_optimization(config_path: str = 'src/strategies/ema_only/config.json'):
    """Executa otimização."""
    with open(config_path) as f:
        config = json.load(f)

    # Carregar dados
    df = load_data_with_ref(config)

    # Otimização
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, config.copy()), n_trials=config['optimize']['trials'])

    # Melhor resultado
    best_params = study.best_params
    best_value = study.best_value

    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor Sharpe mensal: {best_value}")

    # Salvar
    outdir = Path(config['optimize']['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / "optimization_results.json"
    with open(output_file, 'w') as f:
        json.dump({'best_params': best_params, 'best_value': best_value}, f, indent=2)

    print(f"Resultados salvos em {output_file}")

def main():
    config_path = Path(__file__).parent / "config.json"
    run_optimization(str(config_path))

if __name__ == "__main__":
    main()
