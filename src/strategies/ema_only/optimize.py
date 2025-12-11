#!/usr/bin/env python3
"""
Otimização de parâmetros para EMA-only usando Optuna
"""

import json
import sys
from pathlib import Path
import optuna
import pandas as pd

# Adicionar src ao path (Project Root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.strategies.ema_only.backtest import backtest_ema_only, load_data_with_ref

def objective(trial, df, config):
    """Função objetivo para Optuna."""
    # Sugerir parâmetros baseados no search_space do config
    search_space = config['optimize']['search_space']
    
    for param_name, param_cfg in search_space.items():
        param_type = param_cfg['type']
        
        if param_type == 'int':
            config['strategy'][param_name] = trial.suggest_int(
                param_name, param_cfg['low'], param_cfg['high']
            )
        elif param_type == 'float':
            config['strategy'][param_name] = trial.suggest_float(
                param_name, param_cfg['low'], param_cfg['high']
            )
        elif param_type == 'categorical':
            config['strategy'][param_name] = trial.suggest_categorical(
                param_name, param_cfg['choices']
            )

    # Backtest
    result = backtest_ema_only(df, config)

    # Objetivo: maximizar Sharpe ratio diário (ou outro definido no config)
    # Aqui vamos usar o retorno total como proxy ou sharpe conforme config
    returns = pd.Series(result['equity']).pct_change().dropna()
    
    # Calcular Sharpe Anualizado (aprox)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * (365 ** 0.5)
    else:
        sharpe = -999.0
        
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
    print(f"Melhor Valor (Sharpe): {best_value}")

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
