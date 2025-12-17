import optuna
import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Dict, Any

# Adiciona raiz ao path
sys.path.append(os.getcwd())

from src.core.backtest import backtest_ema_only
from src.utils.data_loader import load_data

# Configuração de Logging para limpar saída do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, df_target: pd.DataFrame):
    """
    Função Objetivo: O Optuna vai tentar MAXIMIZAR o retorno desta função.
    """
    
    # 1. Definir Espaço de Busca (Hiperparâmetros)
    fast_period = trial.suggest_int('ts_fast_period', 5, 50)
    slow_period = trial.suggest_int('ts_slow_period', 10, 100)
    
    # Garante que Lenta > Rápida (Constraint)
    if fast_period >= slow_period:
        raise optuna.TrialPruned()

    # Outros parâmetros
    macro_period = trial.suggest_categorical('ts_ema_macro_period', [100, 200, 300])
    trailing_pct = trial.suggest_float('trailing_stop_pct', 0.05, 0.35, step=0.01)
    
    # Configuração Dinâmica
    config = {
        "data": { "symbol": "TARGET", "timeframe": "1d" },
        "strategy": {
            "signal_mode": "ema_strategy_v5_2",
            
            # Parâmetros em Teste
            "ts_fast_period": fast_period,
            "ts_slow_period": slow_period,
            "ts_ema_macro_period": macro_period,
            "trailing_stop_pct": trailing_pct,
            
            # Fixos
            "ts_start_year": 2017,
            "use_all_equity": True,
            "risk_per_trade_pct": 0.02, # Irrelevante se use_all_equity=True
            "stop_loss_fixo_pct": 0.06, # Stop fixo inicial
            "exit_trigger": "cross_ma",
            
            "fee_pct": 0.0003,
            "allow_short": False
        },
        "backtest": { "initial_capital": 1000.0 }
    }

    # 2. Executar Backtest
    # O backtest precisa recalcular indicadores a cada tentativa, pois as médias mudam.
    # Como o cálculo é rápido (Pandas/Numpy), faremos inline.
    
    try:
        # Cópia para não sujar o cache global
        df_run = df_target.copy()
        
        # Injeção manual de indicadores (bypass em `add_indicators` para performance seria ideal, mas vamos usar o padrão)
        # O backtest_ema_only chama add_indicators internamente
        result = backtest_ema_only(df_run, config)
        
        # 3. Métrica de Sucesso
        # Podemos otimizar por: 'total_return_pct', 'sharpe_ratio', 'total_pnl'
        metric = result['metrics']['total_return_pct']
        
        # Penalidade para poucos trades (Overfitting em 1 ou 2 trades de sorte)
        if result['metrics']['total_trades'] < 10:
            return -1.0 # Penaliza fortemente
            
        return metric

    except Exception as e:
        return -1.0

def run_optimization(ticker: str):
    print(f"\n--- Otimizando {ticker} com Optuna ---")
    
    # Carregar Dados (apenas uma vez)
    print("Carregando dados...")
    try:
        # Tenta carregar range fixo
        from src.utils.data_loader import load_data_range
        df = load_data_range(ticker, "1d", "2017-01-01", "2025-12-15")
        
        # Ajuste Colunas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        cols_map = {"Date": "Date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df = df.rename(columns=cols_map)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna()
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # Criar Estudo
    study = optuna.create_study(direction='maximize')
    
    # Rodar Otimização (50 Tentativas)
    print("Executando 50 simulações (trials)... aguarde.")
    study.optimize(lambda trial: objective(trial, df), n_trials=50)

    # Resultados
    best = study.best_trial
    
    print(f"\nRESULTADO FINAL PARA {ticker}:")
    print(f"Melhor Retorno: {best.value * 100:.2f}%")
    print("Melhores Parâmetros:")
    for key, value in best.params.items():
        print(f"  - {key}: {value}")

    # Comparativo Base vs Otimizado
    # (Requer rodar uma vez com os parametros base para comparar visualmente, mas o usuário já sabe o base)

if __name__ == "__main__":
    # Vamos testar em dois perfis diferentes
    run_optimization("WEGE3.SA") # Tendência Clássica
    run_optimization("PETR4.SA") # Volátil / Dividendos (efeito no preço ajustado)
