import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos do projeto
sys.path.append(os.getcwd())

from src.strategies.ema_only.backtest import backtest_ema_only

def run_mglu3_test():
    ticker = "MGLU3.SA"
    print(f"--- Baixando dados {ticker} (Yahoo Finance) ---")
    
    # 1. Download de Dados (Diário)
    # auto_adjust=True é crucial para MGLU3 que teve muitos splits/bonificações
    df = yf.download(ticker, start="2017-01-01", end="2025-12-15", progress=False, auto_adjust=True)
    
    if df.empty:
        print("Erro: Nenhum dado baixado.")
        return

    # 2. Adaptação do Dataframe
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if ticker in df.columns.get_level_values(1):
                 df.columns = df.columns.get_level_values(0)
            else:
                 df.columns = df.columns.get_level_values(0)
        except:
            df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    cols_map = {
        "Date": "Date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }
    df = df.rename(columns=cols_map)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()
    
    print(f"Dados carregados: {len(df)} candles (Diário)\n")

    # 3. Configuração OTIMIZADA (Mesma do BTC/PETR4/KLBN4)
    config = {
        "data": {
            "symbol": ticker,
            "timeframe": "1d", 
            "days": 3000
        },
        "strategy": {
            "signal_mode": "custom_cci_ma",
            "custom_ma_fast": 9,
            "custom_ma_slow": 21,
            "custom_cci_period": 14,
            "custom_cci_level": 100,
            "custom_dist_atr_mult": 0.05,
            "custom_target_factor": 2.0,
            "custom_stop_factor": 0.9,
            "custom_atr_period": 14,
            "compounding_enabled": True,
            "compounding_pct": 0.95,
            "ref_filter_enabled": True,
            "ref_ema_period": 200,
            "ref_buffer_pct": 0.002,
            "lot_size": 100,
            "fee_pct": 0.0003,
            "allow_short": True
        },
        "backtest": {
            "initial_capital": 10000.0
        }
    }

    # 4. Criar referência
    df['ref_ema'] = df['close'].ewm(span=200).mean()
    
    # 5. Backtest
    print("--- Executando Estratégia ---")
    results = backtest_ema_only(df, config)
    
    # 6. Relatório
    equity_curve = pd.Series(results['equity'], index=df['Date'])
    yearly = equity_curve.resample('YE').last()
    
    print("\n" + "="*50)
    print(f"RELATÓRIO DE EVOLUÇÃO ANUAL: {ticker}")
    print(f"Capital Inicial: R$ {config['backtest']['initial_capital']:.2f}")
    print("="*50)
    print(f"{'Ano':<6} | {'Saldo Final':<15} | {'Lucro/Prej (R$)':<15} | {'% Ano':<10}")
    print("-" + "="*55)

    previous_balance = config['backtest']['initial_capital']
    
    for date, balance in yearly.items():
        year = date.year
        pnl = balance - previous_balance
        pct = (balance / previous_balance) - 1
        
        print(f"{year:<6} | R$ {balance:<12.2f} | R$ {pnl:<12.2f} | {pct:>7.2%}")
        previous_balance = balance

    total_return = (results['metrics']['final_equity'] - config['backtest']['initial_capital']) / config['backtest']['initial_capital']
    print("-" + "="*55)
    print(f"SALDO ATUAL: R$ {results['metrics']['final_equity']:.2f}")
    print(f"RETORNO TOTAL: {total_return:.2%}")
    print(f"WIN RATE: {results['metrics']['win_rate']:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_mglu3_test()
