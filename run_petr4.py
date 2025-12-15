import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos do projeto
sys.path.append(os.getcwd())

from src.strategies.ema_only.backtest import backtest_ema_only

def run_petr4_test():
    print("--- Baixando dados PETR4.SA (Yahoo Finance) ---")
    
    # 1. Download de Dados (Diário para pegar longo prazo)
    ticker = "PETR4.SA"
    # Baixar com auto_adjust=True para considerar splits e dividendos (importante para B3)
    df = yf.download(ticker, start="2017-01-01", end="2025-12-15", progress=False, auto_adjust=True)
    
    if df.empty:
        print("Erro: Nenhum dado baixado.")
        return

    # 2. Adaptação do Dataframe para o formato interno
    # O yfinance retorna MultiIndex nas colunas em versões recentes, vamos ajustar
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Tenta pegar o nível do Ticker se existir, senão usa o nível 0
            if ticker in df.columns.get_level_values(1):
                 df.columns = df.columns.get_level_values(0)
            else:
                 df.columns = df.columns.get_level_values(0)
        except:
            df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    # Renomear para minúsculo como o sistema espera
    # Yfinance retorna Date, Open, High, Low, Close, Volume
    cols_map = {
        "Date": "Date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }
    df = df.rename(columns=cols_map)
    
    # Garantir datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Se tiver dados faltantes, preencher
    df = df.dropna()
    
    print(f"Dados carregados: {len(df)} candles (Diário)\n")

    # 3. Configuração OTIMIZADA (Pareto 2021)
    # Ajustei levemente as médias para o Gráfico Diário (são mais lentas que 4h)
    config = {
        "data": {
            "symbol": "PETR4",
            "timeframe": "1d", 
            "days": 3000
        },
        "strategy": {
            "signal_mode": "custom_cci_ma",
            
            # Médias (Ajustadas levemente para Daily)
            "custom_ma_fast": 9,
            "custom_ma_slow": 21,
            
            # Configuração "Vencedora"
            "custom_cci_period": 14,
            "custom_cci_level": 100,
            "custom_dist_atr_mult": 0.05,  # Entrada Antecipada
            "custom_target_factor": 2.0,   # Alvo Maior
            "custom_stop_factor": 0.9,
            
            "custom_atr_period": 14,
            "compounding_enabled": True,
            "compounding_pct": 0.95,
            
            "ref_filter_enabled": True,
            "ref_ema_period": 200,     # Média de 200 dias
            "ref_buffer_pct": 0.002,
            
            "lot_size": 100, # Lote padrão de ações (simbólico, pois compounding está ativo)
            "fee_pct": 0.0003, # Taxa B3 aprox
            "allow_short": True # Permitir venda a descoberto (aluguel)
        },
        "backtest": {
            "initial_capital": 10000.0 # R$ 10.000 inicial
        }
    }

    # 4. Criar dados de referência (Simulando o loader)
    # Como estamos no diário, a referência é o próprio diário
    df['ref_ema'] = df['close'].ewm(span=200).mean()
    
    # 5. Executar Backtest
    print("--- Executando Estratégia ---")
    results = backtest_ema_only(df, config)
    
    # 6. Gerar Relatório Ano a Ano
    equity_curve = pd.Series(results['equity'], index=df['Date'])
    
    # Resample anual para pegar o valor final de cada ano
    yearly = equity_curve.resample('YE').last()
    
    print("\n" + "="*50)
    print(f"RELATÓRIO DE EVOLUÇÃO ANUAL: {ticker}")
    print(f"Capital Inicial: R$ {config['backtest']['initial_capital']:.2f}")
    print("="*50)
    print(f"{ 'Ano':<6} | {'Saldo Final':<15} | {'Lucro/Prej (R$)':<15} | {'% Ano':<10}")
    print("-" * 55)

    previous_balance = config['backtest']['initial_capital']
    
    for date, balance in yearly.items():
        year = date.year
        pnl = balance - previous_balance
        pct = (balance / previous_balance) - 1
        
        print(f"{year:<6} | R$ {balance:<12.2f} | R$ {pnl:<12.2f} | {pct:>7.2%}")
        previous_balance = balance

    total_return = (results['metrics']['final_equity'] - config['backtest']['initial_capital']) / config['backtest']['initial_capital']
    print("-" * 55)
    print(f"SALDO ATUAL: R$ {results['metrics']['final_equity']:.2f}")
    print(f"RETORNO TOTAL: {total_return:.2%}")
    print(f"WIN RATE: {results['metrics']['win_rate']:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_petr4_test()
