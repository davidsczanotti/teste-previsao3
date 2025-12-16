import sys
import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np

# Adiciona o diretório atual ao path para importar módulos locais
sys.path.append(os.getcwd())

# Importa a lógica do core
try:
    from src.core.indicators import add_indicators
    from src.core.signals import apply_signals
except ImportError:
    print("Erro: Não foi possível importar os módulos do núcleo (src/core).")
    print("Verifique se a estrutura de pastas está correta.")
    sys.exit(1)

def get_market_data(ticker: str, period="2y", interval="1d"):
    """Baixa dados recentes do Yahoo Finance."""
    print(f"\n[1/3] Baixando dados para {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}.")
    
    # Ajuste para yfinance novo (MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    df = df.reset_index()
    # Renomear colunas para o padrão do sistema
    cols_map = {
        "Date": "Date", "Datetime": "Date", 
        "Open": "open", "High": "high", 
        "Low": "low", "Close": "close", 
        "Volume": "volume"
    }
    df = df.rename(columns=cols_map)
    
    # Garantir que Date é datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def analyze_ticker(ticker: str):
    """Executa a análise Trend Surfer no ticker."""
    
    # 1. Obter Dados
    try:
        df = get_market_data(ticker)
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return

    # 2. Configurar Estratégia (Trend Surfer v4.1)
    config = {
        "strategy": {
            "signal_mode": "trend_surfer_v4",
            "ts_fast_period": 9,
            "ts_slow_period": 21,
            "ts_ema_macro_period": 200,
            "ts_cci_period": 14,
            "ts_cci_min": 0,
            "allow_short": False
        }
    }

    # 3. Calcular Indicadores e Sinais
    print("[2/3] Calculando indicadores TrendSurfer...")
    df = add_indicators(df, config)
    df = apply_signals(df, config)
    
    # 4. Analisar Último Candle (Hoje/Ontem)
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    # Dados Relevantes
    close = last_candle['close']
    date = last_candle['Date'].strftime('%Y-%m-%d')
    fast_ma = last_candle['ts_fast_ma']
    slow_ma = last_candle['ts_slow_ma']
    macro_ema = last_candle['ts_ema_macro']
    cci = last_candle['ts_cci']
    signal = last_candle['signal']
    
    # 5. Output Bonito
    print(f"\n{ '='*50}")
    print(f" RELATÓRIO TREND SURFER: {ticker.upper()}")
    print(f" Data Base: {date}")
    print(f"{ '='*50}")
    
    print(f"\n> PREÇO ATUAL: {close:.2f}")
    
    print("\n--- INDICADORES ---")
    print(f"• Tendência Curta (MA 9 vs 21): {'ALTA 🟢' if fast_ma > slow_ma else 'BAIXA 🔴'}")
    print(f"• Tendência Macro (Preço vs EMA200): {'ALTA 🟢' if close > macro_ema else 'BAIXA 🔴'}")
    print(f"• Momento (CCI > 0): {'POSITIVO 🟢' if cci > 0 else 'NEGATIVO 🔴'} ({cci:.2f})")
    
    print(f"\n--- VEREDITO ---")
    
    if signal == 1:
        print("🚀 SINAL DE COMPRA DETECTADO!")
        print("  O sistema indica entrada nesta barra.")
        print("  Lembre-se do Stop Loss Inicial (~5%) e Trail Stop (10%).")
        
    elif fast_ma > slow_ma and close > macro_ema and cci > 0:
        print("🌊 EM TENDÊNCIA (ALTA)")
        print("  Não há sinal de entrada HOJE (o cruzamento já ocorreu).")
        print("  Se você já está posicionado: MANTENHA (HOLD) 🛡️")
        print("  Monitore o Trailing Stop (10% do topo).")
        
    else:
        print("💤 AGUARDAR / FORA DO MERCADO")
        print("  As condições não estão alinhadas para compra.")
        if close < macro_ema:
            print("  Motivo principal: Preço abaixo da Média de 200 (Macro Baixa).")
        elif fast_ma < slow_ma:
            print("  Motivo principal: Médias cruzadas para baixo.")

    print(f"{ '='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="Trend Surfer CLI - Análise de Tendência")
    parser.add_argument("ticker", nargs="?", help="Ticker do ativo (ex: MGLU3.SA, BTC-USD)")
    
    args = parser.parse_args()
    
    if args.ticker:
        analyze_ticker(args.ticker)
    else:
        # Modo Interativo
        while True:
            try:
                user_input = input("\nDigite o Ticker para analisar (ou 'sair'): ").strip()
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break
                if not user_input:
                    continue
                
                analyze_ticker(user_input)
            except KeyboardInterrupt:
                print("\nSaindo...")
                break

if __name__ == "__main__":
    main()
