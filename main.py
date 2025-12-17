import sys
import os
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Adiciona o diretório atual ao path para importar módulos locais
sys.path.append(os.getcwd())

# Importa a lógica do core
try:
    from src.core.indicators import add_indicators
    from src.core.signals import apply_signals
    from src.core.backtest import backtest_ema_only
except ImportError:
    print("Erro: Não foi possível importar os módulos do núcleo (src/core).")
    print("Verifique se a estrutura de pastas está correta.")
    sys.exit(1)

def get_market_data(ticker: str, period="5y", interval="1d"): # Periodo maior para gerar histórico de trades
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

    # 2. Configurar Estratégia (SuperTrend AI)
    config = {
        "strategy": {
            "signal_mode": "supertrend_ai",
            "st_length": 10,
            "st_min_mult": 1,
            "st_max_mult": 5,
            "st_step": 0.5,
            "st_perf_alpha": 10,
            "st_from_cluster": "Best",
            "allow_short": False
        },
        "backtest": {
            "initial_capital": 1000
        },
        "data": { # Mock para o backtester não reclamar
            "symbol": ticker,
            "timeframe": "1d"
        }
    }

    # 3. Calcular Indicadores e Sinais (Para Análise Atual)
    print("[2/3] Processando Inteligência Artificial...")
    df_calc = df.copy()
    df_calc = add_indicators(df_calc, config)
    df_calc = apply_signals(df_calc, config)
    
    # 4. Executar Simulação (Para Histórico de Trades)
    # O backtest recalcula internamente, mas é rápido.
    res = backtest_ema_only(df_calc, config)
    trades = res['trades']
    
    # --- ANÁLISE DO MOMENTO ATUAL ---
    last = df_calc.iloc[-1]
    
    # Dados Relevantes
    close = last['close']
    date = last['Date'].strftime('%d/%m/%Y')
    
    # SuperTrend AI Data
    st_val = last.get('supertrend_ai', 0.0)
    st_trend = int(last.get('supertrend_ai_trend', 0)) # 1=Bull, 0=Bear
    
    signal = last.get('signal', 0)
    exit_signal = last.get('exit_signal', 0)
    
    # 5. Output Bonito
    print(f"\n{ '='*60}")
    print(f" 🤖 RELATÓRIO SUPERTREND AI: {ticker.upper()}")
    print(f" 📅 Data Base: {date}")
    print(f"{ '='*60}")
    
    # Bloco Preço
    trend_emoji = "🟢 ALTA" if st_trend == 1 else "🔴 BAIXA"
    stop_dist = ((close - st_val) / close) * 100
    
    print(f"\n📊 STATUS ATUAL")
    print(f"   Preço:      R$ {close:.2f}")
    print(f"   Tendência:  {trend_emoji}")
    print(f"   Stop Loss:  R$ {st_val:.2f} ({abs(stop_dist):.1f}% de distância)")
    
    # Bloco Histórico
    print(f"\n📜 ÚLTIMOS 5 TRADES (Simulação)")
    print(f"   {'DATA ENT.':<12} | {'DATA SAÍDA':<12} | {'RESULTADO':<10}")
    print(f"   {'-'*12} + {'-'*12} + {'-'*10}")
    
    if not trades:
        print("   (Nenhum trade encerrado no período)")
    else:
        last_trades = trades[-5:]
        for t in reversed(last_trades): # Mais recente primeiro
            d_in = pd.to_datetime(t['entry_time']).strftime('%d/%m/%y')
            d_out = pd.to_datetime(t['exit_time']).strftime('%d/%m/%y')
            pnl = (t['exit'] - t['entry']) / t['entry'] * 100
            
            pnl_str = f"{pnl:+.1f}%"
            # Hack de cor ANSI simples
            color_code = "\033[92m" if pnl > 0 else "\033[91m" # Verde / Vermelho
            reset_code = "\033[0m"
            
            print(f"   {d_in:<12} | {d_out:<12} | {color_code}{pnl_str:<10}{reset_code}")

    # Bloco Veredito
    print(f"\n📢 VEREDITO DA IA")
    
    if signal == 1:
        print("   🚀 COMPRAR AGORA! (Sinal Confirmado)")
        print(f"      A tendência virou para ALTA.")
        print(f"      Entrada na abertura de amanhã.")
        
    elif exit_signal == 1:
        print("   ⚠️ VENDER / SAIR! (Sinal Confirmado)")
        print(f"      A tendência virou para BAIXA.")
        print(f"      Feche a posição imediatamente.")

    elif st_trend == 1:
        print("   💎 MANTER (HOLD)")
        print(f"      Você está surfando a tendência.")
        print(f"      Só saia se fechar abaixo de R$ {st_val:.2f}.")
            
    else:
        print("   💤 AGUARDAR (WAIT)")
        print(f"      Tendência de Baixa.")
        print(f"      Espere o preço romper R$ {st_val:.2f} para pensar em compra.")

    print(f"{ '='*60}\n")

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