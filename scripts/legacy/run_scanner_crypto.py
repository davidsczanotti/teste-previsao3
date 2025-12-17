import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from src.strategies.ema_only.backtest import backtest_ema_only

def run_scanner_crypto():
    # Lista de Criptomoedas (Top Market Cap & DeFi/L1s)
    tickers = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
        "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
        "LINK-USD", "DOT-USD", "MATIC-USD", "LTC-USD"
    ]

    print(f"--- Iniciando Crypto Scanner em {len(tickers)} ativos (2017-2025) ---")
    print("Estratégia: EMA Only (Modo Trend Surfer v4.1)\n")

    results_summary = []

    # Configuração Padrão (Trend Surfer v4.1)
    # Ajustamos 'days' para garantir cobertura máxima disponível
    base_config = {
        "data": { "days": 3000, "timeframe": "1d" },
        "strategy": {
            "signal_mode": "trend_surfer_v4",
            
            # Parâmetros Específicos Trend Surfer
            "ts_fast_period": 9, 
            "ts_slow_period": 21, 
            "ts_ema_macro_period": 200, 
            "ts_cci_period": 14, 
            "ts_cci_min": 0,
            
            # Gestão de Risco
            "risk_per_trade_pct": 0.02,
            "initial_stop_pct": 0.05,
            "trailing_stop_pct": 0.10,
            
            # Configuração Geral
            "compounding_enabled": False, 
            "ref_filter_enabled": False, 
            "lot_size": 1000, 
            "fee_pct": 0.001,  # Taxa um pouco maior pra cripto (0.1%)
            "allow_short": False
        },
        "backtest": { "initial_capital": 10000.0 }
    }

    for ticker in tickers:
        try:
            print(f"> Processando {ticker}...", end=" ")
            
            # Download
            df = yf.download(ticker, start="2017-01-01", end="2025-12-15", progress=False, auto_adjust=True)
            
            if df.empty:
                print("Sem dados.")
                continue

            # Adaptação de colunas (tratamento para MultiIndex do yfinance novo)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    # Tenta pegar nível 0 se ticker não estiver no nível 1
                    df.columns = df.columns.get_level_values(0)
                except:
                    pass
            
            df = df.reset_index()
            cols_map = {"Date": "Date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            df = df.rename(columns=cols_map)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.dropna()

            # Config do Ticker
            config = base_config.copy()
            config['data']['symbol'] = ticker
            
            # Cálculo Referência
            df['ref_ema'] = df['close'].ewm(span=200).mean()

            # Backtest
            res = backtest_ema_only(df, config)
            
            # Métricas
            final_equity = res['metrics']['final_equity']
            total_return = (final_equity - 10000) / 10000
            win_rate = res['metrics']['win_rate']
            trades_count = res['metrics']['total_trades']
            
            # Calcular Drawdown Máximo % do histórico
            equity_curve = pd.Series(res['equity'])
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak
            max_dd = dd.min()

            results_summary.append({
                "Ticker": ticker,
                "Retorno Total (%)": total_return * 100,
                "Capital Final": final_equity,
                "Win Rate (%)": win_rate * 100,
                "Trades": trades_count,
                "Max Drawdown (%)": max_dd * 100
            })
            
            print(f"OK ({total_return*100:.1f}%)")

        except Exception as e:
            print(f"Erro: {e}")

    # Gerar DataFrame e Ordenar
    df_results = pd.DataFrame(results_summary)
    if not df_results.empty:
        df_results = df_results.sort_values("Retorno Total (%)", ascending=False)
        print("\n" + "="*80)
        print(f"RANKING CRYPTO (2017-2025) | Base: $ 10.000")
        print("="*80)
        print(df_results.to_string(index=False, float_format="%.2f"))
        print("="*80)
    else:
        print("\nNenhum resultado gerado.")

if __name__ == "__main__":
    run_scanner_crypto()
