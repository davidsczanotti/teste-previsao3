import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from src.strategies.ema_only.backtest import backtest_ema_only

def run_scanner():
    # Lista diversificada de Tickers da B3
    tickers = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", # Blue Chips / Bancos
        "WEGE3.SA", "PRIO3.SA", # Crescimento / Qualidade
        "GGBR4.SA", "CSNA3.SA", "SUZB3.SA", # Commodities / Cíclicas
        "MGLU3.SA", "LREN3.SA", # Varejo (Alta Volatilidade)
        "ELET3.SA", "CMIG4.SA", # Elétricas
        "B3SA3.SA", "RENT3.SA"  # Financeiro / Locadoras
    ]

    print(f"--- Iniciando Scanner em {len(tickers)} ativos (2017-2025) ---")
    print("Estratégia: EMA Only (Modo Trend Surfer v4.1)\n")

    results_summary = []

    # Configuração Padrão (Trend Surfer v4.1)
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
            "compounding_enabled": False, # Usamos RiskPct agora
            "ref_filter_enabled": False, # Integrado na lógica (Close > Macro)
            "lot_size": 1000, # Fallback
            "fee_pct": 0.0003, 
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

            # Adaptação de colunas
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    if ticker in df.columns.get_level_values(1):
                         df.columns = df.columns.get_level_values(0)
                    else:
                         df.columns = df.columns.get_level_values(0)
                except:
                    df.columns = df.columns.get_level_values(0)
            
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
            initial_capital = float(config.get("backtest", {}).get("initial_capital", 10000.0))
            final_equity = res['metrics']['final_equity']
            total_return = (final_equity - initial_capital) / initial_capital
            win_rate = res['metrics']['win_rate']
            trades_count = res['metrics']['total_trades']
            
            # Calcular Drawdown Máximo % do histórico
            equity_curve = pd.Series(res['equity'])
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak
            max_dd = dd.min()

            # --- Novas Métricas Solicitadas ---
            trades = res['trades']
            avg_days = 0
            avg_pnl = 0
            
            if trades_count > 0:
                # 1. Tempo Médio
                # Garante que as datas sejam datetime para subtração
                durations = [(pd.to_datetime(t['exit_time']) - pd.to_datetime(t['entry_time'])).days for t in trades]
                avg_days = sum(durations) / len(durations)
                
                # 2. Lucro Médio por Trade
                avg_pnl = res['metrics']['total_pnl'] / trades_count

            results_summary.append({
                "Ticker": ticker,
                "Retorno Total (%)": total_return * 100,
                "Capital Final": final_equity,
                "Win Rate (%)": win_rate * 100,
                "Trades": trades_count,
                "Tempo Médio (Dias)": avg_days,
                "Lucro Médio (R$)": avg_pnl,
                "Max Drawdown (%)": max_dd * 100
            })
            
            print(f"OK ({total_return*100:.1f}%)")

        except Exception as e:
            print(f"Erro: {e}")

    # Gerar DataFrame e Ordenar
    df_results = pd.DataFrame(results_summary)
    df_results = df_results.sort_values("Retorno Total (%)", ascending=False)

    print("\n" + "="*80)
    print(f"RANKING DE ATIVOS (2017-2025) | Base: R$ {int(base_config['backtest']['initial_capital']):,}".replace(',', '.'))
    print("="*80)
    print(df_results.to_string(index=False, float_format="%.2f"))
    print("="*80)

if __name__ == "__main__":
    run_scanner()
