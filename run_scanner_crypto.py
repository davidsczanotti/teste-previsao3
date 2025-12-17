import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.getcwd())
from src.core.backtest import backtest_ema_only
from src.utils.data_loader import load_data

def run_scanner_crypto():
    # Top Altcoins (USDT Pairs - Binance)
    # Seleção com histórico razoável e volatilidade
    tickers = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
        "DOGEUSDT", "LTCUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
        "DOTUSDT", "UNIUSDT", "ATOMUSDT", "ETCUSDT", "BCHUSDT"
    ]

    print(f"--- Iniciando Scanner Crypto em {len(tickers)} ativos (SuperTrend AI) ---")
    
    results_summary = []

    # Configuração (SuperTrend AI)
    base_config = {
        "data": { "days": 2000, "timeframe": "1d" }, # Cripto 2000 dias (~5.5 anos) é bastante
        "strategy": {
            "signal_mode": "supertrend_ai",
            
            # Parâmetros SuperTrend AI (Padrão)
            "st_length": 10,
            "st_min_mult": 1,
            "st_max_mult": 5,
            "st_step": 0.5,
            "st_perf_alpha": 10,
            "st_from_cluster": "Best",
            
            # Gestão de Risco
            "use_all_equity": True,
            
            # Configuração Geral
            "lot_size": 1000, 
            "fee_pct": 0.001, # 0.1% Binance Spot
            "allow_short": False # Long Only por enquanto
        },
        "backtest": { "initial_capital": 1000.0 }
    }

    for ticker in tickers:
        try:
            print(f"> Processando {ticker}...", end=" ")
            
            try:
                df = load_data(ticker, "1d", days=2000)
            except Exception as e:
                print(f"Sem dados ({e}).")
                continue

            if df.empty:
                print("Sem dados.")
                continue

            config = base_config.copy()
            config['data']['symbol'] = ticker
            
            # Backtest
            res = backtest_ema_only(df, config)
            
            # Métricas
            initial_capital = float(config["backtest"]["initial_capital"])
            final_equity = res['metrics']['final_equity']
            total_return = (final_equity - initial_capital) / initial_capital
            
            # Buy & Hold (Cripto explode, então B&H costuma ser alto)
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            bh_return = (last_price - first_price) / first_price

            win_rate = res['metrics']['win_rate']
            trades_count = res['metrics']['total_trades']
            
            equity_curve = pd.Series(res['equity'])
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak
            max_dd = dd.min() if not dd.empty else 0.0

            results_summary.append({
                "Ticker": ticker,
                "Estratégia (%)": total_return * 100,
                "Buy & Hold (%)": bh_return * 100,
                "Diff (%)": (total_return - bh_return) * 100,
                "Win Rate (%)": win_rate * 100,
                "Trades": trades_count,
                "Max DD (%)": max_dd * 100
            })
            
            print(f"OK (Est: {total_return*100:.1f}% | B&H: {bh_return*100:.1f}%)")

        except Exception as e:
            print(f"Erro: {e}")

    # Gerar Relatório
    df_results = pd.DataFrame(results_summary)
    if not df_results.empty:
        df_results = df_results.sort_values("Estratégia (%)", ascending=False)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"reports/scanner_crypto_{timestamp}.csv"
        df_results.to_csv(filename, index=False, float_format="%.2f")

        print("\n" + "="*100)
        print(f"RANKING CRYPTO (SuperTrend AI) | Base: $ {int(base_config['backtest']['initial_capital']):,}".replace(',', '.'))
        print("="*100)
        print(df_results.to_string(index=False, float_format="%.1f"))
        print("="*100)
        print(f"\n📄 Relatório salvo em: {filename}")
    else:
        print("Nenhum resultado.")

if __name__ == "__main__":
    run_scanner_crypto()
