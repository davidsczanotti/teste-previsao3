import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from src.strategies.ema_only.backtest import backtest_ema_only
from src.scanner.data_source import download_b3_data, download_crypto_data

def run_strategy_on_ticker(df: pd.DataFrame, ticker: str, is_crypto: bool = False) -> Dict:
    """Roda a estratégia em um único ativo."""
    if df.empty or len(df) < 200:
        return None

    # Configuração "Pareto Optimized"
    config = {
        "data": { "symbol": ticker, "timeframe": "4h" if is_crypto else "1d" },
        "strategy": {
            "signal_mode": "custom_cci_ma",
            "custom_ma_fast": 9, "custom_ma_slow": 21,
            "custom_cci_period": 14, "custom_cci_level": 100,
            "custom_dist_atr_mult": 0.05,
            "custom_target_factor": 2.0, "custom_stop_factor": 0.9,
            "custom_atr_period": 14,
            "compounding_enabled": True, "compounding_pct": 0.95,
            "ref_filter_enabled": True, "ref_ema_period": 200, "ref_buffer_pct": 0.002,
            "lot_size": 100 if not is_crypto else 0.001, # Simbólico
            "fee_pct": 0.0004 if is_crypto else 0.0003,
            "allow_short": True
        },
        "backtest": { "initial_capital": 10000.0 }
    }

    # Gera referência
    df = df.copy()
    df['ref_ema'] = df['close'].ewm(span=200).mean()

    # Backtest
    try:
        res = backtest_ema_only(df, config)
        metrics = res['metrics']
        
        # Calcular Max Drawdown
        equity_curve = pd.Series(res['equity'])
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        max_dd = dd.min()

        return {
            "Ticker": ticker,
            "Retorno Total (%)": metrics['total_return_pct'] * 100,
            "Win Rate (%)": metrics['win_rate'] * 100,
            "Trades": metrics['total_trades'],
            "Profit Factor": metrics['profit_factor'],
            "Max Drawdown (%)": max_dd * 100,
            "Final Equity": metrics['final_equity']
        }
    except Exception as e:
        # print(f"Erro backtest {ticker}: {e}")
        return None

def run_scanner_loop(tickers: List[str], asset_type: str = "B3", limit: int = 0) -> pd.DataFrame:
    """Loop principal do scanner."""
    results = []
    
    # Se limit > 0, pega apenas os primeiros X tickers (para teste rápido)
    target_list = tickers[:limit] if limit > 0 else tickers
    
    print(f"\n🚀 Iniciando Scan em {len(target_list)} ativos ({asset_type})...")
    
    for ticker in tqdm(target_list, desc="Processando"):
        if asset_type == "B3":
            df = download_b3_data(ticker)
        else:
            df = download_crypto_data(ticker)
            
        res = run_strategy_on_ticker(df, ticker, is_crypto=(asset_type=="CRYPTO"))
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values("Retorno Total (%)", ascending=False)
    return df_res
