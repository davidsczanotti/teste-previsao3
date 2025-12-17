import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.getcwd())
from src.core.backtest import backtest_ema_only

def run_scanner():
    # Lista Completa IDIV (Índice de Dividendos) - Teórica/Aproximada
    tickers = [
        'ABCB4.SA', 'AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BEEF3.SA', 'B3SA3.SA', 'BRAP4.SA', 'BRSR6.SA',
        'CXSE3.SA', 'CMIG3.SA', 'CMIG4.SA', 'CPLE3.SA', 'CPLE6.SA', 'CPFE3.SA', 'CSMG3.SA', 'CSNA3.SA',
        'CURY3.SA', 'DIRR3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'ENAT3.SA', 'ENBR3.SA', 'GGBR4.SA',
        'GOAU4.SA', 'ITSA4.SA', 'JBSS3.SA', 'JHSF3.SA', 'KEPL3.SA', 'KLBN11.SA', 'LEVE3.SA', 'MRFG3.SA',
        'PETR3.SA', 'PETR4.SA', 'PSSA3.SA', 'RANI3.SA', 'ROMI3.SA', 'SANB11.SA', 'SAPR11.SA', 'SLCE3.SA',
        'TAEE11.SA', 'TASA4.SA', 'TGMA3.SA', 'TRPL4.SA', 'UNIP6.SA', 'VALE3.SA', 'VIVT3.SA', 'WIZC3.SA'
    ]

    print(f"--- Iniciando Scanner em {len(tickers)} ativos do IDIV (2017-2025) ---")
    print("Estratégia: SuperTrend AI (Clustering Adaptativo)\n")

    results_summary = []

    # Configuração Padrão (SuperTrend AI)
    base_config = {
        "data": { "days": 3000, "timeframe": "1d" },
        "strategy": {
            "signal_mode": "supertrend_ai",
            
            # Parâmetros SuperTrend AI
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
            "fee_pct": 0.0003, 
            "allow_short": False
        },
        "backtest": { "initial_capital": 1000.0 }
    }

    for ticker in tickers:
        try:
            print(f"> Processando {ticker}...", end=" ")
            
            # Download (Agora usa o utils.data_loader idealmente, mas vamos manter inline para evitar refactor gigante agora)
            # Mas espera, yf.download direto aqui pode dar erro de compatibilidade se não tratarmos igual ao main.py
            # Vamos usar o data_loader que consertamos!
            
            from src.utils.data_loader import load_data
            try:
                # Carrega ultimos ~8 anos (3000 dias)
                df = load_data(ticker, "1d", days=3000)
            except Exception as e:
                print(f"Sem dados ({e}).")
                continue

            if df.empty:
                print("Sem dados (DF Vazio).")
                continue

            # Config do Ticker
            config = base_config.copy()
            config['data']['symbol'] = ticker
            
            # Cálculo Referência (Opcional, mas bom ter)
            df['ref_ema'] = df['close'].ewm(span=200).mean()

            # Backtest
            res = backtest_ema_only(df, config)
            
            # Métricas Estratégia
            initial_capital = float(config.get("backtest", {}).get("initial_capital", 10000.0))
            final_equity = res['metrics']['final_equity']
            total_return = (final_equity - initial_capital) / initial_capital
            
            # Métricas Buy & Hold
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            bh_return = (last_price - first_price) / first_price

            win_rate = res['metrics']['win_rate']
            trades_count = res['metrics']['total_trades']
            
            # Calcular Drawdown Máximo % do histórico
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

    # Gerar DataFrame e Ordenar
    df_results = pd.DataFrame(results_summary)
    if not df_results.empty:
        df_results = df_results.sort_values("Estratégia (%)", ascending=False)
        
        # Salvar Relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename_csv = f"reports/scanner_idiv_{timestamp}.csv"
        df_results.to_csv(filename_csv, index=False, float_format="%.2f")

        print("\n" + "="*100)
        print(f"COMPARATIVO: ESTRATÉGIA vs BUY & HOLD (2017-2025) | Base: R$ {int(base_config['backtest']['initial_capital']):,}".replace(',', '.'))
        print("="*100)
        print(df_results.to_string(index=False, float_format="%.1f"))
        print("="*100)
        print(f"\n📄 Relatório salvo em: {filename_csv}")
    else:
        print("\nNenhum resultado gerado.")

if __name__ == "__main__":
    run_scanner()

if __name__ == "__main__":
    run_scanner()
