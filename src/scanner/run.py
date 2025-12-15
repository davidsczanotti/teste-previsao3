import sys
import os
import argparse
# Hack para importar módulos do projeto
sys.path.append(os.getcwd())

from src.scanner.fetchers import get_b3_tickers, get_crypto_tickers
from src.scanner.core import run_scanner_loop

def main():
    parser = argparse.ArgumentParser(description="Scanner de Mercado - EMA Only Strategy")
    parser.add_argument("--type", choices=["b3", "crypto", "all"], default="b3", help="Tipo de ativo")
    parser.add_argument("--limit", type=int, default=0, help="Limitar número de ativos (0 = todos)")
    parser.add_argument("--out", type=str, default="scanner_results.csv", help="Arquivo de saída")
    
    args = parser.parse_args()
    
    dfs = []
    
    if args.type in ["b3", "all"]:
        tickers = get_b3_tickers()
        df_b3 = run_scanner_loop(tickers, "B3", args.limit)
        df_b3["Type"] = "B3"
        dfs.append(df_b3)
        
    if args.type in ["crypto", "all"]:
        tickers = get_crypto_tickers()
        # Filtro de fallback se a API falhar ou retornar vazio
        if not tickers:
            tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT"]
        
        df_crypto = run_scanner_loop(tickers, "CRYPTO", args.limit)
        df_crypto["Type"] = "CRYPTO"
        dfs.append(df_crypto)

    if dfs:
        import pandas as pd
        final_df = pd.concat(dfs).sort_values("Retorno Total (%)", ascending=False)
        
        print("\n" + "="*100)
        print(f"🏆 TOP 20 ATIVOS - RESULTADO SCANNER")
        print("="*100)
        print(final_df.head(20).to_string(index=False, float_format="%.2f"))
        print("="*100)
        
        final_df.to_csv(args.out, index=False)
        print(f"\n💾 Resultados completos salvos em: {args.out}")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()
