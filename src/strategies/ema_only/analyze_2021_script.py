import json
import pandas as pd
from pathlib import Path

def analyze_2021():
    # Load the backtest result
    path = Path("src/strategies/ema_only/reports/backtest/ema_only_BTCUSDT_4h.json")
    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    trades = data.get("trades", [])
    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Filter for 2021
    df_2021 = df[df['entry_time'].dt.year == 2021].copy()
    
    if df_2021.empty:
        print("No trades in 2021.")
        return

    print(f"Total Trades 2021: {len(df_2021)}")
    print(f"Total PnL 2021: {df_2021['pnl'].sum():.2f}")
    
    # Win Rate
    wins = df_2021[df_2021['pnl'] > 0]
    losses = df_2021[df_2021['pnl'] <= 0]
    win_rate = len(wins) / len(df_2021)
    print(f"Win Rate: {win_rate:.2%}")

    # Long vs Short
    longs = df_2021[df_2021['side'] == 'long']
    shorts = df_2021[df_2021['side'] == 'short']
    
    print("\n--- Long vs Short ---")
    print(f"Longs: {len(longs)} | PnL: {longs['pnl'].sum():.2f} | Win Rate: {len(longs[longs['pnl']>0])/len(longs):.2%}")
    print(f"Shorts: {len(shorts)} | PnL: {shorts['pnl'].sum():.2f} | Win Rate: {len(shorts[shorts['pnl']>0])/len(shorts):.2%}")

    # Monthly Breakdown
    df_2021['month'] = df_2021['entry_time'].dt.to_period('M')
    monthly = df_2021.groupby('month')['pnl'].sum()
    print("\n--- Monthly PnL ---")
    print(monthly)

    # Reasons
    print("\n--- Exit Reasons (Losses) ---")
    loss_reasons = losses['reason'].value_counts()
    print(loss_reasons)

    # Worst Run
    df_2021['pnl_cumsum'] = df_2021['pnl'].cumsum()
    print(f"\nMax Drawdown in 2021 PnL terms: {df_2021['pnl'].min()} (Single Trade) / {df_2021['pnl_cumsum'].min()} (Cumulative Low)")

    # Sample "Bad" Trades
    print("\n--- Top 5 Worst Trades ---")
    bad_trades = df_2021.sort_values('pnl').head(5)
    for _, row in bad_trades.iterrows():
        print(f"{row['side'].upper()} | Entry: {row['entry_time']} @ {row['entry']} | Exit: {row['exit_time']} @ {row['exit']} | PnL: {row['pnl']:.2f} | Reason: {row['reason']}")

if __name__ == "__main__":
    analyze_2021()
