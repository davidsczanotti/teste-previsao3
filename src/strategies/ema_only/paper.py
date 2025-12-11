import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.binance_client import get_historical_klines, get_current_price
from src.strategies.ema_only.backtest import calculate_mas, generate_signals

# Config Paths
STRATEGY_DIR = Path(__file__).parent
CONFIG_PATH = STRATEGY_DIR / "config.json"
WALLET_PATH = STRATEGY_DIR / "paper_wallet.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def load_wallet():
    if not WALLET_PATH.exists():
        # Initialize default wallet
        wallet = {
            "balance": 1000.0,
            "position": 0.0, # BTC amount (positive=long, negative=short)
            "entry_price": 0.0,
            "stop_price": 0.0,
            "target_price": 0.0,
            "trades_history": []
        }
        save_wallet(wallet)
        return wallet
    with open(WALLET_PATH) as f:
        return json.load(f)

def save_wallet(wallet):
    with open(WALLET_PATH, 'w') as f:
        json.dump(wallet, f, indent=2)

def run_paper_trading():
    config = load_config()
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
    print(f"--- Paper Trading Started: {symbol} {timeframe} ---")
    
    # Initialize wallet if needed
    load_wallet()

    while True:
        try:
            # 1. Fetch Data (Need enough history for EMA200)
            # Fetch last 300 candles (approx 50 days for 4h)
            # Using '90 days ago UTC' to be safe for EMA200 warmup
            df = get_historical_klines(symbol, timeframe, "90 days ago UTC")
            
            if df.empty:
                print("Error fetching data (empty). Retrying...")
                time.sleep(60)
                continue
                
            # 2. Calculate Indicators
            df = calculate_mas(df, config)
            df = generate_signals(df, config)
            
            # Get latest closed candle
            # Binance API usually returns the open candle as the last one (index -1)
            # So the last CLOSED candle is -2.
            last_closed_candle = df.iloc[-2]
            current_candle_metrics = df.iloc[-1] # Metrics for the open candle (incomplete but usable for ATR projection?)
            
            # Fetch real-time price for Exit execution
            current_price = get_current_price(symbol)
            
            wallet = load_wallet()
            position = wallet['position']
            
            # 3. Check Exits (Real-time on Current Price)
            if position != 0:
                pnl = 0
                exit_signal = False
                reason = ""
                
                # Check TP/SL
                if position > 0: # Long
                    if current_price <= wallet['stop_price']:
                        exit_signal = True
                        reason = "Stop Loss"
                    elif current_price >= wallet['target_price']:
                        exit_signal = True
                        reason = "Take Profit"
                    elif last_closed_candle['signal'] == -1:
                        exit_signal = True
                        reason = "Signal Reverse"
                        
                elif position < 0: # Short
                    if current_price >= wallet['stop_price']:
                        exit_signal = True
                        reason = "Stop Loss"
                    elif current_price <= wallet['target_price']:
                        exit_signal = True
                        reason = "Take Profit"
                    elif last_closed_candle['signal'] == 1:
                        exit_signal = True
                        reason = "Signal Reverse"
                
                if exit_signal:
                    # Execute Exit
                    if position > 0:
                        pnl = (current_price - wallet['entry_price']) * position
                    else:
                        pnl = (wallet['entry_price'] - current_price) * abs(position)
                    
                    wallet['balance'] += pnl
                    wallet['trades_history'].append({
                        "date": datetime.now().isoformat(),
                        "type": "EXIT",
                        "price": current_price,
                        "pnl": pnl,
                        "reason": reason,
                        "balance_after": wallet['balance']
                    })
                    wallet['position'] = 0.0
                    wallet['entry_price'] = 0.0
                    wallet['stop_price'] = 0.0
                    wallet['target_price'] = 0.0
                    save_wallet(wallet)
                    print(f"!!! EXIT EXECUTED: {reason} | PnL: {pnl:.2f} | New Balance: {wallet['balance']:.2f}")

            # 4. Check Entries (On Closed Candle Signal)
            # Only enter if no position (Wait for next candle if just exited? No, can reverse immediately)
            # We re-read wallet in case we just exited
            if wallet['position'] == 0:
                signal = last_closed_candle['signal']
                
                # To prevent re-entering the SAME signal bar repeatedly, we should track last_entry_time
                # For simplicity here: we assume if we just exited on Signal Reverse, we might re-enter?
                # Actually, backtest logic allows immediate flip.
                # BUT, if we enter now, we must ensure we don't enter 100 times on the same bar.
                # Ideally check if 'last_closed_candle index' > 'last trade time'.
                # For this simple script, we assume signal lasts for the duration of the bar, but we only act once.
                # Let's add a 'last_action_time' or check if the signal is FRESH (cross logic in generate_signals handles the pulse).
                # generate_signals uses (cross == 1) & (prev == -1), so 'signal' is 1 only for ONE bar.
                # So it is safe, as long as we haven't acted on THIS specific bar yet.
                # However, since we run this loop every minute, and the bar is 4H, 'last_closed_candle' remains the same for 4 hours.
                # We need to store "last_processed_bar_timestamp" in wallet to avoid 240 entries.
                
                last_processed_ts = wallet.get('last_processed_candle_ts', 0)
                current_closed_ts = int(last_closed_candle.name.timestamp()) if hasattr(last_closed_candle.name, 'timestamp') else int(pd.to_datetime(last_closed_candle['Date']).timestamp()) if 'Date' in last_closed_candle else 0
                
                # Fix for index timestamp extraction if index is datetime
                if isinstance(df.index, pd.DatetimeIndex):
                     current_closed_ts = int(df.index[-2].timestamp())

                if signal != 0 and current_closed_ts > last_processed_ts:
                    # Calculate Position Size
                    pct = config['strategy'].get('compounding_pct', 0.95)
                    capital = wallet['balance']
                    qty = (capital * pct) / current_price
                    
                    # Logic specifics (Target/Stop) using ATR from the Closed Candle (confirmed)
                    vol = last_closed_candle.get('custom_atr', 0)
                    target_factor = config['strategy']['custom_target_factor']
                    stop_factor = config['strategy']['custom_stop_factor']
                    
                    entry_executed = False
                    
                    if signal == 1: # Long
                        wallet['position'] = qty
                        wallet['entry_price'] = current_price
                        wallet['target_price'] = current_price + (vol * target_factor)
                        wallet['stop_price'] = current_price - (vol * stop_factor)
                        entry_executed = True
                        print(f"!!! ENTRY LONG at {current_price}")
                        
                    elif signal == -1: # Short
                        wallet['position'] = -qty
                        wallet['entry_price'] = current_price
                        wallet['target_price'] = current_price - (vol * target_factor)
                        wallet['stop_price'] = current_price + (vol * stop_factor)
                        entry_executed = True
                        print(f"!!! ENTRY SHORT at {current_price}")
                    
                    if entry_executed:
                        wallet['trades_history'].append({
                            "date": datetime.now().isoformat(),
                            "type": "ENTRY",
                            "side": "LONG" if signal == 1 else "SHORT",
                            "price": current_price,
                            "qty": qty
                        })
                        wallet['last_processed_candle_ts'] = current_closed_ts
                        save_wallet(wallet)
                
                # Update processed TS even if no signal, to allow skipping? No, only update if we acted or if we want to skip 'no signal' bars?
                # No, only update if we Acted. 
                # Wait: If we start the bot in the middle of a Buy signal bar (that closed 2 hours ago), should we enter?
                # Backtest says yes (it enters at Close). Late entry is risky but acceptable for paper trading.
                # Better approach: Update 'last_processed_candle_ts' if we decide NOT to enter too? 
                # No, simpler: Just check if we already have a position. If 0 position, check signal.
                # If we miss the entry window (e.g. signal was 3 hours ago), price might have moved.
                # Let's keep it simple: If signal is present on the last closed bar and we haven't traded THAT bar index, we enter.

            # Log status
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.2f} | Bal: {wallet['balance']:.2f} | Pos: {wallet['position']:.4f} | Stop: {wallet['stop_price']:.2f} | Targ: {wallet['target_price']:.2f}")
            
            # Sleep 60s
            time.sleep(60)

        except KeyboardInterrupt:
            print("Stopping Paper Trading...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_paper_trading()
