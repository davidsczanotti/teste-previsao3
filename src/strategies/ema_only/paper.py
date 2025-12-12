import time
import json
import os
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.binance_client import get_historical_klines, get_current_price
from src.strategies.ema_only.backtest import calculate_mas, generate_signals

# Config Paths
STRATEGY_DIR = Path(__file__).parent
CONFIG_PATH = STRATEGY_DIR / "config.json"
WALLET_PATH = STRATEGY_DIR / "paper_wallet.json"
LOG_FILE = STRATEGY_DIR / "paper_trading.log"
TRADES_CSV = STRATEGY_DIR / "trades.csv"

# Setup Logger
logger = logging.getLogger("PaperTrader")
logger.setLevel(logging.INFO)
# Limpa handlers anteriores se houver (para evitar duplicação em reloads)
if logger.hasHandlers():
    logger.handlers.clear()

# Formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# File Handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def load_wallet():
    if not WALLET_PATH.exists():
        wallet = {
            "balance": 1000.0,
            "position": 0.0, 
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

def log_trade_to_csv(trade_data):
    """Salva os detalhes do trade em CSV para auditoria fácil."""
    file_exists = TRADES_CSV.exists()
    
    # Definir colunas
    fieldnames = [
        "date", "type", "side", "price", "qty", "pnl", "balance", 
        "reason", "cci", "ema_fast", "ema_slow", "atr", "ref_bias"
    ]
    
    with open(TRADES_CSV, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        # Filtra apenas os campos que temos
        row = {k: trade_data.get(k, '') for k in fieldnames}
        writer.writerow(row)

def get_indicator_snapshot(row):
    """Extrai os indicadores técnicos de uma linha do dataframe para log."""
    return {
        "cci": f"{row.get('cci', 0):.2f}",
        "ema_fast": f"{row.get('sma_fast', 0):.2f}", # Ajustar conforme lógica (custom usa sma?)
        "ema_slow": f"{row.get('sma_slow', 0):.2f}",
        "atr": f"{row.get('custom_atr', row.get('atr', 0)):.2f}",
        "close": f"{row.get('close', 0):.2f}",
        "ref_bias": f"{row.get('ref_bias', 0)}"
    }

def run_paper_trading():
    config = load_config()
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
    logger.info(f"--- Paper Trading Iniciado: {symbol} {timeframe} ---")
    logger.info(f"Modo de Log: Arquivo ({LOG_FILE}) e Console.")
    
    load_wallet() # Init wallet if needed

    while True:
        try:
            # 1. Fetch Data
            # logger.debug("Buscando dados históricos...")
            df = get_historical_klines(symbol, timeframe, "90 days ago UTC")
            
            # Fetch Ref Data (Daily)
            if config['strategy'].get('ref_filter_enabled', False):
                ref_tf = config['data'].get('ref_timeframe', '1d')
                df_ref = get_historical_klines(symbol, ref_tf, "365 days ago UTC")
                
                if not df_ref.empty:
                    ref_period = config['strategy'].get('ref_ema_period', 200)
                    df_ref['ref_ema'] = df_ref['close'].ewm(span=ref_period).mean()
                    
                    df['Date'] = pd.to_datetime(df['Date'])
                    df_ref['Date'] = pd.to_datetime(df_ref['Date'])
                    df = df.sort_values('Date')
                    df_ref = df_ref.sort_values('Date')
                    
                    df = pd.merge_asof(df, df_ref[['Date', 'ref_ema']], on='Date')
                else:
                    logger.warning("Não foi possível buscar dados de referência (Daily).")

            if df.empty:
                logger.error("Dados vazios recebidos da Binance. Tentando novamente em 60s...")
                time.sleep(60)
                continue
                
            # 2. Calculate Indicators
            df = calculate_mas(df, config)
            df = generate_signals(df, config)
            
            # Último candle FECHADO (decisão técnica)
            last_closed_candle = df.iloc[-2]
            # Candle atual (para projeção ou verificação de preço live)
            # current_candle_metrics = df.iloc[-1] 
            
            # Preço REAL (book) para execução
            current_price = get_current_price(symbol)
            
            wallet = load_wallet()
            position = wallet['position']
            
            # 3. Check Exits (Execução imediata baseada no preço atual)
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
                        reason = "Signal Reverse (Short Detectado)"
                        
                elif position < 0: # Short
                    if current_price >= wallet['stop_price']:
                        exit_signal = True
                        reason = "Stop Loss"
                    elif current_price <= wallet['target_price']:
                        exit_signal = True
                        reason = "Take Profit"
                    elif last_closed_candle['signal'] == 1:
                        exit_signal = True
                        reason = "Signal Reverse (Long Detectado)"
                
                if exit_signal:
                    if position > 0:
                        pnl = (current_price - wallet['entry_price']) * position
                    else:
                        pnl = (wallet['entry_price'] - current_price) * abs(position)
                    
                    new_balance = wallet['balance'] + pnl
                    
                    # Log Auditável
                    logger.info(f"!!! SAÍDA EXECUTADA: {reason} | PnL: {pnl:.2f} | Saldo: {new_balance:.2f}")
                    
                    trade_record = {
                        "date": datetime.now().isoformat(),
                        "type": "EXIT",
                        "side": "LONG" if position > 0 else "SHORT",
                        "price": current_price,
                        "qty": abs(position),
                        "pnl": pnl,
                        "balance": new_balance,
                        "reason": reason,
                        # Snapshot de indicadores na SAÍDA (pode ajudar a entender reversões)
                        **get_indicator_snapshot(last_closed_candle)
                    }
                    
                    wallet['balance'] = new_balance
                    wallet['trades_history'].append(trade_record)
                    wallet['position'] = 0.0
                    wallet['entry_price'] = 0.0
                    wallet['stop_price'] = 0.0
                    wallet['target_price'] = 0.0
                    
                    save_wallet(wallet)
                    log_trade_to_csv(trade_record)

            # 4. Check Entries
            if wallet['position'] == 0:
                signal = last_closed_candle['signal']
                
                # Controle para não entrar repetidamente no mesmo candle
                last_processed_ts = wallet.get('last_processed_candle_ts', 0)
                
                # Tenta pegar timestamp, compatível com index datetime ou coluna Date
                if isinstance(df.index, pd.DatetimeIndex):
                     current_closed_ts = int(df.index[-2].timestamp())
                elif 'Date' in last_closed_candle:
                     current_closed_ts = int(pd.to_datetime(last_closed_candle['Date']).timestamp())
                else:
                     current_closed_ts = int(time.time()) # Fallback perigoso, mas evita crash

                if signal != 0 and current_closed_ts > last_processed_ts:
                    # Configurar Posição
                    pct = config['strategy'].get('compounding_pct', 0.95)
                    capital = wallet['balance']
                    qty = (capital * pct) / current_price
                    
                    # Targets via ATR do candle fechado
                    vol = last_closed_candle.get('custom_atr', last_closed_candle.get('atr', current_price*0.01))
                    target_factor = config['strategy']['custom_target_factor']
                    stop_factor = config['strategy']['custom_stop_factor']
                    
                    entry_executed = False
                    side = ""
                    
                    if signal == 1: # Long
                        side = "LONG"
                        wallet['position'] = qty
                        wallet['entry_price'] = current_price
                        wallet['target_price'] = current_price + (vol * target_factor)
                        wallet['stop_price'] = current_price - (vol * stop_factor)
                        entry_executed = True
                        
                    elif signal == -1: # Short
                        side = "SHORT"
                        wallet['position'] = -qty
                        wallet['entry_price'] = current_price
                        wallet['target_price'] = current_price - (vol * target_factor)
                        wallet['stop_price'] = current_price + (vol * stop_factor)
                        entry_executed = True
                    
                    if entry_executed:
                        snapshot = get_indicator_snapshot(last_closed_candle)
                        reason_str = f"Sinal {signal} no candle {last_closed_candle['Date']}"
                        
                        logger.info(f"!!! ENTRADA {side} executada a {current_price:.2f}. Motivo: {reason_str}")
                        logger.info(f"Audit Context: CCI={snapshot['cci']}, ATR={snapshot['atr']}, RefBias={snapshot['ref_bias']}")
                        
                        trade_record = {
                            "date": datetime.now().isoformat(),
                            "type": "ENTRY",
                            "side": side,
                            "price": current_price,
                            "qty": qty,
                            "pnl": 0,
                            "balance": wallet['balance'],
                            "reason": reason_str,
                            **snapshot
                        }

                        wallet['trades_history'].append(trade_record)
                        wallet['last_processed_candle_ts'] = current_closed_ts
                        save_wallet(wallet)
                        log_trade_to_csv(trade_record)
                
                # Se não entrou, mas o candle já fechou, podemos logar um "Sinal Ignorado" se houver lógica de filtro extra?
                # Por enquanto, só logamos status normal.

            # Status Log periódico
            status_msg = f"Price: {current_price:.2f} | Bal: {wallet['balance']:.2f} | Pos: {wallet['position']:.4f}"
            if wallet['position'] != 0:
                dist_tp = abs(wallet['target_price'] - current_price)
                dist_sl = abs(wallet['stop_price'] - current_price)
                status_msg += f" | TP em: {dist_tp:.2f} | SL em: {dist_sl:.2f}"
            
            logger.info(status_msg)
            
            # Sleep 60s
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Parando Paper Trading (User Interrupt)...")
            break
        except Exception as e:
            logger.exception("Erro crítico no loop principal:")
            time.sleep(60)

if __name__ == "__main__":
    run_paper_trading()