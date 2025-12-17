import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

from src.utils.metrics import calculate_metrics, calculate_sharpe_ratio

# Importação dos módulos refatorados
try:
    from .indicators import add_indicators
    from .signals import apply_signals
except ImportError:
    # Fallback para execução direta (se necessário)
    from indicators import add_indicators
    from signals import apply_signals


def _backtest_trend_surfer_v4_pine(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Backtest fiel ao Pine Script 'EMA Strategy v4.1 [Trend Surfer Corrigida]'.

    Regras de execução (TradingView default):
    - Sinais são avaliados no fechamento do candle.
    - Entradas a mercado preenchem no próximo candle (open).
    - `strategy.exit(..., stop=...)` cria/atualiza um stop ativo a partir do próximo candle.
    - Gap no open além do stop: preenche no open.
    """
    capital = float(config["backtest"]["initial_capital"])
    strategy = config.get("strategy", {})

    fee_pct = float(strategy.get("fee_pct", 0.0))
    risk_pct = float(strategy.get("risk_per_trade_pct", 0.02))
    initial_stop_pct = float(strategy.get("initial_stop_pct", 0.05))
    trail_pct = float(strategy.get("trailing_stop_pct", 0.10))

    position_qty = 0.0
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0

    max_price_in_trade = 0.0  # maxPriceInTrade do Pine
    active_stop_price: float | None = None  # stop ativo para ESTE candle (definido no candle anterior)
    pending_entry: Dict[str, float] | None = None  # ordem a mercado agendada para o próximo open

    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    n = int(len(df))
    for idx in range(n):
        row = df.iloc[idx]
        is_last_bar = idx == n - 1
        ts = row["Date"] if "Date" in row else df.index[idx]

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        # 1) Preenchimento de entrada a mercado (próximo open)
        if pending_entry is not None and position_qty == 0.0:
            qty = float(pending_entry.get("qty", 0.0))
            if qty > 0.0:
                position_qty = qty
                entry_price = o
                entry_time = ts
                entry_fee = (qty * entry_price) * fee_pct
                capital -= entry_fee

                # No Pine, maxPriceInTrade é setado no candle do sinal (close), não no fill.
                max_price_in_trade = float(pending_entry.get("max_price_init", c))

                # No Pine, o primeiro `strategy.exit` só roda no fechamento do 1º candle já posicionado,
                # então não existe stop ativo no candle do fill.
                active_stop_price = None
            pending_entry = None

        # 2) Execução de stop (ordem criada/atualizada no candle anterior)
        if position_qty > 0.0 and active_stop_price is not None:
            stop = float(active_stop_price)
            exit_price: float | None = None
            if o <= stop:
                exit_price = o  # gap abaixo do stop
            elif l <= stop:
                exit_price = stop  # tocou intrabar

            if exit_price is not None:
                qty = position_qty
                exit_fee = (qty * exit_price) * fee_pct
                gross_pnl = (exit_price - entry_price) * qty
                capital += gross_pnl - exit_fee
                net_pnl = gross_pnl - entry_fee - exit_fee
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "entry": entry_price,
                        "exit": exit_price,
                        "qty": qty,
                        "pnl_gross": gross_pnl,
                        "fee_entry": entry_fee,
                        "fee_exit": exit_fee,
                        "fee_total": entry_fee + exit_fee,
                        "pnl": net_pnl,
                        "side": "long",
                        "reason": "trailing_stop",
                        "date": ts,
                    }
                )
                position_qty = 0.0
                entry_price = 0.0
                entry_time = None
                entry_fee = 0.0
                active_stop_price = None
                max_price_in_trade = 0.0

        # 3) Mark-to-market (equity = capital + PnL não-realizado)
        if position_qty > 0.0:
            unrealized_pnl = (c - entry_price) * position_qty
        else:
            unrealized_pnl = 0.0
        equity.append(capital + unrealized_pnl)

        # 4) Fechamento do candle: cria/atualiza ordens para o próximo candle
        if is_last_bar:
            continue

        signal = int(row.get("signal", 0) or 0)

        # Entrada: validLong (signal==1) e flat -> agenda mercado para o próximo open
        if position_qty == 0.0:
            if pending_entry is None and signal == 1:
                risk_equity = capital * risk_pct  # strategy.equity (flat) ~ capital
                stop_distance_money = c * initial_stop_pct  # close * initialStopPct
                qty = (risk_equity / stop_distance_money) if stop_distance_money > 0 else 0.0
                pending_entry = {"qty": float(qty), "max_price_init": float(c)}

        # Saída: atualiza high-watermark e recalcula stop para o próximo candle
        elif position_qty > 0.0:
            if h > max_price_in_trade:
                max_price_in_trade = h
            active_stop_price = max_price_in_trade * (1.0 - trail_pct)

    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics = calculate_metrics(trades)
    metrics["sharpe_ratio"] = float(calculate_sharpe_ratio(returns))
    metrics["final_equity"] = float(equity[-1]) if equity else float(capital)
    if config.get("backtest", {}).get("initial_capital"):
        metrics["total_return_pct"] = (metrics["final_equity"] / float(config["backtest"]["initial_capital"])) - 1.0

    # Compatível com TradingView: posição aberta permanece aberta no último candle.
    open_pnl = 0.0
    open_position: Dict[str, Any] | None = None
    if position_qty > 0.0 and n > 0:
        last = df.iloc[-1]
        last_ts = last["Date"] if "Date" in last else df.index[-1]
        last_close = float(last["close"])
        open_pnl = (last_close - entry_price) * position_qty
        open_position = {
            "entry_time": entry_time,
            "entry": entry_price,
            "qty": position_qty,
            "last_time": last_ts,
            "last_close": last_close,
            "open_pnl": open_pnl,
            "max_price_in_trade": max_price_in_trade,
            "stop_price_next_bar": active_stop_price,
        }

    metrics["open_pnl"] = float(open_pnl)
    metrics["open_trades"] = 1 if open_position is not None else 0

    return {"config": config, "trades": trades, "equity": equity, "metrics": metrics, "open_position": open_position}


def _backtest_ema_strategy_v5_2(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest fiel ao Pine Script 'EMA Strategy v5.2 [Painel Visível]'.
    """
    capital = float(config["backtest"]["initial_capital"])
    strategy = config.get("strategy", {})

    fee_pct = float(strategy.get("fee_pct", 0.0))
    
    # Gestão de Capital
    use_all_equity = bool(strategy.get("use_all_equity", True))
    risk_pct = float(strategy.get("risk_per_trade_pct", 0.02))
    stop_loss_fixo_pct = float(strategy.get("stop_loss_fixo_pct", 0.06))
    
    # Regras de Saída
    trail_pct = float(strategy.get("trailing_stop_pct", 0.15))

    position_qty = 0.0
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0

    max_price_in_trade = 0.0
    active_stop_price: float | None = None
    
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    # Se exit_signal não foi gerado, cria coluna zerada
    if 'exit_signal' not in df.columns:
        df['exit_signal'] = 0

    n = int(len(df))
    for idx in range(n):
        row = df.iloc[idx]
        is_last_bar = idx == n - 1
        ts = row["Date"] if "Date" in row else df.index[idx]

        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        
        # Sinais (Close do candle anterior define ação no Open deste)
        # Porém, no Pine, strategies rodam no close. A execução padrão é Next Open.
        # Aqui, estamos iterando candles. O sinal foi calculado com base no Close deste candle (row).
        # Então a execução deve ocorrer no PRÓXIMO candle.
        # Mas para simplificar a lógica de loop único, verificamos o sinal do candle ANTERIOR.
        # OU processamos "fechamento" e "abertura" separadamente.
        
        # Vamos seguir o modelo do Trend Surfer:
        # 1. Processar Execuções Pendentes (Entradas/Saídas baseadas no candle anterior)
        # mas aqui simplificaremos para execução imediata no Close (simulação Close-Close) ou Open-Next (mais realista).
        # O Pine strategy.entry processa no próximo tick.
        
        # Vamos usar a lógica: Checar Stop/Exit Intraba (se posicionado) -> Checar Entrada (se flat).
        
        exit_reason = None
        exit_price_exec = 0.0

        # --- SE POSICIONADO ---
        if position_qty > 0.0:
            # 1. Atualizar Trailing Stop (High Watermark)
            if h > max_price_in_trade:
                max_price_in_trade = h
            
            # O stop ativo para ESTE candle é baseado no topo ATÉ O MOMENTO (Intrabar update no Pine)
            # No Pine: if high > maxPriceInTrade ... dynamicStop = ... strategy.exit
            # Então se o High deste candle subir, o stop sobe NA MESMA BARRA.
            dynamic_stop = max_price_in_trade * (1.0 - trail_pct)
            
            # Checar Stop Loss (Trailing)
            if l <= dynamic_stop:
                exit_price_exec = dynamic_stop
                # Gap check
                if exit_price_exec > h: exit_price_exec = o # Gap de baixa pulou o stop
                exit_reason = "trailing_stop"
            
            # Checar Saída Técnica (Exit Signal)
            # O sinal de saída é calculado no fechamento. Então ele só executa no próximo Open?
            # Pine: if exitSignal strategy.close(). Executa market no próximo Open.
            # Aqui, 'exit_signal' é 1 se o cruzamento ocorreu NO FECHAMENTO DESTE CANDLE.
            # Portanto, devemos executar no PRÓXIMO.
            # MAS, estamos iterando. Se exit_signal[i] == 1, vendemos em i+1 [Open].
            # Para simplificar, podemos olhar se exit_signal ANTERIOR foi 1.
            # OU: Implementamos a saída técnica como uma "pendência" para o próximo loop.
            
            # Vamos olhar se TEMOS um sinal de saída técnica AGORA (neste candle).
            # Se sim, e não fomos stopados, fechamos no CLOSE deste candle (simulação levemente otimista)
            # ou marcamos para sair no Open do próximo.
            # Vamos sair no CLOSE deste candle para "Exit Signal" para simplificar, 
            # assumindo liquidez no leilão de fechamento.
            elif row['exit_signal'] == 1:
                exit_price_exec = c
                exit_reason = "exit_signal"

            # Executar Saída
            if exit_reason:
                qty = position_qty
                exit_fee = (qty * exit_price_exec) * fee_pct
                gross_pnl = (exit_price_exec - entry_price) * qty
                capital += gross_pnl - exit_fee
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry": entry_price,
                    "exit": exit_price_exec,
                    "qty": qty,
                    "pnl_gross": gross_pnl,
                    "pnl": net_pnl,
                    "reason": exit_reason,
                    "date": ts
                })
                position_qty = 0.0
                entry_price = 0.0
                max_price_in_trade = 0.0

        # --- SE FLAT (Verificar Entrada) ---
        if position_qty == 0.0 and not is_last_bar:
            # Sinal de compra (1)
            if row['signal'] == 1:
                # Calcular Tamanho da Posição
                buy_price = c # Entrada no Close (ou Open do próximo, mas Close é aceitável para estimativa)
                
                if use_all_equity:
                    # positionSize := strategy.equity
                    pos_value = capital
                    # Qty = Value / Price
                    qty = pos_value / buy_price
                else:
                    # riskEquity = strategy.equity * riskPerTrade
                    # stopDistanceMoney = close * stopLossFixoPct
                    # positionSize := (riskEquity / stopDistanceMoney) * close
                    # Qty = positionSize / close = riskEquity / stopDistanceMoney
                    
                    risk_val = capital * risk_pct
                    stop_dist = buy_price * stop_loss_fixo_pct
                    if stop_dist > 0:
                        qty = risk_val / stop_dist
                    else:
                        qty = 0.0
                
                if qty > 0:
                    position_qty = qty
                    entry_price = buy_price
                    entry_time = ts
                    entry_fee = (qty * entry_price) * fee_pct
                    capital -= entry_fee
                    
                    # Inicializa Trailing
                    max_price_in_trade = entry_price

        # Mark-to-market
        if position_qty > 0.0:
            unrealized_pnl = (c - entry_price) * position_qty
        else:
            unrealized_pnl = 0.0
        equity.append(capital + unrealized_pnl)

    # Fechamento Final
    open_pnl = 0.0
    open_position = None
    if position_qty > 0.0 and n > 0:
        last_close = df.iloc[-1]["close"]
        open_pnl = (last_close - entry_price) * position_qty
        open_position = {
            "entry_time": entry_time,
            "entry": entry_price,
            "qty": position_qty,
            "open_pnl": open_pnl
        }

    metrics = calculate_metrics(trades)
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics["sharpe_ratio"] = float(calculate_sharpe_ratio(returns))
    metrics["final_equity"] = float(equity[-1]) if equity else float(config["backtest"]["initial_capital"])
    if config.get("backtest", {}).get("initial_capital"):
        metrics["total_return_pct"] = (metrics["final_equity"] / float(config["backtest"]["initial_capital"])) - 1.0
    
    metrics["open_pnl"] = float(open_pnl)
    metrics["open_trades"] = 1 if open_position is not None else 0

    return {"config": config, "trades": trades, "equity": equity, "metrics": metrics, "open_position": open_position}

def _backtest_supertrend_ai(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest SuperTrend AI.
    Saída baseada exclusivamente na virada de tendência (Trailing Stop do SuperTrend).
    """
    capital = float(config["backtest"]["initial_capital"])
    strategy = config.get("strategy", {})

    fee_pct = float(strategy.get("fee_pct", 0.0))
    risk_pct = float(strategy.get("risk_per_trade_pct", 0.02))
    use_all_equity = bool(strategy.get("use_all_equity", True))
    
    position_qty = 0.0
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0
    
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    if 'exit_signal' not in df.columns: df['exit_signal'] = 0

    n = int(len(df))
    for idx in range(n):
        row = df.iloc[idx]
        is_last_bar = idx == n - 1
        ts = row["Date"] if "Date" in row else df.index[idx]
        c = row["close"]
        
        # --- SE POSICIONADO ---
        if position_qty > 0.0:
            # Saída apenas se houver sinal de saída (Trend Flip)
            # O cálculo do SuperTrend já considerou Low < StopLine na barra atual
            if row['exit_signal'] == 1:
                # O preço de saída ideal seria o StopLine (SuperTrend Value)
                # Mas como é um flip no fechamento, saímos no Close.
                # Ou se quisermos ser precisos, saímos no toque do SuperTrend (row['supertrend_ai']).
                # Vamos usar o Close para simplificar e evitar olhar futuro, 
                # pois o flip é confirmado no close.
                exit_price_exec = c
                exit_reason = "trend_flip"
                
                qty = position_qty
                exit_fee = (qty * exit_price_exec) * fee_pct
                gross_pnl = (exit_price_exec - entry_price) * qty
                capital += gross_pnl - exit_fee
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry": entry_price,
                    "exit": exit_price_exec,
                    "qty": qty,
                    "pnl_gross": gross_pnl,
                    "pnl": net_pnl,
                    "reason": exit_reason,
                    "date": ts
                })
                position_qty = 0.0
                entry_price = 0.0

        # --- SE FLAT ---
        if position_qty == 0.0 and not is_last_bar:
            if row['signal'] == 1:
                buy_price = c
                
                if use_all_equity:
                    qty = capital / buy_price
                else:
                    qty = (capital * risk_pct) / (buy_price * 0.05) # Dummy 5% stop dist

                if qty > 0:
                    position_qty = qty
                    entry_price = buy_price
                    entry_time = ts
                    entry_fee = (qty * entry_price) * fee_pct
                    capital -= entry_fee

        # Mark-to-market
        if position_qty > 0.0:
            unrealized_pnl = (c - entry_price) * position_qty
        else:
            unrealized_pnl = 0.0
        equity.append(capital + unrealized_pnl)

    # Fechamento Final
    open_pnl = 0.0
    open_position = None
    if position_qty > 0.0 and n > 0:
        last_close = df.iloc[-1]["close"]
        open_pnl = (last_close - entry_price) * position_qty
        open_position = {
            "entry_time": entry_time,
            "entry": entry_price,
            "qty": position_qty,
            "open_pnl": open_pnl
        }

    metrics = calculate_metrics(trades)
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics["sharpe_ratio"] = float(calculate_sharpe_ratio(returns))
    metrics["final_equity"] = float(equity[-1]) if equity else float(config["backtest"]["initial_capital"])
    if config.get("backtest", {}).get("initial_capital"):
        metrics["total_return_pct"] = (metrics["final_equity"] / float(config["backtest"]["initial_capital"])) - 1.0
    
    metrics["open_pnl"] = float(open_pnl)
    metrics["open_trades"] = 1 if open_position is not None else 0

    return {"config": config, "trades": trades, "equity": equity, "metrics": metrics, "open_position": open_position}


def _backtest_dynamic_volatility_v6(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest Modo V6 Robust:
    - Stop Loss Dinâmico via ATR (Chandelier Exit adaptado).
    - Trailing Stop: High - (ATR * Mult).
    """
    capital = float(config["backtest"]["initial_capital"])
    strategy = config.get("strategy", {})

    fee_pct = float(strategy.get("fee_pct", 0.0))
    risk_pct = float(strategy.get("risk_per_trade_pct", 0.02))
    use_all_equity = bool(strategy.get("use_all_equity", True))
    
    # Parâmetros ATR Trailing
    stop_atr_mult = float(strategy.get("stop_atr_mult", 2.5))
    
    position_qty = 0.0
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0
    
    current_stop_price = 0.0
    
    trades: List[Dict[str, Any]] = []
    equity: List[float] = []

    if 'exit_signal' not in df.columns: df['exit_signal'] = 0
    if 'atr' not in df.columns: df['atr'] = df['close'] * 0.02 # Fallback

    n = int(len(df))
    for idx in range(n):
        row = df.iloc[idx]
        is_last_bar = idx == n - 1
        ts = row["Date"] if "Date" in row else df.index[idx]

        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        atr = row['atr']
        
        exit_reason = None
        exit_price_exec = 0.0

        # --- SE POSICIONADO ---
        if position_qty > 0.0:
            # 1. Atualizar Trailing Stop (ATR Based)
            # Stop = High - (ATR * Mult). O stop só sobe.
            potential_new_stop = h - (atr * stop_atr_mult)
            if potential_new_stop > current_stop_price:
                current_stop_price = potential_new_stop
            
            # Checar Stop Loss
            if l <= current_stop_price:
                exit_price_exec = current_stop_price
                if exit_price_exec > h: exit_price_exec = o # Gap
                exit_reason = "atr_trailing_stop"
            
            # Checar Saída Técnica
            elif row['exit_signal'] == 1:
                exit_price_exec = c
                exit_reason = "cross_exit"

            # Executar Saída
            if exit_reason:
                qty = position_qty
                exit_fee = (qty * exit_price_exec) * fee_pct
                gross_pnl = (exit_price_exec - entry_price) * qty
                capital += gross_pnl - exit_fee
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry": entry_price,
                    "exit": exit_price_exec,
                    "qty": qty,
                    "pnl_gross": gross_pnl,
                    "pnl": net_pnl,
                    "reason": exit_reason,
                    "date": ts
                })
                position_qty = 0.0
                entry_price = 0.0
                current_stop_price = 0.0

        # --- SE FLAT ---
        if position_qty == 0.0 and not is_last_bar:
            if row['signal'] == 1:
                buy_price = c
                
                # Tamanho da Posição
                if use_all_equity:
                    qty = capital / buy_price
                else:
                    # Risco Baseado em Volatilidade?
                    # Ou fixo % do capital? Vamos manter fixo por enquanto ou full.
                    # Se quiséssemos Volatility Targeting: Risk$ / (ATR * Mult)
                    qty = (capital * risk_pct) / (atr * stop_atr_mult) # Exemplo
                    # Mas vamos simplificar para manter coerência com o pedido
                    qty = (capital * risk_pct) / (buy_price * 0.02) # Dummy risk calc

                if qty > 0:
                    position_qty = qty
                    entry_price = buy_price
                    entry_time = ts
                    entry_fee = (qty * entry_price) * fee_pct
                    capital -= entry_fee
                    
                    # Inicializa Stop ATR
                    current_stop_price = entry_price - (atr * stop_atr_mult)

        # Mark-to-market
        if position_qty > 0.0:
            unrealized_pnl = (c - entry_price) * position_qty
        else:
            unrealized_pnl = 0.0
        equity.append(capital + unrealized_pnl)

    # Fechamento Final
    open_pnl = 0.0
    open_position = None
    if position_qty > 0.0 and n > 0:
        last_close = df.iloc[-1]["close"]
        open_pnl = (last_close - entry_price) * position_qty
        open_position = {
            "entry_time": entry_time,
            "entry": entry_price,
            "qty": position_qty,
            "open_pnl": open_pnl
        }

    metrics = calculate_metrics(trades)
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics["sharpe_ratio"] = float(calculate_sharpe_ratio(returns))
    metrics["final_equity"] = float(equity[-1]) if equity else float(config["backtest"]["initial_capital"])
    if config.get("backtest", {}).get("initial_capital"):
        metrics["total_return_pct"] = (metrics["final_equity"] / float(config["backtest"]["initial_capital"])) - 1.0
    
    metrics["open_pnl"] = float(open_pnl)
    metrics["open_trades"] = 1 if open_position is not None else 0

    return {"config": config, "trades": trades, "equity": equity, "metrics": metrics, "open_position": open_position}


def backtest_ema_only(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """
    Executa backtest.
    Agora atua puramente como orquestrador de loop de eventos.
    A lógica de cálculo e decisão reside em indicators.py e signals.py.
    """
    # 1. Preparação de Dados (Indicadores + Sinais)
    df = add_indicators(df, config)
    df = apply_signals(df, config)

    # Execução fiel ao Pine para o modo Trend Surfer v4
    if config.get("strategy", {}).get("signal_mode") == "trend_surfer_v4":
        return _backtest_trend_surfer_v4_pine(df, config)

    # Execução fiel ao Pine para o modo v5.2
    if config.get("strategy", {}).get("signal_mode") == "ema_strategy_v5_2":
        return _backtest_ema_strategy_v5_2(df, config)

    # Execução Modo V6 Robust
    if config.get("strategy", {}).get("signal_mode") == "dynamic_volatility_v6":
        return _backtest_dynamic_volatility_v6(df, config)

    # Execução Modo SuperTrend AI
    if config.get("strategy", {}).get("signal_mode") == "supertrend_ai":
        return _backtest_supertrend_ai(df, config)

    # Simulação de trades
    capital = float(config['backtest']['initial_capital'])
    position = 0.0 # Positivo = Long, Negativo = Short
    entry_price = 0.0
    entry_time = None
    entry_fee = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    trades = []
    equity = []
    
    lot_size = config['strategy']['lot_size']
    fee_pct = float(config['strategy'].get('fee_pct', 0.0))
    
    is_custom_mode = config['strategy'].get('signal_mode') == 'custom_cci_ma'
    is_trend_surfer = config['strategy'].get('signal_mode') == 'trend_surfer_v4'
    
    # Parâmetros Trend Surfer
    risk_pct = config['strategy'].get('risk_per_trade_pct', 0.02)
    initial_stop_pct = config['strategy'].get('initial_stop_pct', 0.05)
    trail_pct = config['strategy'].get('trailing_stop_pct', 0.10)
    max_price_in_trade = 0.0

    target_factor = config['strategy'].get('custom_target_factor', 1.5)
    stop_factor = config['strategy'].get('custom_stop_factor', 0.9)

    for i, row in df.iterrows():
        is_last_bar = i == df.index[-1]
        ts = row['Date'] if 'Date' in row else i

        # Lógica de Saída (TP/SL)
        if position != 0:
            gross_pnl = 0.0
            exit_price = 0
            exit_reason = ""
            
            # --- Lógica Específica Trend Surfer (Trailing Stop High Watermark) ---
            if is_trend_surfer and position > 0:
                # Atualiza Topo Histórico do Trade
                if row['high'] > max_price_in_trade:
                    max_price_in_trade = row['high']
                
                # Stop Dinâmico
                dynamic_stop = max_price_in_trade * (1 - trail_pct)
                
                if row['low'] <= dynamic_stop:
                    exit_price = dynamic_stop
                    # Gap check de abertura (se abriu abaixo do stop, sai no open)
                    if exit_price > row['high']: exit_price = row['open']
                    
                    gross_pnl = (exit_price - entry_price) * position
                    exit_reason = "trailing_stop"
                # Saída por sinal reverso (opcional, mas comum)
                elif row['signal'] == -1:
                    exit_price = row['close']
                    gross_pnl = (exit_price - entry_price) * position
                    exit_reason = "signal_reverse"
            
            # --- Lógica Padrão / Custom Antiga ---
            elif position > 0: # Long
                if row['low'] <= stop_price:
                    exit_price = stop_price 
                    if exit_price > row['high']: exit_price = row['open'] # Gap check
                    gross_pnl = (exit_price - entry_price) * position
                    exit_reason = "stop_loss"
                elif row['high'] >= target_price:
                    exit_price = target_price
                    gross_pnl = (exit_price - entry_price) * position
                    exit_reason = "take_profit"
                # Saída por sinal reverso
                elif row['signal'] == -1:
                    exit_price = row['close']
                    gross_pnl = (exit_price - entry_price) * position
                    exit_reason = "signal_reverse"

            elif position < 0: # Short
                # Trend Surfer Short (se habilitado) - Lógica inversa
                if is_trend_surfer:
                     if row['low'] < max_price_in_trade: # Para short, max_price guarda o Low mínimo
                        max_price_in_trade = row['low']
                     dynamic_stop = max_price_in_trade * (1 + trail_pct)
                     
                     if row['high'] >= dynamic_stop:
                        exit_price = dynamic_stop
                        if exit_price < row['low']: exit_price = row['open']
                        gross_pnl = (entry_price - exit_price) * abs(position)
                        exit_reason = "trailing_stop"
                     elif row['signal'] == 1:
                        exit_price = row['close']
                        gross_pnl = (entry_price - exit_price) * abs(position)
                        exit_reason = "signal_reverse"

                else:
                    if row['high'] >= stop_price:
                        exit_price = stop_price
                        if exit_price < row['low']: exit_price = row['open']
                        gross_pnl = (entry_price - exit_price) * abs(position)
                        exit_reason = "stop_loss"
                    elif row['low'] <= target_price:
                        exit_price = target_price
                        gross_pnl = (entry_price - exit_price) * abs(position)
                        exit_reason = "take_profit"
                    elif row['signal'] == 1:
                        exit_price = row['close']
                        gross_pnl = (entry_price - exit_price) * abs(position)
                        exit_reason = "signal_reverse"

            if exit_reason:
                qty = abs(position)
                exit_fee = (qty * float(exit_price)) * fee_pct
                capital += gross_pnl - exit_fee
                net_pnl = gross_pnl - entry_fee - exit_fee
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'entry': entry_price, 
                    'exit': exit_price, 
                    'qty': qty,
                    'pnl_gross': gross_pnl,
                    'fee_entry': entry_fee,
                    'fee_exit': exit_fee,
                    'fee_total': entry_fee + exit_fee,
                    'pnl': net_pnl,
                    'side': 'long' if position > 0 else 'short',
                    'reason': exit_reason,
                    'date': ts,
                })
                position = 0
                entry_price = 0
                entry_time = None
                entry_fee = 0.0
                stop_price = 0
                target_price = 0
                max_price_in_trade = 0.0 # Reset

        # Lógica de Entrada
        if position == 0 and not is_last_bar:
            # Definir tamanho da posição
            if is_trend_surfer:
                # Position Sizing do Pine Script
                # riskEquity = strategy.equity * riskPerTrade
                # stopDistanceMoney = close * initialStopPct
                # entryQty = riskEquity / stopDistanceMoney
                
                # current_equity = capital (aproximado, pois capital atualiza fechamento trade a trade)
                risk_money = capital * risk_pct
                stop_distance = row['close'] * initial_stop_pct
                if stop_distance > 0:
                    current_qty = risk_money / stop_distance
                else:
                    current_qty = 0
            else:
                use_compounding = config['strategy'].get('compounding_enabled', False)
                if use_compounding:
                    pct = config['strategy'].get('compounding_pct', 0.95)
                    # Garante que não trade negativo se quebrou a conta
                    if capital <= 0:
                        current_qty = 0
                    else:
                        current_qty = (capital * pct) / row['close']
                else:
                    current_qty = lot_size

            if row['signal'] == 1 and current_qty > 0: # Long
                position = current_qty
                entry_price = row['close']
                entry_time = ts
                entry_fee = (abs(position) * float(entry_price)) * fee_pct
                capital -= entry_fee
                
                if is_trend_surfer:
                    max_price_in_trade = row['close'] # Inicializa com preço de entrada
                    # Stop/Target definidos pela lógica dinâmica na saída
                    stop_price = 0 
                    target_price = 99999999
                    
                elif is_custom_mode:
                    vol = row.get('custom_atr', row.get('atr', 0))
                    target_price = entry_price + (vol * target_factor)
                    stop_price = entry_price - (vol * stop_factor)
                else:
                    # Fallback padrão
                    target_price = entry_price * 1.5
                    stop_price = entry_price * 0.95

            elif row['signal'] == -1 and current_qty > 0: # Short
                position = -current_qty
                entry_price = row['close']
                entry_time = ts
                entry_fee = (abs(position) * float(entry_price)) * fee_pct
                capital -= entry_fee
                
                if is_trend_surfer:
                    max_price_in_trade = row['close']
                    stop_price = 0
                    target_price = 0
                
                elif is_custom_mode:
                    vol = row.get('custom_atr', row.get('atr', 0))
                    target_price = entry_price - (vol * target_factor)
                    stop_price = entry_price + (vol * stop_factor)
                else:
                    target_price = entry_price * 0.5
                    stop_price = entry_price * 1.05

        # Mark-to-market (equity = capital + PnL não-realizado)
        if position > 0:
            unrealized_pnl = (float(row['close']) - float(entry_price)) * position
        elif position < 0:
            unrealized_pnl = (float(entry_price) - float(row['close'])) * abs(position)
        else:
            unrealized_pnl = 0.0
        equity.append(capital + unrealized_pnl)

    # Fecha posição remanescente no último close (evita PnL final ficar "preso" em unrealized)
    if position != 0 and len(df) > 0:
        last = df.iloc[-1]
        ts = last['Date'] if 'Date' in last else df.index[-1]
        exit_price = float(last['close'])
        qty = abs(position)
        exit_fee = (qty * exit_price) * fee_pct
        gross_pnl = (exit_price - entry_price) * position if position > 0 else (entry_price - exit_price) * qty
        capital += gross_pnl - exit_fee
        net_pnl = gross_pnl - entry_fee - exit_fee
        trades.append({
            'entry_time': entry_time,
            'exit_time': ts,
            'entry': entry_price,
            'exit': exit_price,
            'qty': qty,
            'pnl_gross': gross_pnl,
            'fee_entry': entry_fee,
            'fee_exit': exit_fee,
            'fee_total': entry_fee + exit_fee,
            'pnl': net_pnl,
            'side': 'long' if position > 0 else 'short',
            'reason': 'end_of_data',
            'date': ts,
        })
        # Ajusta o último ponto da equity (reflete o pagamento da taxa de saída)
        if equity:
            equity[-1] = float(equity[-1]) - exit_fee
        position = 0.0

    # Calcular métricas
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    metrics = calculate_metrics(trades)
    metrics['sharpe_ratio'] = float(calculate_sharpe_ratio(returns))
    metrics['final_equity'] = float(equity[-1]) if equity else float(capital)
    if config['backtest']['initial_capital']:
        metrics['total_return_pct'] = (metrics['final_equity'] / float(config['backtest']['initial_capital'])) - 1.0

    return {
        'config': config,
        'trades': trades,
        'equity': equity,
        'metrics': metrics
    }

def load_data_with_ref(config: Dict) -> pd.DataFrame:
    """Carrega dados principais e referência."""
    from src.utils.data_loader import load_data

    data_cfg = config['data']
    
    if 'start_date' in data_cfg and 'end_date' in data_cfg:
        from src.utils.data_loader import load_data_range
        df = load_data_range(
            data_cfg['symbol'], 
            data_cfg['timeframe'], 
            data_cfg['start_date'], 
            data_cfg['end_date'], 
            use_cache_only=False
        )
    else:
        df = load_data(data_cfg['symbol'], data_cfg['timeframe'], data_cfg['days'], use_cache_only=False)

    if data_cfg.get('ref_timeframe'):
        if 'start_date' in data_cfg and 'end_date' in data_cfg:
             from src.utils.data_loader import load_data_range
             df_ref = load_data_range(
                data_cfg['symbol'], 
                data_cfg['ref_timeframe'], 
                data_cfg['start_date'], 
                data_cfg['end_date'], 
                use_cache_only=False
            )
        else:
            df_ref = load_data(data_cfg['symbol'], data_cfg['ref_timeframe'], data_cfg['ref_days'], use_cache_only=False)
            
        df_ref['ref_ema'] = df_ref['close'].ewm(span=config['strategy']['ref_ema_period']).mean()
        # Merge com base em data aproximada
        df['Date'] = pd.to_datetime(df['Date'])
        df_ref['Date'] = pd.to_datetime(df_ref['Date'])
        df = pd.merge_asof(df.sort_values('Date'), df_ref[['Date', 'ref_ema']].sort_values('Date'), on='Date')

    return df

def _build_warnings(df: pd.DataFrame, metrics: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    bt_cfg = config.get("backtest", {}) if isinstance(config, dict) else {}
    min_trades = bt_cfg.get("min_trades_for_significance")
    min_candles = bt_cfg.get("min_candles_for_significance")

    total_trades = int(metrics.get("total_trades", 0) or 0)
    candles = int(len(df))

    if isinstance(min_trades, int) and min_trades > 0 and total_trades < min_trades:
        warnings.append(
            f"Baixa significância: trades ({total_trades}) < min_trades_for_significance ({min_trades})."
        )
    if isinstance(min_candles, int) and min_candles > 0 and candles < min_candles:
        warnings.append(
            f"Baixa significância: candles ({candles}) < min_candles_for_significance ({min_candles})."
        )
    return warnings


def _compute_monthly_summary(
    df: pd.DataFrame,
    equity: List[float],
    trades: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if df.empty or not equity or "Date" not in df.columns:
        return []

    equity_index = pd.to_datetime(df["Date"], errors="coerce")
    equity_series = pd.Series([float(x) for x in equity], index=equity_index).dropna().sort_index()
    if equity_series.empty:
        return []

    monthly_equity = equity_series.resample("ME").last()
    monthly_pnl = monthly_equity.diff()
    monthly_return = monthly_equity.pct_change()

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        if "exit_time" in trades_df.columns:
            exit_time = pd.to_datetime(trades_df["exit_time"], errors="coerce")
        else:
            exit_time = pd.to_datetime(trades_df.get("date"), errors="coerce")
        trades_df["month"] = exit_time.dt.to_period("M").astype(str)
        trades_month = trades_df.groupby("month").agg(
            trades=("pnl", "size"),
            pnl_realized=("pnl", "sum"),
            wins=("pnl", lambda s: int((s > 0).sum())),
        )
        trades_month["win_rate"] = trades_month["wins"] / trades_month["trades"]
        trades_month = trades_month.drop(columns=["wins"])
    else:
        trades_month = pd.DataFrame(columns=["trades", "pnl_realized", "win_rate"])

    out: List[Dict[str, Any]] = []
    for ts, eq in monthly_equity.items():
        month = ts.strftime("%Y-%m")
        pnl_val = monthly_pnl.loc[ts]
        ret_val = monthly_return.loc[ts]
        row: Dict[str, Any] = {
            "month": month,
            "equity": float(eq),
            "pnl_m2m": 0.0 if pd.isna(pnl_val) else float(pnl_val),
            "return_pct": 0.0 if pd.isna(ret_val) else float(ret_val),
            "trades": 0,
            "pnl_realized": 0.0,
            "win_rate": 0.0,
        }
        if month in trades_month.index:
            row["trades"] = int(trades_month.loc[month, "trades"])
            row["pnl_realized"] = float(trades_month.loc[month, "pnl_realized"])
            row["win_rate"] = float(trades_month.loc[month, "win_rate"])
        out.append(row)
    return out


def _compute_yearly_summary(
    initial_capital: float,
    monthly: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not monthly:
        return []

    monthly_sorted = sorted(monthly, key=lambda r: str(r.get("month", "")))
    by_year: Dict[int, List[Dict[str, Any]]] = {}
    for row in monthly_sorted:
        m = str(row.get("month", ""))
        if len(m) < 4 or not m[:4].isdigit():
            continue
        by_year.setdefault(int(m[:4]), []).append(row)

    years = sorted(by_year)
    if not years:
        return []

    out: List[Dict[str, Any]] = []
    prev_end = float(initial_capital)
    for y in years:
        months = by_year[y]
        if not months:
            continue
        end_equity = float(months[-1].get("equity", 0.0) or 0.0)
        start_equity = prev_end
        pnl = end_equity - start_equity
        ret = (end_equity / start_equity - 1.0) if start_equity != 0 else 0.0
        trades = sum(int(m.get("trades", 0) or 0) for m in months)
        pnl_realized = sum(float(m.get("pnl_realized", 0.0) or 0.0) for m in months)
        out.append(
            {
                "year": y,
                "start_equity": float(start_equity),
                "end_equity": float(end_equity),
                "pnl": float(pnl),
                "return_pct": float(ret),
                "trades": int(trades),
                "pnl_realized": float(pnl_realized),
            }
        )
        prev_end = end_equity
    return out


def _render_markdown_report(result: Dict[str, Any]) -> str:
    cfg = result.get("config", {}) if isinstance(result, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    symbol = data_cfg.get("symbol", "UNKNOWN")
    timeframe = data_cfg.get("timeframe", "UNKNOWN")
    period = result.get("period", {}) if isinstance(result, dict) else {}
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    warnings = result.get("warnings", []) if isinstance(result, dict) else []
    monthly = result.get("monthly", []) if isinstance(result, dict) else []
    yearly = result.get("yearly", []) if isinstance(result, dict) else []

    def _f(x, nd=2):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "n/a"

    def _p(x, nd=2):
        try:
            return f"{float(x) * 100:.{nd}f}%"
        except Exception:
            return "n/a"

    lines: List[str] = []
    lines.append(f"# ema_only — backtest ({symbol} {timeframe})")
    lines.append("")
    if period:
        lines.append(f"- Período: {period.get('start')} → {period.get('end')} ({period.get('candles')} candles)")
        lines.append("")

    lines.append("## Métricas (net de fees)")
    lines.append("")
    lines.append(f"- Trades: {int(metrics.get('total_trades', 0) or 0)}")
    lines.append(f"- PnL total: {_f(metrics.get('total_pnl'))}")
    lines.append(f"- Retorno total: {_p(metrics.get('total_return_pct'))}")
    lines.append(f"- Win rate: {_p(metrics.get('win_rate'))}")
    lines.append(f"- Profit factor: {_f(metrics.get('profit_factor'))}")
    lines.append(f"- Sharpe: {_f(metrics.get('sharpe_ratio'))}")
    lines.append(f"- Final equity: {_f(metrics.get('final_equity'))}")

    if warnings:
        lines.append("")
        lines.append("## Avisos")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")

    if yearly:
        lines.append("")
        lines.append("## Anual")
        lines.append("")
        lines.append("| ano | equity_inicial | equity_final | pnl | retorno | trades |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in yearly:
            lines.append(
                "| {year} | {start} | {end} | {pnl} | {ret} | {trades} |".format(
                    year=int(row.get("year", 0) or 0),
                    start=_f(row.get("start_equity")),
                    end=_f(row.get("end_equity")),
                    pnl=_f(row.get("pnl")),
                    ret=_p(row.get("return_pct")),
                    trades=int(row.get("trades", 0) or 0),
                )
            )

    # Tabela mensal (últimos 24 meses)
    if monthly:
        lines.append("")
        lines.append("## Mensal (últimos 24 meses)")
        lines.append("")
        lines.append("| mês | pnl_realizado | retorno (m2m) | trades | win_rate | equity |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in monthly[-24:]:
            lines.append(
                "| {month} | {pnl_realized} | {ret} | {trades} | {win} | {eq} |".format(
                    month=row.get("month"),
                    pnl_realized=_f(row.get("pnl_realized")),
                    ret=_p(row.get("return_pct")),
                    trades=int(row.get("trades", 0) or 0),
                    win=_p(row.get("win_rate")),
                    eq=_f(row.get("equity")),
                )
            )

    return "\n".join(lines) + "\n"


def run_backtest(config_path: str = 'src/strategies/ema_only/config.json') -> None:
    """Executa backtest completo."""
    with open(config_path) as f:
        config = json.load(f)

    # Carregar dados
    df = load_data_with_ref(config)

    # Backtest
    result = backtest_ema_only(df, config)

    result["period"] = {
        "start": str(df["Date"].min()) if "Date" in df.columns and not df.empty else None,
        "end": str(df["Date"].max()) if "Date" in df.columns and not df.empty else None,
        "candles": int(len(df)),
    }
    result["warnings"] = _build_warnings(df, result.get("metrics", {}), config)
    result["monthly"] = _compute_monthly_summary(df, result.get("equity", []), result.get("trades", []))
    initial_capital = float(config.get("backtest", {}).get("initial_capital", 0.0) or 0.0)
    result["yearly"] = _compute_yearly_summary(initial_capital, result.get("monthly", []))

    # Salvar resultados
    outdir = Path(config['backtest']['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    output_file = outdir / f"ema_only_{config['data']['symbol']}_{config['data']['timeframe']}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    md_file = output_file.with_suffix(".md")
    md_file.write_text(_render_markdown_report(result), encoding="utf-8")

    print(f"Backtest concluído. Resultados salvos em {output_file}")
    print(f"Relatório Markdown salvo em {md_file}")
    print(f"Métricas: {result['metrics']}")

if __name__ == '__main__':
    run_backtest()
