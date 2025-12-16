import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from pathlib import Path

from src.utils.data_loader import load_data
from src.utils.metrics import calculate_metrics, calculate_sharpe_ratio

# Importação dos módulos refatorados
try:
    from .indicators import add_indicators
    from .signals import apply_signals
except ImportError:
    # Fallback para execução direta (se necessário)
    from indicators import add_indicators
    from signals import apply_signals


def backtest_ema_only(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """
    Executa backtest.
    Agora atua puramente como orquestrador de loop de eventos.
    A lógica de cálculo e decisão reside em indicators.py e signals.py.
    """
    # 1. Preparação de Dados (Indicadores + Sinais)
    df = add_indicators(df, config)
    df = apply_signals(df, config)

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
