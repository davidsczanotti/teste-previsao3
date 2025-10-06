import pandas as pd
import numpy as np
from ...binance_client import get_historical_klines
from .config import load_active_config
import sys


def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_triple_rsi(
    df,
    short_period=33,
    med_period=44,
    long_period=115,
    buy_entry=36,
    sell_entry=64,
    buy_exit=50,
    sell_exit=21,
    invert=True,
    initial_capital=10000,
    lot_size=1,
    day_trade=True,
):
    # Prepare data
    df = df.sort_values("Date").reset_index(drop=True)
    # Binance client retorna colunas em minúsculas (open/high/low/close/volume)
    closes = df["close"].astype(float)

    # Calculate RSIs
    rsi_short = calculate_rsi(closes, short_period)
    rsi_med = calculate_rsi(closes, med_period)
    rsi_long = calculate_rsi(closes, long_period)

    # Drop NaNs
    df = df.iloc[long_period:].copy()
    rsi_short = rsi_short.iloc[long_period:].values
    rsi_med = rsi_med.iloc[long_period:].values
    rsi_long = rsi_long.iloc[long_period:].values
    closes = closes.iloc[long_period:].values

    position = 0  # 1 long, -1 short, 0 none
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]
    realized_pnl = 0
    max_drawdown = 0
    peak = initial_capital
    sharpe_returns = []
    sharpe_vol = 0

    for i in range(len(df)):
        current_rsi_s = rsi_short[i]
        current_rsi_m = rsi_med[i]
        current_rsi_l = rsi_long[i]
        current_price = closes[i]  # Using the numpy array is faster
        current_time = df["Date"].iloc[i].time()

        # Day trade: no entry after 16:40, exit at 16:40 if open
        if day_trade and current_time >= pd.to_datetime("16:40").time():
            if position != 0:
                if position == 1:  # Long position
                    pnl = (current_price - entry_price) * lot_size
                    action = "SELL (Day End)"
                else:
                    pnl = (entry_price - current_price) * lot_size
                    action = "BUY_TO_COVER (Day End)"
                realized_pnl += pnl
                trades.append(
                    {"date": df["Date"].iloc[i], "action": action, "price": float(current_price), "pnl": pnl.item()}
                )
                position = 0

            # Update equity curve at the end of the day if a trade was closed
            equity_curve.append(float(initial_capital + realized_pnl))
            continue  # Skip entry after 16:40

        # Sinal único com base em helpers para reduzir duplicação
        def all_below(x):
            return (current_rsi_s < x) and (current_rsi_m < x) and (current_rsi_l < x)

        def all_above(x):
            return (current_rsi_s > x) and (current_rsi_m > x) and (current_rsi_l > x)

        def any_below(x):
            return (current_rsi_s < x) or (current_rsi_m < x) or (current_rsi_l < x)

        def any_above(x):
            return (current_rsi_s > x) or (current_rsi_m > x) or (current_rsi_l > x)

        if invert:
            buy_signal = all_above(sell_entry)
            sell_signal = all_below(buy_entry)
            long_exit = any_below(sell_exit)
            short_exit = any_above(buy_exit)
        else:
            buy_signal = all_below(buy_entry)
            sell_signal = all_above(sell_entry)
            long_exit = any_above(buy_exit)
            short_exit = any_below(sell_exit)

        # Entry logic (only if not day end)
        if position == 0:
            if buy_signal:
                position = 1
                entry_price = current_price
                trades.append({"date": df["Date"].iloc[i], "action": "BUY", "price": float(current_price)})
            elif sell_signal:
                position = -1
                entry_price = current_price
                trades.append({"date": df["Date"].iloc[i], "action": "SELL", "price": float(current_price)})

        # Exit logic
        elif position == 1:  # Long
            if long_exit:
                pnl = (current_price - entry_price) * lot_size
                realized_pnl += pnl
                trades.append(
                    {"date": df["Date"].iloc[i], "action": "SELL", "price": float(current_price), "pnl": pnl.item()}
                )
                position = 0
                entry_price = 0
        elif position == -1:  # Short
            if short_exit:
                pnl = (entry_price - current_price) * lot_size
                realized_pnl += pnl
                trades.append(
                    {
                        "date": df["Date"].iloc[i],
                        "action": "BUY_TO_COVER",
                        "price": float(current_price),
                        "pnl": pnl.item(),
                    }
                )
                position = 0
                entry_price = 0

        # Update equity curve
        if position == 1:  # Long
            unrealized_pnl = (current_price - entry_price) * lot_size
        elif position == -1:  # Short
            unrealized_pnl = (entry_price - current_price) * lot_size
        else:  # No position
            unrealized_pnl = 0

        equity_curve.append(float(initial_capital + realized_pnl + unrealized_pnl))

    # Final position close if open
    if position != 0:
        if position == 1:
            pnl = (closes[-1] - entry_price) * lot_size
            action = "SELL"
        else:
            pnl = (entry_price - closes[-1]) * lot_size
            action = "BUY_TO_COVER"
        realized_pnl += pnl
        trades.append({"date": df["Date"].iloc[-1], "action": action, "price": float(closes[-1]), "pnl": pnl.item()})
        equity_curve[-1] = initial_capital + realized_pnl  # Update final equity

    total_trades = len([t for t in trades if "pnl" in t])
    realized_pnl = float(realized_pnl)  # Ensure it's a scalar float
    winning_trades = len([t for t in trades if "pnl" in t and t["pnl"] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = realized_pnl
    return_pct = (total_pnl / initial_capital) * 100
    equity = initial_capital + total_pnl

    # Advanced metrics
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0  # Annualized
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = ((equity_curve - running_max) / running_max * 100).min() if len(equity_curve) > 0 else 0

    # Use o ticker dinamicamente no print
    print(
        f"Backtest Results for {df.attrs.get('ticker', 'Unknown')} (Triple RSI {short_period},{med_period},{long_period}):"
    )
    print(f"Invert signals: {invert}, Lot size: {lot_size}, Day trade: {day_trade}")
    print(f"Total P&L: $ {total_pnl:.2f} ({return_pct:.2f}%)")
    print(f"Number of trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Max drawdown: {drawdown:.2f}%")
    print(f"Final equity: $ {equity:.2f}")
    print("\nTrades:")
    for trade in trades:
        print(trade)

    # Plotting
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Equity curve
    ax1.plot(df["Date"], equity_curve[1:], label="Equity Curve", color="blue")  # Skip initial capital for alignment
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True)

    # RSI plot with price subplot
    ax2.plot(df["Date"], rsi_short, label=f"RSI {short_period}", color="green")
    ax2.plot(df["Date"], rsi_med, label=f"RSI {med_period}", color="orange")
    ax2.plot(df["Date"], rsi_long, label=f"RSI {long_period}", color="red")

    # Threshold lines
    ax2.axhline(y=buy_entry, color="g", linestyle="--", label="Buy Entry" if not invert else "Sell Entry")
    ax2.axhline(y=sell_entry, color="r", linestyle="--", label="Sell Entry" if not invert else "Buy Entry")
    ax2.axhline(y=buy_exit, color="r", linestyle=":", label="Long Exit" if not invert else "Short Exit")
    ax2.axhline(y=sell_exit, color="g", linestyle=":", label="Short Exit" if not invert else "Long Exit")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.grid(True)

    # Price plot on a secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["Date"], closes, label="Close", color="black")
    ax2_twin.set_ylabel("Price ($)")

    ax2.set_title("Triple RSI with Signals and Price")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(f'triple_rsi_backtest_{"inverted" if invert else "normal"}.png')
    plt.show()

    return trades, total_pnl


if __name__ == "__main__":
    from datetime import datetime, timedelta, UTC

    # Parâmetros para buscar dados da Binance (ex: BTCUSDT, ETHUSDT)
    ticker = "BTCUSDT"
    interval = "5m"
    # Se existir configuração ativa, usar a mesma janela de dias da otimização
    cfg = load_active_config(ticker, interval)
    cfg_days = cfg.days if cfg and cfg.days else 60
    # Calcula a data de início de forma explícita (mesma janela do estudo quando possível)
    start_date_dt = datetime.now(UTC) - timedelta(days=cfg_days)
    start_date = start_date_dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Buscando dados de {interval} para {ticker} da Binance por {cfg_days} dias...")
    df = get_historical_klines(ticker, interval, start_date)
    df.attrs["ticker"] = ticker  # Armazena o ticker para uso posterior

    if df.empty:
        print("Não foi possível obter dados da Binance. Encerrando.")
        sys.exit(1)

    print(f"Dados carregados. Total de {len(df)} registros.")

    # Se existir configuração ativa do otimizador para este ticker/intervalo, use-a
    if cfg:
        print(f"Usando configuração ativa de reports/active/{ticker}_{interval}.json")
        print(
            "Parâmetros carregados: short={s}, med={m}, long={l}, be={be}, se={se}, bx={bx}, sx={sx}, invert={inv}".format(
                s=cfg.short_period,
                m=cfg.med_period,
                l=cfg.long_period,
                be=cfg.buy_entry,
                se=cfg.sell_entry,
                bx=cfg.buy_exit,
                sx=cfg.sell_exit,
                inv=cfg.invert,
            )
        )
        trades, pnl = backtest_triple_rsi(
            df.copy(),
            **cfg.to_backtest_kwargs(),
            lot_size=cfg.lot_size if cfg.lot_size is not None else 0.1,
            day_trade=False,
        )
        print(f"\nResultado final com config ativa: P&L ${pnl:.2f}")
    else:
        print("\nNenhuma config ativa encontrada. Rodando backtests padrão...")
        trades_normal, pnl_normal = backtest_triple_rsi(df.copy(), lot_size=0.1, day_trade=False)
        print(f"Normal: P&L ${pnl_normal:.2f}")
        trades_invert, pnl_invert = backtest_triple_rsi(df.copy(), invert=True, lot_size=0.1, day_trade=False)
        print(f"Invertido: P&L ${pnl_invert:.2f}")
