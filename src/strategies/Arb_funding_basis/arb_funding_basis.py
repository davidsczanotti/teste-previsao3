from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

import requests

from src.binance_client import client, get_historical_klines


def get_funding_rate_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Busca o histórico de funding rate da Binance para um símbolo."""
    all_funding_rates = []
    limit = 1000
    current_start = start_ms

    while current_start < end_ms:
        try:
            funding_rates = client.futures_funding_rate(symbol=symbol, startTime=current_start, limit=limit)
            if not funding_rates:
                break
            all_funding_rates.extend(funding_rates)
            current_start = funding_rates[-1]["fundingTime"] + 1
            time.sleep(0.2)  # Evitar rate limiting
        except Exception as e:
            print(f"Erro ao buscar funding rates: {e}")
            break

    if not all_funding_rates:
        return pd.DataFrame(columns=["Date", "fundingRate"])

    df = pd.DataFrame(all_funding_rates)
    df["Date"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df[["Date", "fundingRate"]]


class ArbFundingBasisStrategy:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        entry_threshold: float = 0.0001,  # Entrar se funding rate > 0.01%
        exit_threshold: float = 0.0000,  # Sair se funding rate <= 0.00%
        leverage: float = 2.0,
        usdc_depeg_threshold: float = 0.01,  # Sair se USDC desviar 1%
        liquidation_margin_pct: float = 0.8,  # % da margem antes da liquidação
        spot_fee_rate: float = 0.001,  # 0.1% taxa taker spot
        futures_fee_rate: float = 0.0005,  # 0.05% taxa taker futuros
    ):
        self.symbol = symbol
        self.entry_threshold = entry_threshold  # Funding rate mínimo para entrada
        self.exit_threshold = exit_threshold  # Funding rate para saída
        self.leverage = leverage
        self.usdc_depeg_threshold = usdc_depeg_threshold
        self.liquidation_margin_pct = liquidation_margin_pct
        self.spot_fee_rate = spot_fee_rate
        self.futures_fee_rate = futures_fee_rate

    def _prepare_data(self, start: str, end: Optional[str]) -> pd.DataFrame:
        """Busca e prepara os dados para o backtest."""
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
        end_ms = int(end_ts.timestamp() * 1000)
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)

        print("Buscando dados de funding, preço spot e USDC...")
        funding_df = get_funding_rate_history(self.symbol, start_ms, end_ms)
        spot_df = get_historical_klines(self.symbol, "1h", start, end)
        usdc_df = get_historical_klines("USDCUSDT", "1h", start, end)

        spot_df.set_index("Date", inplace=True)
        usdc_df.set_index("Date", inplace=True)
        funding_df.set_index("Date", inplace=True)

        df = spot_df[["close"]].join(usdc_df[["close"]], lsuffix="_spot", rsuffix="_usdc").join(funding_df)
        df.rename(columns={"close_spot": "spot_price", "close_usdc": "usdc_price"}, inplace=True)
        df["is_funding_time"] = df["fundingRate"].notna()
        return df.ffill().dropna(subset=["spot_price", "usdc_price", "fundingRate"])

    def backtest(
        self, start: str, end: Optional[str] = None, initial_capital: float = 1000.0, use_bnb_discount: bool = True
    ):
        """Executa um backtest da estratégia de arbitragem de funding."""
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
        end_ms = int(end_ts.timestamp() * 1000)
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)

        # Ajusta taxas se o desconto de BNB for usado
        if use_bnb_discount:
            self.spot_fee_rate *= 0.75  # Desconto de 25% na taxa spot
            self.futures_fee_rate *= 0.90  # Desconto de 10% na taxa de futuros (para USDT-M)

        df = self._prepare_data(start, end)
        if df.empty:
            print("Dados insuficientes para o backtest.")
            return [], pd.DataFrame()

        # 3. Simulação
        capital = initial_capital
        position_size = 0.0
        spot_entry_price = 0.0
        perp_entry_price = 0.0
        notional_value_at_entry = 0.0

        in_position = False
        trades: List[Dict] = []
        equity_curve = []
        funding_accrued = 0.0

        print("Iniciando simulação...")
        for index, row in df.iterrows():
            current_spot_price = row["spot_price"]
            current_funding_rate = row["fundingRate"]
            current_usdc_price = row["usdc_price"]
            is_funding_time = bool(row.get("is_funding_time", False))
            funding_payment = 0.0

            if in_position and is_funding_time:
                funding_payment = current_funding_rate * position_size * current_spot_price
                capital += funding_payment
                funding_accrued += funding_payment

            # Lógica de Saída
            if in_position:
                # Risco de liquidação
                margin_used = notional_value_at_entry / self.leverage
                liquidation_price = perp_entry_price * (1 + (1 / self.leverage) * self.liquidation_margin_pct)

                exit_signal = current_funding_rate <= self.exit_threshold

                # Risco de De-peg do USDC
                usdc_depeg = abs(current_usdc_price - 1.0) > self.usdc_depeg_threshold

                if exit_signal or current_spot_price >= liquidation_price or usdc_depeg:
                    reason = (
                        "Funding Invertido"
                        if exit_signal
                        else ("Risco de Liquidação" if not usdc_depeg else "USDC De-peg")
                    )

                    # Deduz taxas de fechamento com possibilidade de desconto BNB em futuros
                    spot_exit_fee = current_spot_price * position_size * self.spot_fee_rate
                    perp_exit_fee = current_spot_price * position_size * self.futures_fee_rate
                    capital -= spot_exit_fee + perp_exit_fee

                    trades.append(
                        {
                            "date": index,
                            "type": "exit",
                            "price": current_spot_price,
                            "pnl_hedge": 0.0,  # P&L do hedge é zero por definição da estratégia
                            "funding": funding_payment,
                            "fees": spot_exit_fee + perp_exit_fee,
                            "reason": reason,
                        }
                    )
                    in_position = False
                    position_size = 0.0
                    spot_entry_price = 0.0
                    perp_entry_price = 0.0
                    notional_value_at_entry = 0.0
            elif not in_position and current_funding_rate > self.entry_threshold:
                # O valor nocional da posição é baseado no capital ATUAL da conta
                notional_value_at_entry = capital * self.leverage
                position_size = notional_value_at_entry / current_spot_price

                # Deduz taxas de abertura
                spot_entry_fee = notional_value_at_entry * self.spot_fee_rate
                perp_entry_fee = notional_value_at_entry * self.futures_fee_rate
                capital -= spot_entry_fee + perp_entry_fee

                in_position = True
                spot_entry_price = current_spot_price
                perp_entry_price = current_spot_price

                trades.append(
                    {
                        "date": index,
                        "type": "entry",
                        "price": spot_entry_price,
                        "size": position_size,
                        "fees": spot_entry_fee + perp_entry_fee,
                    }
                )

            # A curva de equity reflete o capital da conta a cada passo
            equity_curve.append({"Date": index, "equity": capital})

        equity_df = pd.DataFrame(equity_curve).set_index("Date") if equity_curve else pd.DataFrame()
        if equity_df.empty:
            print("Nenhum trade foi executado. Não há resultados para exibir.")
            return [], pd.DataFrame()

        total_pnl = equity_df["equity"].iloc[-1] - initial_capital if not equity_df.empty else 0
        returns = equity_df["equity"].pct_change().dropna()

        num_trades = len([t for t in trades if t["type"] == "entry"])
        pnl_hedge_trades = [t["pnl_hedge"] for t in trades if t.get("pnl_hedge")]
        funding_collected = funding_accrued
        total_fees = sum(t["fees"] for t in trades if t.get("fees"))

        print("\n--- RESULTADO DO BACKTEST ---")
        print(f"Período: {start} a {end_ts.strftime('%Y-%m-%d')}")
        print(f"Capital Inicial: ${initial_capital:,.2f}")
        print(f"Capital Final: ${equity_df['equity'].iloc[-1]:,.2f}")
        print(f"P&L Total: ${total_pnl:,.2f} ({(total_pnl/initial_capital):.2%})")
        print("-" * 30)
        print(f"Número de Trades: {num_trades}")
        print(f"Total de Funding Coletado: ${funding_collected:,.2f}")
        print(f"Total de Taxas Pagas: ${total_fees:,.2f}")
        if pnl_hedge_trades:
            print(f"P&L Médio do Hedge (Spot+Perp): ${np.mean(pnl_hedge_trades):,.2f}")
        print(f"Resultado Líquido (Funding - Taxas): ${(funding_collected - total_fees):,.2f}")
        # Plotar curva de equity
        try:
            import matplotlib.pyplot as plt

            equity_df["equity"].plot(figsize=(12, 6), title=f"Curva de Capital - {self.symbol} Funding Arbitrage")
            plt.ylabel("Capital (USDC)")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("\nMatplotlib não instalado. Curva de equity não será plotada.")

        return trades, equity_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executa estratégia de arbitragem Spot-Perp")
    parser.add_argument("--symbol", default="BTCUSDT", help="Par de negociação (default: BTCUSDT)")
    parser.add_argument("--start", dest="start", default="2023-01-01", help="Data de início (UTC)")
    parser.add_argument("--end", dest="end", default=None, help="Data de fim (UTC), default: hoje")
    parser.add_argument(
        "--leverage", dest="leverage", type=float, default=2.0, help="Alavancagem máxima (default: 2.0x)"
    )
    parser.add_argument("--capital", type=float, default=1000.0, help="Capital inicial (default: 1000.0)")
    parser.add_argument("--entry", type=float, default=0.0001, help="Funding rate mínimo para entrar (default: 0.0001)")
    parser.add_argument("--exit", type=float, default=0.0000, help="Funding rate para sair (default: 0.0000)")
    parser.add_argument(
        "--no-bnb", action="store_false", dest="use_bnb", help="Não usar desconto de 25% com BNB nas taxas spot"
    )

    args = parser.parse_args()

    strategy = ArbFundingBasisStrategy(
        symbol=args.symbol,
        leverage=args.leverage,
        entry_threshold=args.entry,
        exit_threshold=args.exit,
    )

    trades, equity = strategy.backtest(
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        use_bnb_discount=args.use_bnb,
    )

    if trades:
        print(f"\nÚltimos 5 eventos de trade:")
        for trade in trades[-5:]:
            print(trade)
