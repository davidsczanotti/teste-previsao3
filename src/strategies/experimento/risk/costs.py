from __future__ import annotations


def fee_amount(notional: float, fee_bp: float) -> float:
    return abs(notional) * (fee_bp / 10_000.0)


def apply_slippage(price: float, side: str, slippage_ticks: float, tick_size: float) -> float:
    slip = slippage_ticks * tick_size
    if side == "buy":
        return price + slip
    else:
        return price - slip

