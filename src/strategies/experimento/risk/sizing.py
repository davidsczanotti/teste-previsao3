from __future__ import annotations


def fixed_fraction(capital: float, fraction: float, price: float) -> float:
    risk_amount = capital * max(0.0, min(1.0, fraction))
    if price <= 0:
        return 0.0
    qty = risk_amount / price
    return max(0.0, qty)

