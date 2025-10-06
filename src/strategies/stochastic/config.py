from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class StochConfig:
    ticker: str
    interval: str  # e.g., "5m", "15m"
    k_period: int
    oversold: float
    overbought: float
    d_period: int = 3
    use_kd_cross: bool = True
    ema_period: int | None = None
    confirm_bars: int = 0
    cooldown_bars: int = 0
    min_hold_bars: int = 0
    use_adx: bool = False
    adx_period: int = 14
    min_adx: float = 20.0
    lot_size: Optional[float] = None
    days: Optional[int] = None
    fee_rate: float = 0.001

    @property
    def interval_minutes(self) -> int:
        s = self.interval.strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        if s.endswith("d"):
            return int(s[:-1]) * 24 * 60
        try:
            return int(s)
        except Exception:
            return 5


def _parse(data: Dict[str, Any], ticker: str, interval: str) -> StochConfig:
    if "best_params" in data:
        p = data["best_params"]
    else:
        p = data
    return StochConfig(
        ticker=ticker,
        interval=interval,
        k_period=int(p["k_period"] if "k_period" in p else p["k"]),
        oversold=float(p["oversold"]),
        overbought=float(p["overbought"]),
        d_period=int(p.get("d_period", 3)),
        use_kd_cross=bool(p.get("use_kd_cross", True)),
        ema_period=(int(p["ema_period"]) if p.get("ema_period") not in (None, "", 0) else None),
        confirm_bars=int(p.get("confirm_bars", 0)),
        cooldown_bars=int(p.get("cooldown_bars", 0)),
        min_hold_bars=int(p.get("min_hold_bars", 0)),
        use_adx=bool(p.get("enable_adx", p.get("use_adx", False))),
        adx_period=int(p.get("adx_period", 14)),
        min_adx=float(p.get("min_adx", 20)),
        lot_size=float(p.get("lot_size")) if p.get("lot_size") is not None else None,
        days=int(data.get("days")) if data.get("days") is not None else None,
        fee_rate=float(data.get("fee_rate")) if data.get("fee_rate") is not None else 0.001,
    )


def load_active_config(ticker: str, interval: str, reports_dir: str = "reports") -> Optional[StochConfig]:
    path = Path(reports_dir) / "active" / f"STOCH_{ticker}_{interval}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _parse(data, ticker, interval)
    except Exception:
        return None


def save_active_config(rec: Dict[str, Any], reports_dir: str = "reports") -> Path:
    ticker = rec.get("ticker") or rec.get("best_params", {}).get("ticker")
    interval = rec.get("interval") or rec.get("best_params", {}).get("interval")
    if not ticker or not interval:
        raise ValueError("ticker/interval ausentes para salvar config ativa do Estocástico")
    out = Path(reports_dir) / "active"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"STOCH_{ticker}_{interval}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
