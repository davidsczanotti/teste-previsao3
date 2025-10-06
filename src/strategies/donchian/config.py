from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class DonchianConfig:
    ticker: str
    interval: str
    window_high: int
    window_low: int
    atr_period: int
    atr_mult: float
    use_ema: bool = True
    ema_period: int = 200
    allow_short: bool = False
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
            return int(s[:-1]) * 1440
        try:
            return int(s)
        except Exception:
            return 15


def _parse(data: Dict[str, Any], ticker: str, interval: str) -> DonchianConfig:
    if "best_params" in data:
        p = data["best_params"]
    else:
        p = data
    return DonchianConfig(
        ticker=ticker,
        interval=interval,
        window_high=int(p["window_high"]),
        window_low=int(p["window_low"]),
        atr_period=int(p["atr_period"]),
        atr_mult=float(p["atr_mult"]),
        use_ema=bool(p.get("use_ema", True)),
        ema_period=int(p.get("ema_period", 200)),
        allow_short=bool(p.get("allow_short", False)),
        lot_size=float(p.get("lot_size")) if p.get("lot_size") is not None else None,
        days=int(data.get("days")) if data.get("days") is not None else None,
        fee_rate=float(data.get("fee_rate", 0.001)),
    )


def load_active_config(ticker: str, interval: str, reports_dir: str = "reports") -> Optional[DonchianConfig]:
    path = Path(reports_dir) / "active" / f"DONCH_{ticker}_{interval}.json"
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
        raise ValueError("ticker/interval ausentes para salvar config ativa Donchian")
    out = Path(reports_dir) / "active"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"DONCH_{ticker}_{interval}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

