from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class TripleRsiConfig:
    ticker: str
    interval: str  # e.g., "5m", "15m"
    short_period: int
    med_period: int
    long_period: int
    buy_entry: int
    sell_entry: int
    buy_exit: int
    sell_exit: int
    invert: bool
    lot_size: Optional[float] = None
    days: Optional[int] = None

    @property
    def interval_minutes(self) -> int:
        s = self.interval.strip().lower()
        if s.endswith("m"):
            return int(s[:-1])
        if s.endswith("h"):
            return int(s[:-1]) * 60
        if s.endswith("d"):
            return int(s[:-1]) * 1440
        # fallback: try int minutes
        try:
            return int(s)
        except Exception:
            return 15

    def to_live_kwargs(self) -> Dict[str, Any]:
        # Não propaga lot_size para o live por padrão (normalmente é diferente do backtest)
        return dict(
            ticker=self.ticker,
            short_period=self.short_period,
            med_period=self.med_period,
            long_period=self.long_period,
            buy_entry=self.buy_entry,
            sell_entry=self.sell_entry,
            buy_exit=self.buy_exit,
            sell_exit=self.sell_exit,
            invert=self.invert,
            interval_minutes=self.interval_minutes,
        )

    def to_backtest_kwargs(self) -> Dict[str, Any]:
        return dict(
            short_period=self.short_period,
            med_period=self.med_period,
            long_period=self.long_period,
            buy_entry=self.buy_entry,
            sell_entry=self.sell_entry,
            buy_exit=self.buy_exit,
            sell_exit=self.sell_exit,
            invert=self.invert,
        )


def _parse_best_params(ticker: str, interval: str, data: Dict[str, Any]) -> TripleRsiConfig:
    # Accept both full record with best_params or direct params
    if "best_params" in data:
        p = data["best_params"]
    else:
        p = data
    return TripleRsiConfig(
        ticker=ticker,
        interval=interval,
        short_period=int(p["short_period"] if "short_period" in p else p["short"]),
        med_period=int(p["med_period"] if "med_period" in p else p["med"]),
        long_period=int(p["long_period"] if "long_period" in p else p["long"]),
        buy_entry=int(p["buy_entry"]),
        sell_entry=int(p["sell_entry"]),
        buy_exit=int(p["buy_exit"]),
        sell_exit=int(p["sell_exit"]),
        invert=bool(p["invert"]),
        lot_size=float(p.get("lot_size")) if p.get("lot_size") is not None else None,
        days=int(data.get("days")) if data.get("days") is not None else None,
    )


def load_active_config(ticker: str, interval: str, reports_dir: str = "reports") -> Optional[TripleRsiConfig]:
    path = Path(reports_dir) / "active" / f"{ticker}_{interval}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _parse_best_params(ticker, interval, data)
    except Exception:
        return None


def save_active_config(rec: Dict[str, Any], reports_dir: str = "reports") -> Path:
    ticker = rec.get("ticker") or rec.get("best_params", {}).get("ticker")
    interval = rec.get("interval") or rec.get("best_params", {}).get("interval")
    if not ticker or not interval:
        raise ValueError("ticker/interval ausentes no registro de configuração")
    out = Path(reports_dir) / "active"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{ticker}_{interval}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

