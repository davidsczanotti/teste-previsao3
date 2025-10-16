from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json


def save_active_config(strategy_name: str, symbol: str, timeframe: str, best_params: Dict[str, Any], reports_dir: str = "reports") -> Path:
    out = Path(reports_dir) / "active"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{strategy_name}_{symbol}_{timeframe}.json"
    rec = {
        "strategy": strategy_name,
        "symbol": symbol,
        "interval": timeframe,
        "best_params": best_params,
    }
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

