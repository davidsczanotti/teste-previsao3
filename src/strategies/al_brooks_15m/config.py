from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPORTS_DIR = Path("reports")
ACTIVE_CONFIG_DIR = REPORTS_DIR / "active"


@dataclass
class AlBrooksConfig:
    """Estrutura para os parâmetros da estratégia Al Brooks."""

    ticker: str
    interval: str
    days: int
    lot_size: float
    ema_fast_period: int
    ema_medium_period: int
    ema_slow_period: int
    risk_reward_ratio: float
    max_avg_deviation_pct: float

    @classmethod
    def from_dict(cls, data: dict) -> AlBrooksConfig:
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def save_active_config(config: AlBrooksConfig) -> Path:
    """Salva a configuração como a 'ativa' para o par/intervalo."""
    ACTIVE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Nome do arquivo de configuração ativa
    filename = f"ALBROOKS_{config.ticker}_{config.interval}.json"
    filepath = ACTIVE_CONFIG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    return filepath


def load_active_config(ticker: str, interval: str) -> Optional[AlBrooksConfig]:
    """Carrega a configuração ativa para o par/intervalo, se existir."""
    filename = f"ALBROOKS_{ticker}_{interval}.json"
    filepath = ACTIVE_CONFIG_DIR / filename

    if not filepath.exists():
        return None

    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return AlBrooksConfig.from_dict(data)
    except (json.JSONDecodeError, TypeError):
        return None
