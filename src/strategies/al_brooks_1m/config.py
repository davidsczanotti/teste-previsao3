from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

# Relatórios por estratégia: pasta local à estratégia
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
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
    adx_period: int = 14
    adx_threshold: float = 22.0
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_trail_multiplier: float = 0.5
    htf_lookback: int = 20
    use_htf_bias: bool = True
    pullback_lookback: int = 10
    min_trades_per_window: int = 15
    min_atr: float = 0.0
    # Custos de execução
    taker_fee_pct: float = 0.0004  # 4 bps
    slippage_pct: float = 0.0005   # 5 bps

    @classmethod
    def from_dict(cls, data: dict) -> AlBrooksConfig:
        """Cria uma configuração tolerante a chaves extras.

        - Ignora chaves desconhecidas com aviso simples (stdout).
        - Mantém compatibilidade com JSONs antigos/novos.
        """
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        unknown = sorted(k for k in data.keys() if k not in valid_keys)
        if unknown:
            print(
                f"[al_brooks_1m.config] Aviso: chaves desconhecidas ignoradas na config ativa: {unknown}"
            )
        return cls(**filtered)

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
