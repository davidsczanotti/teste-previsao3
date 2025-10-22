from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

# Fonte única de configuração da estratégia
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


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
    # Inside bar controls
    use_inside_bar: bool = True
    inside_bar_inclusive: bool = False
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
                f"[al_brooks.config] Aviso: chaves desconhecidas ignoradas na config: {unknown}"
            )
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


def save_active_config(config: AlBrooksConfig) -> Path:
    """Atualiza o arquivo único de configuração (config.json) da estratégia.

    - Preserva o bloco "optimize" existente (se houver).
    - Substitui os demais campos pelos da configuração fornecida.
    """
    existing: dict = {}
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    data = config.to_dict()
    # Preserva o bloco optimize
    if "optimize" in existing and isinstance(existing.get("optimize"), dict):
        data_out = {**data, "optimize": existing["optimize"]}
    else:
        data_out = data

    CONFIG_PATH.write_text(json.dumps(data_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return CONFIG_PATH


def load_active_config(ticker: str, interval: str) -> Optional[AlBrooksConfig]:
    """Carrega a configuração da estratégia a partir de config.json.

    Observação: ticker/interval passados são usados apenas para emitir aviso
    em caso de divergência do arquivo.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = AlBrooksConfig.from_dict(data)
        # Aviso opcional de divergência
        try:
            if cfg.ticker != ticker or cfg.interval != interval:
                print(
                    f"[al_brooks.config] Aviso: config.json define {cfg.ticker}@{cfg.interval}, "
                    f"mas foi solicitado {ticker}@{interval}. Usando valores do arquivo."
                )
        except Exception:
            pass
        return cfg
    except Exception:
        return None
