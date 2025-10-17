from __future__ import annotations

"""
Executa WFO sequencialmente comparando ma_trend.ma_type em ['ema','wma','hma'].
Mantém toda a configuração no JSON, salvando e restaurando no final.
Ideal para rodar no ambiente local com data.update_cache=true (para dados recentes).
"""

import json
from pathlib import Path
from importlib import import_module


CFG_PATH = Path("src/strategies/experimento/config/config_active.json")


def load_cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def save_cfg(cfg: dict) -> None:
    CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_wfo_for_ma(ma_type: str) -> None:
    cfg = load_cfg()
    # Ajustes finos pedidos: thresholds e trials
    cfg.setdefault("filters", {})
    cfg["filters"].setdefault("ma_trend", {"tf": "15m", "ma_type": "ema", "fast": 9, "slow": 20})
    cfg["filters"]["ma_trend"]["ma_type"] = ma_type
    cfg["filters"].setdefault("vwap_bias", {"tf": cfg.get("base_timeframe", "30m"), "mode": "long_only"})
    cfg["filters"].setdefault("atr_min", {"tf": cfg.get("base_timeframe", "30m"), "length": 14, "min_atr_frac": 0.0008})
    cfg["filters"].setdefault("volume_min", {"tf": cfg.get("base_timeframe", "30m"), "percentile": 0.5})
    cfg.setdefault("optimization", {})
    # Reduz trials para execução rápida no ambiente atual
    cfg["optimization"]["trials"] = min(int(cfg["optimization"].get("trials", 80)), 20)
    # Desativa update de cache neste fluxo (ambiente pode estar sem rede)
    cfg.setdefault("data", {})
    cfg["data"]["update_cache"] = False
    save_cfg(cfg)

    # Rodar WFO + relatório agregado
    wfo = import_module("src.strategies.experimento.scripts.walk_forward")
    wfo.main()
    report = import_module("src.strategies.experimento.scripts.report_wfo")
    report.main()


def main() -> None:
    original = load_cfg()
    try:
        for ma in ["ema", "wma", "hma"]:
            print(f"\n=== Running WFO for ma_type={ma} ===")
            run_wfo_for_ma(ma)
    finally:
        # Restore original config
        save_cfg(original)
        print("\nConfiguration restored to original.")


if __name__ == "__main__":
    main()
