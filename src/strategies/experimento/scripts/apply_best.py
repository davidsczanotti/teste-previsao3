from __future__ import annotations

"""
Aplica automaticamente ao config_active.json os "best params" do último run de Optimize.

O que atualiza:
- Base EMA (30m): fast/slow
- ATR (30m): length
- Filtro de volatilidade: filters.atr_min.min_atr_frac
- Stops: risk.stop.mult e risk.trailing.mult
- Tendência 15m: se existir filters.ma_trend, ajusta fast/slow com trend_ema_* (mantém ma_type atual);
  caso contrário, se existir filters.trend_tf (legado), ajusta ema_fast/ema_slow lá.

Também salva backup do config antes/depois em artifacts/<run_id>/.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any


CFG_PATH = Path("src/strategies/experimento/config/config_active.json")


def load_cfg() -> Dict[str, Any]:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def save_cfg(cfg: Dict[str, Any]) -> None:
    CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def get_last_opt_run_id(db_path: str) -> str | None:
    with sqlite3.connect(db_path) as cx:
        row = cx.execute(
            "SELECT run_id FROM runs WHERE finished_at IS NOT NULL AND run_id LIKE 'opt-%' ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None


def load_best_params(db_path: str, run_id: str) -> Dict[str, Any]:
    with sqlite3.connect(db_path) as cx:
        rows = cx.execute(
            "SELECT key, value FROM params WHERE run_id=? AND key LIKE 'best.%'",
            (run_id,),
        ).fetchall()
    out: Dict[str, Any] = {}
    for k, v in rows:
        key = k.split("best.", 1)[-1]
        try:
            out[key] = json.loads(v)
        except Exception:
            try:
                out[key] = float(v) if v.replace('.', '', 1).isdigit() else v
            except Exception:
                out[key] = v
    return out


def apply_best_to_config(cfg: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
    base_tf = cfg.get("base_timeframe", "30m")
    # Base EMA
    if cfg.get("indicators"):
        cfg["indicators"][0]["params"]["fast"] = int(best.get("ema_fast", cfg["indicators"][0]["params"].get("fast", 9)))
        cfg["indicators"][0]["params"]["slow"] = int(best.get("ema_slow", cfg["indicators"][0]["params"].get("slow", 21)))

    # ATR length (30m)
    for ind in cfg.get("indicators", []):
        if ind.get("name") == "atr" and ind.get("tf") == base_tf:
            ind.setdefault("params", {})["length"] = int(best.get("atr_len", ind.get("params", {}).get("length", 14)))
            break

    # Filtro de volatilidade
    cfg.setdefault("filters", {}).setdefault("atr_min", {"tf": base_tf, "length": 14, "min_atr_frac": 0.001})
    cfg["filters"]["atr_min"]["min_atr_frac"] = float(best.get("atr_min_frac", cfg["filters"]["atr_min"].get("min_atr_frac", 0.001)))

    # Stops
    cfg.setdefault("risk", {}).setdefault("stop", {"type": "atr", "mult": 2.0})
    cfg.setdefault("risk", {}).setdefault("trailing", {"type": "atr", "mult": 1.5})
    if "stop_mult" in best:
        cfg["risk"]["stop"]["mult"] = float(best["stop_mult"])
    if "trailing_mult" in best:
        cfg["risk"]["trailing"]["mult"] = float(best["trailing_mult"])

    # Tendência (15m): ma_trend ou legado trend_tf
    if "ma_trend" in cfg.get("filters", {}):
        cfg["filters"]["ma_trend"]["fast"] = int(best.get("trend_ema_fast", cfg["filters"]["ma_trend"].get("fast", 9)))
        cfg["filters"]["ma_trend"]["slow"] = int(best.get("trend_ema_slow", cfg["filters"]["ma_trend"].get("slow", 20)))
    elif "trend_tf" in cfg.get("filters", {}):
        cfg["filters"]["trend_tf"]["ema_fast"] = int(best.get("trend_ema_fast", cfg["filters"]["trend_tf"].get("ema_fast", 50)))
        cfg["filters"]["trend_tf"]["ema_slow"] = int(best.get("trend_ema_slow", cfg["filters"]["trend_tf"].get("ema_slow", 200)))

    return cfg


def main() -> None:
    cfg = load_cfg()
    db = cfg["storage"]["results_db"]
    run_id = get_last_opt_run_id(db)
    if not run_id:
        print("Nenhum run 'opt-*' encontrado no DB.")
        return
    best = load_best_params(db, run_id)
    if not best:
        print(f"Run {run_id} não possui 'best.*' em params.")
        return

    # Backup + apply
    artifacts = Path(cfg["storage"]["artifacts_dir"]) / run_id
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "config_before.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cfg2 = apply_best_to_config(cfg, best)
    save_cfg(cfg2)
    (artifacts / "config_after.json").write_text(json.dumps(cfg2, indent=2), encoding="utf-8")

    print("Best params aplicados ao config_active.json a partir de:", run_id)
    print(json.dumps(best, indent=2))
    print("Backups:", artifacts / "config_before.json", ",", artifacts / "config_after.json")


if __name__ == "__main__":
    main()

