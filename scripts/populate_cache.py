from __future__ import annotations

"""
Populate o cache local (`data/klines_cache.db`).

Uso (a partir da raiz do repo):

  poetry run python -m scripts.populate_cache BTCUSDT

- O símbolo é obrigatório no CLI; timeframe/start/dias são lidos do `config.json`
  (data.*) e podem ser sobrescritos pelos argumentos `interval`/`--start`/`--days`.
- Para popular todos os símbolos declarados no config, rode sem argumentos.
- Requer acesso à rede (não usar com `BINANCE_OFFLINE=1`).
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Iterable, List, Dict, Any

from src.binance_client import get_historical_klines


DEFAULT_CFG_PATH = Path("src/strategies/exper_corr_pos/config.json")


@dataclass
class PopulateTask:
    symbol: str
    interval: str
    start: str | None
    end: str | None
    days: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Populate local klines cache (SQLite)")
    ap.add_argument("symbol", nargs="?", help="Symbol, e.g., BTCUSDT")
    ap.add_argument("interval", nargs="?", help="Interval override. Quando omitido, usa tempo do config")
    ap.add_argument("--start", default=None, help="Start datetime (YYYY-MM-DD HH:MM:SS, UTC)")
    ap.add_argument("--end", default=None, help="End datetime (YYYY-MM-DD HH:MM:SS, UTC)")
    ap.add_argument("--days", type=int, default=365, help="If --start not given, go back this many days")
    ap.add_argument(
        "--config",
        default=None,
        help=(
            "Optional strategy config (JSON). Quando usado sem symbol/interval, popula todos os pares"
            " definidos em data.*"
        ),
    )
    return ap.parse_args()


def _tasks_from_config(cfg_path: Path, fallback_days: int) -> List[PopulateTask]:
    cfg = json.loads(cfg_path.read_text())
    data_cfg = cfg.get("data", {})
    timeframe = data_cfg.get("timeframe")
    base_symbol = data_cfg.get("base_symbol")
    confirm_symbol = data_cfg.get("confirm_symbol")
    lookback_days = int(data_cfg.get("lookback_days", fallback_days))
    start = data_cfg.get("start")
    end = data_cfg.get("end")
    confirm_timeframe = data_cfg.get("confirm_timeframe", timeframe)

    tasks: List[PopulateTask] = []

    def _push(symbol: str | None, interval: str | None, start_override: str | None = None, days_override: int | None = None):
        if not symbol or not interval:
            return
        tasks.append(
            PopulateTask(
                symbol=symbol,
                interval=interval,
                start=start_override if start_override is not None else start,
                end=end,
                days=int(days_override if days_override is not None else lookback_days),
            )
        )

    _push(base_symbol, timeframe)
    _push(confirm_symbol, confirm_timeframe)

    extras = data_cfg.get("extra_symbols") or []
    if isinstance(extras, list):
        for entry in extras:
            if not isinstance(entry, dict):
                continue
            _push(
                entry.get("symbol"),
                entry.get("timeframe", timeframe),
                entry.get("start"),
                entry.get("lookback_days") or entry.get("days"),
            )

    if not tasks:
        raise SystemExit(
            f"Nenhum par encontrado em {cfg_path}. Defina ao menos data.base_symbol/timeframe ou informe symbol/interval."
        )
    return tasks


def _iter_tasks(args: argparse.Namespace) -> Iterable[PopulateTask]:
    cfg_path = Path(args.config) if args.config else DEFAULT_CFG_PATH
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    if args.symbol:
        yield _task_from_symbol(cfg, args.symbol, args.interval, args.start, args.end, args.days)
        return

    if not cfg:
        raise SystemExit(
            "Nenhum symbol fornecido e config padrão não encontrado. Informe symbol ou passe --config caminho_do_json."
        )
    yield from _tasks_from_config(cfg_path, args.days)


def _task_from_symbol(
    cfg: Dict[str, Any],
    symbol: str,
    interval_override: str | None,
    start_override: str | None,
    end_override: str | None,
    days_override: int,
) -> PopulateTask:
    data_cfg = cfg.get("data", {}) if cfg else {}
    timeframe_default = data_cfg.get("timeframe")
    start_default = data_cfg.get("start")
    end_default = data_cfg.get("end")
    lookback_default = int(data_cfg.get("lookback_days", days_override))

    matches: List[Dict[str, Any]] = []

    if symbol == data_cfg.get("base_symbol"):
        matches.append({
            "symbol": symbol,
            "timeframe": timeframe_default,
            "start": start_default,
            "end": end_default,
            "days": lookback_default,
        })
    if symbol == data_cfg.get("confirm_symbol"):
        matches.append({
            "symbol": symbol,
            "timeframe": data_cfg.get("confirm_timeframe", timeframe_default),
            "start": start_default,
            "end": end_default,
            "days": lookback_default,
        })

    extras = data_cfg.get("extra_symbols") or []
    if isinstance(extras, list):
        for entry in extras:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("symbol")) == symbol:
                matches.append(
                    {
                        "symbol": symbol,
                        "timeframe": entry.get("timeframe", timeframe_default),
                        "start": entry.get("start", start_default),
                        "end": entry.get("end", end_default),
                        "days": int(entry.get("lookback_days", entry.get("days", lookback_default))),
                    }
                )

    if not matches:
        if interval_override:
            return PopulateTask(symbol=symbol, interval=interval_override, start=start_override, end=end_override, days=days_override)
        raise SystemExit(
            f"Símbolo {symbol} não encontrado no config e nenhum intervalo foi informado."
        )

    # usa a primeira correspondência (prioridade: base -> confirm -> extra)
    chose = matches[0]
    interval = interval_override or chose.get("timeframe")
    if not interval:
        raise SystemExit(f"Intervalo não encontrado para {symbol}. Defina em data.timeframe ou extras[].timeframe.")

    start = start_override if start_override is not None else chose.get("start")
    end = end_override if end_override is not None else chose.get("end")
    days = int(chose.get("days", days_override))
    if start_override is not None:
        days = int(days_override)

    return PopulateTask(symbol=symbol, interval=str(interval), start=start, end=end, days=days)


def main() -> None:
    args = parse_args()
    if os.environ.get("BINANCE_OFFLINE", "0") == "1":
        raise SystemExit(
            "BINANCE_OFFLINE=1 detectado. Para popular o cache, remova essa variável (ou defina 0) e rode novamente."
        )

    tasks = list(_iter_tasks(args))
    for task in tasks:
        if task.start is None:
            start_dt = datetime.now(UTC) - timedelta(days=max(1, int(task.days)))
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_str = task.start

        print(
            f"[populate] Baixando {task.symbol} @ {task.interval} desde {start_str} até {task.end or '(agora)'}..."
        )
        df = get_historical_klines(task.symbol, task.interval, start_str, task.end)
        count = 0 if df is None else len(df)
        if count == 0:
            raise SystemExit("Nenhum dado retornado. Verifique símbolo/intervalo/tempo e conectividade.")
        first = df.iloc[0]["Date"].strftime("%Y-%m-%d %H:%M:%S")
        last = df.iloc[-1]["Date"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"[populate] OK: {count} candles persistidos em data/klines_cache.db ({first} -> {last})")


if __name__ == "__main__":
    main()
