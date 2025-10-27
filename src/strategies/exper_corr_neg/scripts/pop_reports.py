from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("src/strategies/exper_corr_neg")
POP_ROOT = ROOT / "reports" / "train" / "pop"
TRAIN_ROOT = ROOT / "reports" / "train"


def latest_round_dir(run_dir: Path) -> Optional[Path]:
    rounds = sorted([p for p in run_dir.glob("round_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[-1]))
    return rounds[-1] if rounds else None


def eval_every_from_cfg(cfg_path: Path) -> int:
    try:
        cfg = json.loads(cfg_path.read_text())
        return int(cfg.get("train", {}).get("eval_every", 50))
    except Exception:
        return 50


def last_eval(metrics_csv: Path, eval_every: int) -> Tuple[Optional[float], Optional[int]]:
    if not metrics_csv.exists():
        return None, None
    try:
        df = pd.read_csv(metrics_csv)
        if "episode" not in df.columns:
            return None, None
        if eval_every > 0:
            df = df[df["episode"] % eval_every == 0]
        if df.empty or "greedy_equity" not in df.columns:
            return None, None
        ge = pd.to_numeric(df["greedy_equity"], errors="coerce")
        ruined = df["greedy_ruined"].fillna(1) if "greedy_ruined" in df.columns else None
        last_idx = ge.dropna().index.max()
        if pd.isna(last_idx):
            return None, None
        last_ge = float(ge.loc[last_idx])
        last_ruined = int(ruined.loc[last_idx]) if ruined is not None else None
        return last_ge, last_ruined
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera visão consolidada de runs em população (PBT‑lite)")
    ap.add_argument("--copy-to-train", action="store_true", help="Copia metrics.png do campeão para reports/train/pop_best_metrics.png")
    args = ap.parse_args()

    if not POP_ROOT.exists():
        print(f"Pasta não encontrada: {POP_ROOT}")
        return

    runs = sorted([p for p in POP_ROOT.glob("run_*") if p.is_dir()], key=lambda x: int(x.name.split("_")[-1]))
    if not runs:
        print("Nenhum run encontrado em pop/")
        return

    rows: List[Dict[str, Any]] = []
    for run in runs:
        last_round = latest_round_dir(run)
        if last_round is None:
            continue
        cfgs = list((POP_ROOT / "configs").glob(f"{run.name}_{last_round.name}.json"))
        eval_every = 50
        if cfgs:
            eval_every = eval_every_from_cfg(cfgs[0])
        metrics_csv = last_round / "metrics.csv"
        ge, ruined = last_eval(metrics_csv, eval_every)
        rows.append({
            "run": run.name,
            "round": last_round.name,
            "greedy_equity": ge,
            "ruined": ruined,
            "metrics_csv": str(metrics_csv),
        })

    if not rows:
        print("Sem métricas encontradas nos rounds.")
        return

    # salva resumo
    summary_path = POP_ROOT / "pop_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Resumo salvo em {summary_path}")

    # gráfico de barras com greedy_equity por run
    plt.style.use("dark_background")
    names = [r["run"] for r in rows]
    vals = [(-1 if r["greedy_equity"] is None else r["greedy_equity"]) for r in rows]
    colors = ["#ff6666" if r.get("ruined", 1) == 1 else "#4da6ff" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=colors)
    ax.set_title("População — última greedy_equity por run (vermelho=ruína)")
    ax.set_ylabel("greedy_equity")
    fig.tight_layout()
    overview_path = POP_ROOT / "pop_overview.png"
    fig.savefig(overview_path, dpi=130)
    plt.close(fig)
    print(f"Visão consolidada salva em {overview_path}")

    # campeão atual (maior greedy_equity sem ruína)
    valid = [r for r in rows if r["greedy_equity"] is not None and r.get("ruined", 1) == 0]
    if not valid:
        print("Sem campeão válido (todas avaliações com ruína ou vazias).")
        return
    champion = max(valid, key=lambda r: r["greedy_equity"])
    champ_dir = POP_ROOT / champion["run"] / champion["round"]
    champ_metrics = champ_dir / "metrics.png"
    if champ_metrics.exists():
        dst = TRAIN_ROOT / "pop_best_metrics.png"
        dst.write_bytes(champ_metrics.read_bytes())
        print(f"Copiado metrics do campeão para {dst}")
    else:
        print("Campeão não tem metrics.png disponível.")


if __name__ == "__main__":
    main()

