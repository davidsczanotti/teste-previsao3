from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna

from .train import DEFAULT_CONFIG, train_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for exper_corr_pos (config-driven)")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to JSON configuration (default: src/strategies/exper_corr_pos/config.json)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Pular a execução de baseline (config atual) antes da busca.",
    )
    return parser.parse_args()


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any], num_experts: int) -> Any:
    param_type = spec.get("type", "uniform").lower()
    if param_type == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if param_type == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if param_type == "int":
        high = int(spec["high"])
        if "model.top_k" in name:
            high = min(high, num_experts)
        return trial.suggest_int(name, int(spec["low"]), high)
    if param_type == "categorical":
        choices = spec.get("choices")
        if not choices:
            raise ValueError(f"{name}: 'choices' obrigatório para categorical.")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Tipo de busca desconhecido para {name}: {param_type}")


def _set_nested(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _build_param_overrides(
    trial: optuna.Trial,
    search_space: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    num_experts = int(base_config.get("model", {}).get("num_experts", 4))
    overrides: Dict[str, Any] = {}
    for dotted_key, spec in search_space.items():
        value = _suggest_param(trial, dotted_key, spec, num_experts)
        _set_nested(overrides, dotted_key, value)
    return overrides


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    config = json.loads(cfg_path.read_text())
    original_config = deepcopy(config)

    optimize_cfg = config.get("optimize") or {}
    if not optimize_cfg:
        raise ValueError("Bloco 'optimize' ausente no config.json.")

    trials = int(optimize_cfg.get("trials", 20))
    sampler_seed = int(optimize_cfg.get("sampler_seed", config.get("train", {}).get("seed", 42)))
    direction = optimize_cfg.get("direction", "maximize")
    storage = optimize_cfg.get("storage")
    study_name = optimize_cfg.get("study_name")
    timeout = optimize_cfg.get("timeout")
    score_metric = optimize_cfg.get("score", "best_greedy")
    experiment_id = str(optimize_cfg.get("experiment_id", "exper_corr_pos")).strip()
    apply_best_only = bool(optimize_cfg.get("apply_best_only", False))

    search_space: Dict[str, Dict[str, Any]] = optimize_cfg.get("parameters") or {
        "ppo.learning_rate": {"type": "loguniform", "low": 1e-5, "high": 5e-4},
        "ppo.gamma": {"type": "uniform", "low": 0.95, "high": 0.999},
        "model.top_k": {
            "type": "int",
            "low": 1,
            "high": config.get("model", {}).get("num_experts", 4),
        },
    }

    episodes_override = int(optimize_cfg.get("episodes", config.get("train", {}).get("episodes", 500)))
    rollout_override = int(optimize_cfg.get("rollout_steps", config.get("train", {}).get("rollout_steps", 2048)))
    eval_every = int(optimize_cfg.get("eval_every", episodes_override))
    enable_plots = bool(optimize_cfg.get("enable_plots", False))
    log_every_cfg = optimize_cfg.get("log_every")
    if log_every_cfg is None:
        log_every = max(1, episodes_override // 20)
    else:
        log_every = max(1, int(log_every_cfg))
    trial_outdir_base = Path(optimize_cfg.get("outdir", "src/strategies/exper_corr_pos/reports/optuna"))
    trial_outdir_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = trial_outdir_base / timestamp
    out_base.mkdir(parents=True, exist_ok=True)

    train_defaults = {
        "episodes": episodes_override,
        "rollout_steps": rollout_override,
        "eval_every": eval_every,
        "resume": False,
        "ckpt_every": 0,
        "final_every": 0,
        "plot_every": 0,
        "log_every": log_every,
    }

    baseline_result: Optional[Dict[str, Any]] = None
    baseline_cfg = optimize_cfg.get("baseline", {}) or {}
    run_baseline = not args.no_baseline and bool(baseline_cfg.get("enabled", True))
    if run_baseline:
        baseline_dir = out_base / "baseline"
        baseline_train = dict(train_defaults)
        baseline_train["outdir"] = str(baseline_dir)
        if "episodes" in baseline_cfg:
            baseline_train["episodes"] = int(baseline_cfg["episodes"])
        if "rollout_steps" in baseline_cfg:
            baseline_train["rollout_steps"] = int(baseline_cfg["rollout_steps"])
        if "eval_every" in baseline_cfg:
            baseline_train["eval_every"] = int(baseline_cfg["eval_every"])
        if "log_every" in baseline_cfg:
            baseline_train["log_every"] = max(1, int(baseline_cfg["log_every"]))
        baseline_overrides = {"train": baseline_train}
        print(
            "[baseline] Rodando baseline com configuração atual "
            f"(episodes={baseline_train['episodes']}, rollout_steps={baseline_train['rollout_steps']})"
            f" — logs a cada {baseline_train['log_every']} episódios"
        )
        baseline_result = train_agent(
            original_config,
            overrides=baseline_overrides,
            record_manifest=False,
            enable_plots=False,
            trial_id="baseline",
            disable_wandb=True,
        )
        (baseline_dir / "baseline_summary.json").write_text(json.dumps(baseline_result, indent=2))
        print(
            "[baseline] Concluído — "
            f"best_greedy={baseline_result['best_greedy']:.2f}, "
            f"final={baseline_result['final_greedy']:.2f}, "
            f"ruined={baseline_result['final_ruined']}"
        )

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage and study_name),
    )
    # Amarrar estudo ao experimento corrente para não misturar repositórios/estratégias distintos
    try:
        current_exp = study.user_attrs.get("experiment_id")  # type: ignore[attr-defined]
    except Exception:
        current_exp = None
    if current_exp is None:
        try:
            study.set_user_attr("experiment_id", experiment_id)
        except Exception:
            pass
    elif str(current_exp) != experiment_id:
        raise RuntimeError(
            f"Este estudo ('{study.study_name}') pertence a outro experimento (experiment_id='{current_exp}'). "
            f"Defina optimize.study_name ou optimize.experiment_id diferentes no config para separar estudos."
        )

    # Modo apply-best-only: não roda trials; aplica melhor já existente e sai
    if apply_best_only:
        completed_records: List[Dict[str, Any]] = []
        for trial in study.get_trials(deepcopy=False):
            if trial.value is None:
                continue
            params_attr = trial.user_attrs.get("param_overrides")
            try:
                params = json.loads(params_attr) if isinstance(params_attr, str) else (params_attr or {})
            except Exception:
                params = {}
            summary_attr = trial.user_attrs.get("result_summary")
            try:
                summary = json.loads(summary_attr) if isinstance(summary_attr, str) else (summary_attr or {})
            except Exception:
                summary = {}
            completed_records.append(
                {"trial": trial.number, "value": float(trial.value), "params": params, "summary": summary}
            )
        if not completed_records:
            raise RuntimeError("Nenhum trial completo encontrado neste estudo; nada a aplicar.")
        best_record = max(completed_records, key=lambda r: r["value"]) if direction == "maximize" else min(
            completed_records, key=lambda r: r["value"]
        )
        best_params = best_record["params"]
        best_trial_number = best_record["trial"]

        # Construir best_config
        best_config = deepcopy(config)
        def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
            for k, v in updates.items():
                if isinstance(v, dict) and isinstance(target.get(k), dict):
                    _deep_update(target[k], v)
                else:
                    target[k] = v
        _deep_update(best_config, best_params)

        # Persistir no mesmo esquema de outputs
        (out_base / "best_config.json").write_text(json.dumps(best_config, indent=2))
        backup_path = cfg_path.with_name(f"{cfg_path.stem}_backup_{timestamp}{cfg_path.suffix}")
        backup_path.write_text(json.dumps(config, indent=2))
        cfg_path.write_text(json.dumps(best_config, indent=2))
        # Sumário mínimo
        (out_base / "summary.json").write_text(
            json.dumps(
                {
                    "study_name": study.study_name,
                    "direction": study.direction.name,
                    "best_trial": {"number": best_trial_number, "value": best_record["value"], "params": best_params},
                    "mode": "apply_best_only",
                    "note": "Aplicado melhor trial já existente sem rodar novos trials.",
                },
                indent=2,
            )
        )
        print(
            f"[optimize] apply-best-only: Trial {best_trial_number} aplicado ao {cfg_path} (backup: {backup_path})."
        )
        return

    def objective(trial: optuna.Trial) -> float:
        trial_id = f"trial_{trial.number:04d}"
        trial_dir = out_base / trial_id
        trial_train = dict(train_defaults)
        trial_train["outdir"] = str(trial_dir)

        param_overrides = _build_param_overrides(trial, search_space, config)
        overrides = deepcopy(param_overrides)
        train_section = overrides.setdefault("train", {})
        train_section.update(trial_train)

        trial_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[trial {trial.number}] Iniciando — params={param_overrides} "
            f"episodes={trial_train['episodes']} rollout_steps={trial_train['rollout_steps']} "
            f"(log_every={trial_train['log_every']})"
        )
        result = train_agent(
            config,
            cfg_path=cfg_path,
            overrides=overrides,
            record_manifest=False,
            enable_plots=enable_plots,
            trial_id=trial_id,
            disable_wandb=True,
        )

        last_metrics = result.get("last_metrics", {})
        if score_metric == "avg_reward":
            value = float(last_metrics.get("avg_reward", float("nan")))
        elif score_metric == "sum_reward":
            value = float(last_metrics.get("sum_reward", float("nan")))
        else:
            value = float(result.get("best_greedy", float("nan")))

        if value != value:  # NaN check
            raise optuna.TrialPruned("Valor NaN encontrado para a métrica de score.")

        trial.set_user_attr("param_overrides", json.dumps(param_overrides))
        trial.set_user_attr("result_summary", json.dumps({
            "best_greedy": result.get("best_greedy"),
            "final_greedy": result.get("final_greedy"),
            "final_ruined": result.get("final_ruined"),
            "score_metric": score_metric,
        }))
        trial.set_user_attr("outdir", str(trial_dir))
        print(
            f"[trial {trial.number}] Concluído — score={value:.2f}, "
            f"best_greedy={result.get('best_greedy'):.2f}, "
            f"final={result.get('final_greedy'):.2f}, "
            f"ruined={result.get('final_ruined')}"
        )
        return value

    # Executa a rodada de trials, mas garante sumarização mesmo se houver interrupção
    try:
        study.optimize(objective, n_trials=trials, timeout=timeout)
    finally:
        pass

    # Consolida resultados; fallback robusto se helper não estiver no escopo
    try:
        completed_records = _load_records(study)
    except NameError:  # pragma: no cover — segurança adicional
        completed_records = []
        for trial in study.get_trials(deepcopy=False):
            if trial.value is None:
                continue
            try:
                params_attr = trial.user_attrs.get("param_overrides")
                params = json.loads(params_attr) if isinstance(params_attr, str) else (params_attr or {})
            except Exception:
                params = {}
            try:
                summary_attr = trial.user_attrs.get("result_summary")
                summary = json.loads(summary_attr) if isinstance(summary_attr, str) else (summary_attr or {})
            except Exception:
                summary = {}
            completed_records.append(
                {
                    "trial": trial.number,
                    "value": float(trial.value),
                    "params": params,
                    "summary": summary,
                    "outdir": trial.user_attrs.get("outdir"),
                    "datetime_start": str(trial.datetime_start) if trial.datetime_start else "",
                    "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else "",
                }
            )
    if not completed_records:
        print("[optimize] Nenhum trial completo (talvez todos interrompidos/pruned). Nada a aplicar.")
        return

    best_record = max(completed_records, key=lambda r: r["value"])
    best_params = best_record["params"]
    best_trial_number = best_record["trial"]

    best_config = deepcopy(config)
    def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                _deep_update(target[k], v)
            else:
                target[k] = v

    _deep_update(best_config, best_params)

    summary: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_value": study.best_value,
        "best_trial": {
            "number": best_trial_number,
            "params": best_params,
            "user_summary": best_record.get("summary", {}),
        },
    }
    if baseline_result:
        summary["baseline"] = baseline_result

    summary_md_lines: List[str] = [
        "# Optuna Summary",
        f"- Study: `{study.study_name}`",
        f"- Direction: `{study.direction.name}`",
        f"- Trials concluídos: {len(completed_records)}",
        "",
    ]
    if baseline_result:
        summary_md_lines.extend(
            [
                "## Baseline",
                f"- best_greedy: {baseline_result['best_greedy']:.2f}",
                f"- final_greedy: {baseline_result['final_greedy']:.2f}",
                f"- ruined: {baseline_result['final_ruined']}",
                f"- episodes: {baseline_result['episodes']}",
                "",
            ]
        )
    summary_md_lines.append("## Top Trials")
    top_rows = sorted(completed_records, key=lambda r: r["value"], reverse=True)[: min(10, len(completed_records))]
    summary_md_lines.append("| Trial | Score | Best Greedy | Final Greedy | Ruined | Params |")
    summary_md_lines.append("|-------|-------|-------------|--------------|--------|--------|")
    for r in top_rows:
        s = r.get("summary", {})
        summary_md_lines.append(
            f"| {r['trial']} | {r['value']:.2f} | "
            f"{s.get('best_greedy', float('nan')):.2f} | "
            f"{s.get('final_greedy', float('nan')):.2f} | "
            f"{s.get('final_ruined')} | `{json.dumps(r['params'])}` |"
        )
    summary_md_lines.append("")
    summary_md_lines.append("## Observações")
    summary_md_lines.append(
        "- Arquivos de cada trial em `trial_XXXX/`; baseline (quando habilitado) em `baseline/`."
    )
    summary_md_lines.append(
        "- O `config.json` principal foi atualizado automaticamente com o conjunto campeão. Backup disponível."
    )
    summary_md_lines.append("")
    summary_md_lines.append("## Scoreboard (console)")
    summary_md_lines.append("```\n" + _format_scoreboard(completed_records) + "\n```")

    (out_base / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_base / "best_config.json").write_text(json.dumps(best_config, indent=2))
    (out_base / "summary.md").write_text("\n".join(summary_md_lines))
    study.trials_dataframe().to_csv(out_base / "trials.csv", index=False)

    backup_path = cfg_path.with_name(f"{cfg_path.stem}_backup_{timestamp}{cfg_path.suffix}")
    backup_path.write_text(json.dumps(config, indent=2))
    cfg_path.write_text(json.dumps(best_config, indent=2))

    print(f"[optimize] Melhor valor: {study.best_value:.4f} (trial {best_trial_number})")
    if baseline_result:
        improvement = study.best_value - baseline_result.get("best_greedy", 0.0)
        print(f"[optimize] Comparado ao baseline: Δbest_greedy={improvement:.2f}")
    print(f"[optimize] Config principal atualizado em {cfg_path} (backup: {backup_path})")
    print(f"[optimize] Scoreboard resumido:\n{_format_scoreboard(completed_records)}")
    print(f"[optimize] Relatórios: {out_base}")


if __name__ == "__main__":
    main()
def _load_records(study: optuna.study.Study) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        if trial.value is None:
            continue
        params_attr = trial.user_attrs.get("param_overrides")
        if isinstance(params_attr, str):
            try:
                params = json.loads(params_attr)
            except json.JSONDecodeError:
                params = {}
        elif isinstance(params_attr, dict):
            params = params_attr
        else:
            params = {}
        summary_attr = trial.user_attrs.get("result_summary")
        if isinstance(summary_attr, str):
            try:
                summary = json.loads(summary_attr)
            except json.JSONDecodeError:
                summary = {}
        elif isinstance(summary_attr, dict):
            summary = summary_attr
        else:
            summary = {}
        records.append(
            {
                "trial": trial.number,
                "value": float(trial.value),
                "params": params,
                "summary": summary,
                "outdir": trial.user_attrs.get("outdir"),
                "datetime_start": str(trial.datetime_start) if trial.datetime_start else "",
                "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else "",
            }
        )
    return records


def _format_scoreboard(records: List[Dict[str, Any]], top_n: int = 5) -> str:
    if not records:
        return "Nenhum trial concluído."
    rows = sorted(records, key=lambda r: r["value"], reverse=True)[:top_n]
    lines = ["Top trials:"]
    for r in rows:
        best = r.get("summary", {}).get("best_greedy")
        final = r.get("summary", {}).get("final_greedy")
        ruined = r.get("summary", {}).get("final_ruined")
        lines.append(
            f"  • Trial {r['trial']:>3}: score={r['value']:.2f}, best_greedy={best:.2f} "
            f"final={final:.2f}, ruined={ruined}, params={r['params']}"
        )
    return "\n".join(lines)
