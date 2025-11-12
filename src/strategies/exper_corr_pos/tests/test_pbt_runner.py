import json
import sys
from pathlib import Path

from src.strategies.exper_corr_pos.scripts import pop_runner


def test_pbt_runner_respects_config(monkeypatch, tmp_path):
    strategy_root = tmp_path / "exper_corr_pos"
    reports_root = strategy_root / "reports" / "train"
    reports_root.mkdir(parents=True)

    seed_ckpt = reports_root / "moe_policy_final.pt"
    seed_ckpt.write_text("seed-weights")

    config = {
        "model": {"temperature": 1.0, "top_k": 1, "num_experts": 2},
        "train": {"eval_every": 1, "episodes": 32, "rollout_steps": 64},
        "ppo": {"learning_rate": 3e-4},
        "pbt": {
            "pop": 2,
            "rounds": 2,
            "episodes": 7,
            "concurrency": 2,
            "threads": 3,
            "seed_checkpoint": str(seed_ckpt),
            "promote_to_root": True,
        },
    }
    config_path = strategy_root / "config.json"
    config_path.write_text(json.dumps(config))

    pop_root = reports_root / "pop"
    pop_configs = pop_root / "configs"

    monkeypatch.setattr(pop_runner, "ROOT", strategy_root)
    monkeypatch.setattr(pop_runner, "POP_ROOT", pop_root)
    monkeypatch.setattr(pop_runner, "CONF_ROOT", pop_configs)

    run_calls: list[dict[str, object]] = []

    def fake_run_process(cfg_path: Path, env_extra: dict[str, str]):
        run_calls.append({"cfg_path": Path(cfg_path), "env": dict(env_extra)})

        class _Proc:
            def wait(self_inner):
                cfg = json.loads(Path(cfg_path).read_text())
                outdir = Path(cfg["train"]["outdir"])
                outdir.mkdir(parents=True, exist_ok=True)
                stem = Path(cfg_path).stem
                _, run_idx, _, round_idx = stem.split("_")
                equity = 1000 + int(run_idx) * 100 + int(round_idx) * 10
                metrics = outdir / "metrics.csv"
                metrics.write_text(
                    "episode,greedy_equity,greedy_ruined\n"
                    f"{cfg['train']['episodes']},{equity},0\n"
                )
                (outdir / "moe_policy_final.pt").write_text(f"ckpt_{equity}")

        return _Proc()

    monkeypatch.setattr(pop_runner, "run_process", fake_run_process)

    monkeypatch.setattr(sys, "argv", ["pop_runner", "--base", str(config_path)])
    pop_runner.main()

    expected_runs = config["pbt"]["pop"] * config["pbt"]["rounds"]
    assert len(run_calls) == expected_runs

    for call in run_calls:
        env = call["env"]
        assert env["OMP_NUM_THREADS"] == str(config["pbt"]["threads"])
        assert env["MKL_NUM_THREADS"] == str(config["pbt"]["threads"])
        assert env["PYTORCH_NUM_THREADS"] == str(config["pbt"]["threads"])

    scoreboard_path = pop_root / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    assert len(scoreboard) == config["pbt"]["rounds"]
    for idx, entry in enumerate(scoreboard):
        assert entry["round"] == idx
        champion_path = Path(entry["champion"])
        assert "run_1" in champion_path.parts

    cfg_round0 = json.loads((pop_configs / "run_0_round_0.json").read_text())
    assert cfg_round0["train"]["episodes"] == config["pbt"]["episodes"]
    assert cfg_round0["train"]["resume_path"] == str(seed_ckpt)

    cfg_round1 = json.loads((pop_configs / "run_0_round_1.json").read_text())
    assert cfg_round1["train"]["resume_path"] == scoreboard[0]["checkpoint"]

    final_ckpt = reports_root / "moe_policy_final.pt"
    assert final_ckpt.exists()
    assert final_ckpt.read_text() == Path(scoreboard[-1]["checkpoint"]).read_text()
