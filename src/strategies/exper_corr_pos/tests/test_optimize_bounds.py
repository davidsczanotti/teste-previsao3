from __future__ import annotations

from typing import Any, Dict

from src.strategies.exper_corr_pos.optimize import _suggest_param, _build_param_overrides


class FakeTrial:
    def __init__(self) -> None:
        self.called: Dict[str, Any] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        # record the bounds passed so the test can assert
        self.called[name] = {"low": low, "high": high}
        # return the upper bound to make the effect visible in overrides
        return int(high)

    # the following are unused in these tests but present for completeness
    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:  # pragma: no cover
        self.called[name] = {"low": low, "high": high, "log": log}
        return float(high)

    def suggest_categorical(self, name: str, choices):  # pragma: no cover
        self.called[name] = {"choices": list(choices)}
        return choices[0]


def _base_config_enabled_two() -> Dict[str, Any]:
    return {
        "model": {
            "expert_names": ["TrendML", "MultiFrame", "Spread", "Pattern"],
            "num_experts": 4,
            "especialistas": {
                "TrendML": True,
                "MultiFrame": True,
                "Spread": False,
                "Pattern": False,
            },
        }
    }


def test_suggest_param_topk_is_bounded_by_enabled_experts():
    trial = FakeTrial()
    spec = {"type": "int", "low": 1, "high": 4}
    # With only 2 enabled experts, high must be clamped to 2
    val = _suggest_param(trial, "model.top_k", spec, num_experts=2)
    assert trial.called["model.top_k"]["high"] == 2
    assert val == 2


def test_build_param_overrides_clamps_topk_to_enabled_experts():
    trial = FakeTrial()
    config = _base_config_enabled_two()
    search_space = {"model.top_k": {"type": "int", "low": 1, "high": 99}}
    overrides = _build_param_overrides(trial, search_space, config)
    assert overrides["model"]["top_k"] == 2
