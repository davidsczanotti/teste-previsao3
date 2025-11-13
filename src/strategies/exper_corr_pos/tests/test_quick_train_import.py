from __future__ import annotations

import importlib


def test_quick_train_module_loads_and_has_main():
    mod = importlib.import_module("src.strategies.exper_corr_pos.scripts.quick_train")
    assert hasattr(mod, "main") and callable(getattr(mod, "main"))
