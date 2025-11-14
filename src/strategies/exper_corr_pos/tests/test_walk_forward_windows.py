from __future__ import annotations

from src.strategies.exper_corr_pos.walk_forward import WFWindow, build_windows


def test_build_windows_basic_properties():
    length = 100
    train_len = 40
    val_len = 20
    step = 10

    windows = build_windows(length, train_len, val_len, step)

    # Esperado: starts em 0,10,20,30 (pois 40+60 == 100 não satisfaz < length).
    assert len(windows) == 4
    starts = [w.train_start for w in windows]
    assert starts == [0, 10, 20, 30]

    for w in windows:
        # Comprimentos de treino e validação corretos
        assert w.train_end - w.train_start == train_len
        assert w.val_end - w.val_start == val_len
        # Ordem e limites dentro da série
        assert 0 <= w.train_start < w.train_end <= w.val_start < w.val_end < length


def test_build_windows_empty_when_series_too_short():
    # Quando length não comporta train+val, nenhuma janela deve ser criada.
    length = 50
    train_len = 30
    val_len = 21
    windows = build_windows(length, train_len, val_len, step=10)
    assert windows == []


def test_build_windows_single_window_when_step_large():
    length = 80
    train_len = 30
    val_len = 20
    step = 100  # maior que o total, só primeira janela deve aparecer

    windows = build_windows(length, train_len, val_len, step)
    assert len(windows) == 1
    w = windows[0]
    assert w.train_start == 0
    assert w.train_end == train_len
    assert w.val_start == train_len
    assert w.val_end == train_len + val_len

