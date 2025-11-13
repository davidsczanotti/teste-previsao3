import numpy as np

from src.strategies.exper_corr_pos.features import _online_ridge_predictions


def test_online_ridge_predictions_uses_delayed_updates():
    # Simple 1D features with constant value, constant positive target
    n = 20
    H = 3
    X = np.ones((n, 1), dtype=np.float64)
    y = np.full(n, 2.0, dtype=np.float64)

    preds = _online_ridge_predictions(X, y, decay=1.0, ridge=1e-3, horizon=H)

    # Before any update is possible (indices < H), weights are initial zeros -> predictions must be 0
    assert np.allclose(preds[:H], 0.0)
    # The first delayed update is applied at step H (after predicting index H),
    # so the first non-zero prediction must appear at H+1 or later
    assert preds[H + 1] > 0.0
