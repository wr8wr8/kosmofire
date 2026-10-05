import numpy as np

from kosmofire.lgbm_utils import fit_binary


def test_repeated_training_gives_identical_predictions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 12)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.3, 3000) > 0).astype(np.float32)
    names = [f"f{i}" for i in range(12)]
    a = fit_binary(x, y, 40, names, num_leaves=15).predict(x)
    b = fit_binary(x, y, 40, names, num_leaves=15).predict(x)
    assert np.array_equal(a, b)


def test_score_difference_between_two_runs_within_tolerance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 6)).astype(np.float32)
    y = (x[:, 0] > 0.3).astype(np.float32)
    names = [f"f{i}" for i in range(6)]
    runs = [(fit_binary(x, y, 30, names).predict(x) >= 0.5) for _ in range(2)]
    accuracies = [float((r == y.astype(bool)).mean()) for r in runs]
    assert abs(accuracies[0] - accuracies[1]) <= 0.005
