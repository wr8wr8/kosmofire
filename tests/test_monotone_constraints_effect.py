import numpy as np

from kosmofire.lgbm_utils import fit_binary


def _dataset(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    other = rng.normal(size=n)
    p = 1 / (1 + np.exp(-4 * x))
    y = (rng.random(n) < p).astype(np.float32)
    flip = (x > 0.5) & (rng.random(n) < 0.35)
    y[flip] = 0.0
    return np.column_stack([x, other]).astype(np.float32), y


def _is_monotone(model):
    grid = np.column_stack([np.linspace(-1, 1, 200), np.zeros(200)]).astype(np.float32)
    pred = model.predict(grid)
    return bool(np.all(np.diff(pred) >= -1e-9))


def test_constraint_forces_monotone_response_where_unconstrained_model_is_not():
    x, y = _dataset()
    names = ["dnbr", "noise"]
    free = fit_binary(x, y, 60, names, num_leaves=15, min_data_in_leaf=5)
    constrained = fit_binary(x, y, 60, names, num_leaves=15, min_data_in_leaf=5, monotone_constraints=[1, 0])
    assert not _is_monotone(free)
    assert _is_monotone(constrained)
