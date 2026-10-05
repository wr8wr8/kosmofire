import numpy as np

from kosmofire.bootstrap import grouped_bootstrap_difference


def _score(items):
    tp = sum(i[0] for i in items)
    fp = sum(i[1] for i in items)
    fn = sum(i[2] for i in items)
    return tp / (tp + fp + fn)


def _make(rng, n_groups, tp_mean):
    return {g: (rng.poisson(tp_mean), rng.poisson(20), rng.poisson(20)) for g in range(n_groups)}


def test_real_difference_excludes_zero():
    rng = np.random.default_rng(0)
    a = _make(rng, 60, 200)
    b = _make(rng, 60, 120)
    result = grouped_bootstrap_difference(a, b, _score, iterations=300, seed=1)
    assert result.excludes_zero
    assert result.low > 0


def test_identical_models_do_not_trigger_false_positive():
    rng = np.random.default_rng(0)
    a = _make(rng, 60, 150)
    result = grouped_bootstrap_difference(a, dict(a), _score, iterations=300, seed=1)
    assert not result.excludes_zero
    assert result.mean_difference == 0.0


def test_resampling_is_over_groups_not_items():
    a = {g: (1, 0, 0) for g in range(5)}
    seen = []

    def spy(items):
        seen.append(len(items))
        return 0.0

    grouped_bootstrap_difference(a, a, spy, iterations=3, seed=0)
    assert set(seen) == {5}
