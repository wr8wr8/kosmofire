import numpy as np
from sklearn.isotonic import IsotonicRegression

from kosmofire.ordinal import decode_severity, pava_nonincreasing


def test_documented_example_is_pooled_not_clipped():
    q1, q2, q3 = pava_nonincreasing(np.array([0.45]), np.array([0.70]), np.array([0.20]))
    assert np.allclose([q1[0], q2[0], q3[0]], [0.575, 0.575, 0.20])
    assert not np.isclose(q2[0], 0.45)


def test_matches_sklearn_isotonic_on_random_triples():
    rng = np.random.default_rng(0)
    triples = rng.random((500, 3))
    q1, q2, q3 = pava_nonincreasing(triples[:, 0], triples[:, 1], triples[:, 2])
    ours = np.column_stack([q1, q2, q3])
    for row, expected in zip(triples, ours):
        fit = IsotonicRegression(increasing=False).fit_transform([0, 1, 2], row)
        assert np.allclose(fit, expected, atol=1e-12)


def test_output_is_monotone_and_class_probabilities_are_nonnegative():
    rng = np.random.default_rng(1)
    t = rng.random((1000, 3))
    q1, q2, q3 = pava_nonincreasing(t[:, 0], t[:, 1], t[:, 2])
    assert (q1 >= q2 - 1e-12).all() and (q2 >= q3 - 1e-12).all()
    probs = np.stack([1 - q1, q1 - q2, q2 - q3, q3], axis=-1)
    assert (probs >= -1e-12).all()


def test_decode_uses_projection():
    out = decode_severity(np.array([0.45]), np.array([0.70]), np.array([0.20]), burn_threshold=0.4)
    assert out[0] == 2


CASES = {
    "audit_example": ((0.45, 0.70, 0.20), (0.575, 0.575, 0.20)),
    "already_monotone": ((0.9, 0.5, 0.1), (0.9, 0.5, 0.1)),
    "fully_reversed": ((0.1, 0.5, 0.9), (0.5, 0.5, 0.5)),
    "first_pair_pooled_then_stays_above_third": ((0.2, 0.6, 0.1), (0.4, 0.4, 0.1)),
    "first_pair_pooled_then_merges_with_third": ((0.1, 0.5, 0.4), (1 / 3.0, 1 / 3.0, 1 / 3.0)),
    "second_pair_pooled_below_first": ((0.9, 0.3, 0.5), (0.9, 0.4, 0.4)),
    "second_pair_pooled_then_merges_with_first": ((0.5, 0.3, 0.9), (0.5666666666666667, 0.5666666666666667, 0.5666666666666667)),
    "ties_unchanged": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "zeros": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "ones": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "boundary_reversed_zero_one": ((0.0, 0.0, 1.0), (1 / 3.0, 1 / 3.0, 1 / 3.0)),
    "boundary_one_zero_one": ((1.0, 0.0, 1.0), (1.0, 0.5, 0.5)),
    "boundary_zero_one_zero": ((0.0, 1.0, 0.0), (0.5, 0.5, 0.0)),
}


def test_every_branch_and_boundary_case_gives_the_exact_projection():
    for name, (inputs, expected) in CASES.items():
        got = pava_nonincreasing(*(np.array([v]) for v in inputs))
        assert np.allclose([g[0] for g in got], expected, atol=1e-12), name
        fit = IsotonicRegression(increasing=False).fit_transform([0, 1, 2], list(inputs))
        assert np.allclose(fit, expected, atol=1e-12), f"{name}: reference disagrees with expectation"


def test_grid_of_boundary_values_matches_sklearn_on_all_combinations():
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for a in values:
        for b in values:
            for c in values:
                got = pava_nonincreasing(np.array([a]), np.array([b]), np.array([c]))
                fit = IsotonicRegression(increasing=False).fit_transform([0, 1, 2], [a, b, c])
                assert np.allclose([g[0] for g in got], fit, atol=1e-12), (a, b, c)
