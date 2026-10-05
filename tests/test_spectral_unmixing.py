import numpy as np

from kosmofire.unmixing import estimate_endmembers, unmix

ENDMEMBERS = np.array(
    [
        [0.04, 0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.20, 0.25],
        [0.03, 0.05, 0.04, 0.10, 0.30, 0.38, 0.42, 0.20, 0.09],
        [0.10, 0.13, 0.17, 0.20, 0.22, 0.24, 0.26, 0.35, 0.30],
    ]
).T


def test_known_mixture_is_recovered_within_five_percent():
    truth = np.array([0.2, 0.5, 0.3])
    pixel = (ENDMEMBERS @ truth).reshape(9, 1, 1)
    fractions = unmix(pixel, ENDMEMBERS)[:, 0, 0]
    assert np.allclose(fractions, truth, atol=0.05)


def test_fractions_sum_to_one_on_arbitrary_and_degenerate_input():
    rng = np.random.default_rng(0)
    cube = rng.random((9, 8, 8)) * 0.6
    cube[:, 0, 0] = 0.0
    cube[:, 1, 1] = 5.0
    fractions = unmix(cube, ENDMEMBERS)
    assert np.allclose(fractions.sum(axis=0), 1.0, atol=1e-5)
    assert (fractions >= 0).all() and (fractions <= 1).all()


def test_endmember_estimation_uses_medians_per_class():
    rng = np.random.default_rng(1)
    samples = {
        "char": ENDMEMBERS[:, 0] + rng.normal(0, 0.001, (200, 9)),
        "vegetation": ENDMEMBERS[:, 1] + rng.normal(0, 0.001, (200, 9)),
        "soil": ENDMEMBERS[:, 2] + rng.normal(0, 0.001, (200, 9)),
    }
    estimated = estimate_endmembers(samples)
    assert np.allclose(estimated, ENDMEMBERS, atol=0.005)
