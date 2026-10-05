import numpy as np

from kosmofire.ordinal import class_probabilities


def test_projection_makes_probabilities_valid_without_extra_clipping():
    rng = np.random.default_rng(2)
    q = rng.random((5000, 3))
    probs = class_probabilities(q[:, 0], q[:, 1], q[:, 2])
    assert probs.min() >= -1e-12
    assert np.allclose(probs.sum(axis=-1), 1.0)
