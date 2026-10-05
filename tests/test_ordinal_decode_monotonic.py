import numpy as np

from kosmofire.ordinal import class_probabilities, decode_severity, enforce_monotonic


def test_monotonicity_is_enforced_after_violation():
    p1 = np.array([0.9, 0.4])
    p2 = np.array([0.95, 0.6])
    p3 = np.array([0.99, 0.7])
    a, b, c = enforce_monotonic(p1, p2, p3)
    assert (a >= b).all() and (b >= c).all()


def test_class_probabilities_are_never_negative():
    rng = np.random.default_rng(0)
    p1, p2, p3 = rng.random((3, 1000))
    probs = class_probabilities(p1, p2, p3)
    assert (probs >= 0).all()


def test_decode_returns_valid_classes_and_zero_below_threshold():
    p1 = np.array([0.1, 0.9, 0.9, 0.9])
    p2 = np.array([0.0, 0.2, 0.8, 0.85])
    p3 = np.array([0.0, 0.1, 0.1, 0.8])
    out = decode_severity(p1, p2, p3, burn_threshold=0.5)
    assert out.tolist() == [0, 1, 2, 3]
    assert out.dtype == np.uint8
