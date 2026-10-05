import numpy as np

from kosmofire.dozier import LAMBDA_MIR, LAMBDA_TIR, dozier_retrieval, inverse_planck, planck


def _mix(p, tf, tb):
    l4 = p * planck(LAMBDA_MIR, np.array(tf)) + (1 - p) * planck(LAMBDA_MIR, np.array(tb))
    l5 = p * planck(LAMBDA_TIR, np.array(tf)) + (1 - p) * planck(LAMBDA_TIR, np.array(tb))
    return inverse_planck(LAMBDA_MIR, l4), inverse_planck(LAMBDA_TIR, l5)


def test_planck_inverse_roundtrip():
    t = np.array([280.0, 300.0, 600.0, 1000.0])
    assert np.allclose(inverse_planck(LAMBDA_MIR, planck(LAMBDA_MIR, t)), t, rtol=1e-9)


def test_recovers_known_subpixel_fraction_and_temperature():
    cases = [(0.02, 800.0, 295.0), (0.005, 1000.0, 300.0), (0.05, 700.0, 290.0)]
    for p_true, tf_true, tb in cases:
        i4, i5 = _mix(p_true, tf_true, tb)
        p, tf, ok = dozier_retrieval(np.array([i4]), np.array([i5]), np.array([tb]), np.array([tb]))
        assert ok[0]
        assert abs(p[0] - p_true) / p_true < 0.05
        assert abs(tf[0] - tf_true) / tf_true < 0.05


def test_impossible_input_is_flagged_not_nan_leaking():
    p, tf, ok = dozier_retrieval(np.array([290.0]), np.array([285.0]), np.array([295.0]), np.array([285.0]))
    assert not ok[0]
    assert np.isnan(p[0]) and np.isnan(tf[0])


def test_nan_input_does_not_raise():
    p, tf, ok = dozier_retrieval(np.array([np.nan, 340.0]), np.array([285.0, 290.0]), np.array([295.0, 295.0]), np.array([285.0, 285.0]))
    assert not ok[0]
    assert ok.dtype == bool
