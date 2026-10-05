import numpy as np

from kosmofire.features_bs import bs_features
from tests.synthetic import bs_column, make_bs_chip


def test_sar_features_finite_and_change_detected():
    chip = make_bs_chip()
    features, _ = bs_features(*chip, groups=())
    assert np.isfinite(features).all()
    dvh = bs_column(features, "dvh")
    assert dvh[30, 30] < -3.0
    assert abs(dvh[5, 5]) < 0.5


def test_ratio_is_vh_minus_vv_in_db():
    chip = make_bs_chip()
    features, _ = bs_features(*chip, groups=())
    ratio = bs_column(features, "ratio_pre")
    assert np.allclose(ratio, bs_column(features, "vh_pre") - bs_column(features, "vv_pre"), atol=1e-4)


def test_all_zero_sar_does_not_produce_nan():
    s2_pre, s2_post, s1_pre, s1_post, aux = make_bs_chip()
    features, _ = bs_features(s2_pre, s2_post, np.zeros_like(s1_pre), np.zeros_like(s1_post), aux, groups=())
    assert np.isfinite(features).all()
