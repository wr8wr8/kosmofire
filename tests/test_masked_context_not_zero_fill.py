import numpy as np

from kosmofire.features_bs import bs_features, masked_window_stats
from tests.synthetic import bs_column, make_bs_chip


def test_masked_mean_uses_only_valid_pixels():
    rng = np.random.default_rng(0)
    values = np.full((60, 60), 0.6, dtype=np.float32)
    valid = rng.random((60, 60)) < 0.4
    mean, _, support = masked_window_stats(values, valid, 15)
    centre = mean[20:40, 20:40]
    assert np.nanmean(centre) > 0.59
    zero_fill = (np.where(valid, values, 0.0)).mean()
    assert zero_fill < 0.3


def test_zero_support_gives_nan_not_zero():
    values = np.full((40, 40), 0.6, dtype=np.float32)
    valid = np.zeros((40, 40), dtype=bool)
    mean, std, support = masked_window_stats(values, valid, 15)
    assert np.isnan(mean).all() and np.isnan(std).all()
    assert (support == 0).all()


def test_feature_pipeline_context_survives_heavy_cloud():
    s2_pre, s2_post, s1_pre, s1_post, aux = make_bs_chip(size=96)
    rng = np.random.default_rng(1)
    cloud = rng.random((96, 96)) < 0.6
    s2_post[9][cloud] = 9
    s2_post[6, 30:70, 30:70] = 1200
    s2_post[8, 30:70, 30:70] = 2400
    features, valid = bs_features(s2_pre, s2_post, s1_pre, s1_post, aux, groups=())
    mean15 = bs_column(features, "dnbr_mean_15")
    support = bs_column(features, "dnbr_support_15")
    assert np.nanmean(mean15[45:55, 45:55]) > 0.5
    assert 0.2 < support[45:55, 45:55].mean() < 0.6
