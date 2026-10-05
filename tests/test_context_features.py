import numpy as np

from kosmofire.features_af import af_features
from tests.synthetic import af_column, make_af_chip


def test_z_score_highlights_synthetic_outlier():
    viirs, aux = make_af_chip()
    viirs[3, 30, 30] = 345.0
    viirs[4, 30, 30] = 292.0
    features = af_features(viirs, aux)
    z = af_column(features, "i4_z_7")
    assert z[30, 30] > 10
    assert np.abs(np.delete(z.ravel(), 30 * 64 + 30)).max() < z[30, 30]


def test_delta_mir_tir_is_i4_minus_i5():
    viirs, aux = make_af_chip()
    features = af_features(viirs, aux)
    assert np.allclose(af_column(features, "delta_mir_tir"), viirs[3] - viirs[4], atol=1e-4)


def test_no_inf_and_only_dozier_columns_may_be_nan_with_nan_input():
    viirs, aux = make_af_chip()
    viirs[3, 5, 5] = np.nan
    features = af_features(viirs, aux)
    assert not np.isinf(features).any()
    base = af_features(viirs, aux, groups=())
    assert np.isfinite(base).all()
    assert af_column(features, "dozier_converged")[5, 5] == 0


def test_landcover_encoding_variants_expose_matching_columns():
    viirs, aux = make_af_chip()
    categorical = af_features(viirs, aux, groups=())
    one_hot = af_features(viirs, aux, groups=("onehot",))
    assert one_hot.shape[0] == categorical.shape[0] + 10
    assert one_hot[-11:].sum(axis=0).max() <= 1.0
