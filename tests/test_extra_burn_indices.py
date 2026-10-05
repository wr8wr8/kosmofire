import numpy as np

from kosmofire.features_bs import bs_features
from tests.synthetic import bs_column, make_bs_chip

ENDMEMBERS = np.tile(np.linspace(0.05, 0.4, 9)[:, None], (1, 3)) * np.array([1.0, 1.4, 0.7])


def _features(chip=None):
    chip = chip or make_bs_chip()
    return bs_features(*chip, groups=("extra", "sar_texture", "unmix", "cloud"), endmembers=ENDMEMBERS)[0]


def test_indices_match_manual_calculation():
    chip = make_bs_chip()
    features = _features(chip)
    b12 = chip[1][8].astype(np.float64) / 1e4
    b11 = chip[1][7].astype(np.float64) / 1e4
    assert np.allclose(bs_column(features, "mirbi_post"), 10 * b12 - 9.8 * b11 + 2, atol=1e-4)
    b8a, b4 = chip[1][6].astype(np.float64) / 1e4, chip[1][2].astype(np.float64) / 1e4
    assert np.allclose(bs_column(features, "savi_post"), 1.5 * (b8a - b4) / (b8a + b4 + 0.5), atol=1e-4)


def test_boundary_inputs_do_not_produce_nan_or_inf():
    s2_pre, s2_post, s1_pre, s1_post, aux = make_bs_chip()
    s2_pre[2] = 1000
    s2_pre[6] = 600
    s2_post[2] = 0
    s2_post[6] = 0
    features = _features((s2_pre, s2_post, s1_pre, s1_post, aux))
    assert np.isfinite(features).all()


def test_burn_signal_separates_dnbr_and_dndvi_residual():
    features = _features()
    residual = bs_column(features, "dnbr_minus_dndvi")
    assert np.isfinite(residual).all()
    assert residual[30, 30] != residual[5, 5]
