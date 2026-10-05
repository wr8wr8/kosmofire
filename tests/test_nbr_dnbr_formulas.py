import numpy as np

from kosmofire.features_bs import bs_features
from tests.synthetic import bs_column, make_bs_chip


def test_nbr_dnbr_rdnbr_match_manual_values():
    chip = make_bs_chip()
    features, _ = bs_features(*chip, groups=())
    s2_pre, s2_post = chip[0], chip[1]
    b8a_pre, b12_pre = s2_pre[6].astype(np.float64) / 1e4, s2_pre[8].astype(np.float64) / 1e4
    b8a_post, b12_post = s2_post[6].astype(np.float64) / 1e4, s2_post[8].astype(np.float64) / 1e4
    nbr_pre = (b8a_pre - b12_pre) / (b8a_pre + b12_pre + 1e-6)
    nbr_post = (b8a_post - b12_post) / (b8a_post + b12_post + 1e-6)
    dnbr = nbr_pre - nbr_post
    assert np.allclose(bs_column(features, "nbr_pre"), nbr_pre, atol=1e-4)
    assert np.allclose(bs_column(features, "dnbr"), dnbr, atol=1e-4)
    assert np.allclose(bs_column(features, "rdnbr"), dnbr / np.sqrt(np.abs(nbr_pre) + 1e-3), atol=1e-3)


def test_burned_block_has_much_higher_dnbr_than_background():
    features, _ = bs_features(*make_bs_chip(), groups=())
    dnbr = bs_column(features, "dnbr")
    assert dnbr[30, 30] > 0.5
    assert abs(dnbr[5, 5]) < 0.05
