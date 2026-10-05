import numpy as np

from kosmofire.config import REFLECTANCE_SCALE, SAR_DB_SCALE
from kosmofire.features_bs import bs_features
from tests.synthetic import bs_column, make_bs_chip


def test_constants_match_data_dictionary():
    assert REFLECTANCE_SCALE == 10000.0
    assert SAR_DB_SCALE == 100.0


def test_reflectance_and_sar_are_converted_to_physical_units():
    chip = make_bs_chip()
    features, _ = bs_features(*chip, groups=())
    assert abs(bs_column(features, "b12_post")[5, 5] - 0.09) < 0.01
    assert abs(bs_column(features, "vh_pre")[5, 5] - (-18.0)) < 0.6
    assert abs(bs_column(features, "vv_pre")[5, 5] - (-10.0)) < 0.6
