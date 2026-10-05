import numpy as np

from kosmofire.features_af import af_features
from tests.synthetic import af_column, make_af_chip


def test_saturation_flag_triggers_at_threshold():
    viirs, aux = make_af_chip()
    viirs[3, 10, 10] = 365.9
    viirs[3, 10, 11] = 366.0
    viirs[3, 10, 12] = 367.0
    flag = af_column(af_features(viirs, aux), "i4_saturated")
    assert flag[10, 10] == 0
    assert flag[10, 11] == 1
    assert flag[10, 12] == 1
