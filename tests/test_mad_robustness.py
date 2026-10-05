import numpy as np
from scipy import ndimage as ndi

from kosmofire.features_af import _robust_context


def test_mad_context_stays_calm_next_to_a_burning_neighbour():
    rng = np.random.default_rng(1)
    band = 295.0 + rng.normal(0, 0.5, (64, 64)).astype(np.float32)
    band[30, 30] = 350.0
    band[30, 31] = 348.0
    _, _, z = _robust_context(band, 7)
    neighbour_z = z[30, 32]
    assert abs(neighbour_z) < 4


def test_mean_std_context_raises_false_alarm_for_the_same_neighbour():
    rng = np.random.default_rng(1)
    band = 295.0 + rng.normal(0, 0.5, (64, 64)).astype(np.float32)
    band[30, 30] = 350.0
    band[30, 31] = 348.0
    mean = ndi.uniform_filter(band, 7)
    sq = ndi.uniform_filter(band * band, 7)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    z_mean = (band - mean) / (std + 1e-6)
    _, _, z_mad = _robust_context(band, 7)
    assert abs(z_mad[30, 30]) > abs(z_mean[30, 30])
    assert z_mad[30, 30] > 30
