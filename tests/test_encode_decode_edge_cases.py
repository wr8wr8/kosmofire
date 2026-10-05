import numpy as np

from kosmofire.rle import decode_rle, encode_rle


def test_last_pixel_of_af_chip():
    mask = np.zeros((256, 256), dtype=bool)
    mask[-1, -1] = True
    assert decode_rle(encode_rle(mask), mask.shape)[-1, -1]


def test_first_pixel_of_bs_chip():
    mask = np.zeros((512, 512), dtype=bool)
    mask[0, 0] = True
    assert encode_rle(mask) == "1 1"


def test_chip_without_fire_has_empty_encoding():
    assert encode_rle(np.zeros((256, 256), dtype=bool)) == ""
