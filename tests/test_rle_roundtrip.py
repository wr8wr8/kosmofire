import numpy as np
import pytest

from kosmofire.rle import decode_rle, encode_rle


def test_documented_example_matches_encoding():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 3:5] = True
    mask[4, 1:4] = True
    assert encode_rle(mask) == "20 2 34 3"


def test_roundtrip_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = rng.random((64, 64)) > 0.7
        assert np.array_equal(decode_rle(encode_rle(mask), mask.shape), mask)


def test_empty_mask_gives_empty_string():
    mask = np.zeros((16, 16), dtype=bool)
    assert encode_rle(mask) == ""
    assert not decode_rle("", mask.shape).any()


def test_full_chip_is_single_run():
    mask = np.ones((16, 16), dtype=bool)
    assert encode_rle(mask) == "1 256"
    assert decode_rle("1 256", mask.shape).all()


def test_runs_wrap_across_rows():
    mask = np.zeros((4, 4), dtype=bool)
    mask.ravel()[3:6] = True
    assert encode_rle(mask) == "4 3"


@pytest.mark.parametrize("bad", ["1 3 3 2", "5 2 1 1", "17 1", "0 2", "1", "1 0"])
def test_decode_rejects_invalid_runs(bad):
    with pytest.raises(ValueError):
        decode_rle(bad, (4, 4))
