from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from kosmofire.bs_train import fold_params


def _write(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0], dtype=array.dtype, crs="EPSG:32637", transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)) as dst:
        dst.write(array)


def _chip(root: Path, name: str, char_value: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pre = np.zeros((10, 64, 64), dtype=np.uint16)
    for band, value in enumerate([300, 500, 400, 1200, 2800, 3200, 3400, 1800, 900]):
        pre[band] = value + rng.integers(-15, 15, (64, 64))
    pre[9] = 4
    post = pre.copy()
    truth = np.zeros((64, 64), dtype=np.uint8)
    post[6, 10:30, 10:30] = 1200
    post[2, 10:30, 10:30] = 900
    post[8, 10:30, 10:30] = char_value
    truth[10:30, 10:30] = 3
    pre[2, 40:60, 40:60] = 2500
    pre[6, 40:60, 40:60] = 2700
    pre[8, 40:60, 40:60] = 3000
    post[:, 40:60, 40:60] = pre[:, 40:60, 40:60]
    paths = {"s2_pre": root / f"{name}_pre.tif", "s2_post": root / f"{name}_post.tif", "mask": root / f"{name}_mask.tif"}
    _write(paths["s2_pre"], pre)
    _write(paths["s2_post"], post)
    _write(paths["mask"], truth[None])
    return paths


def test_fold_endmembers_ignore_labels_of_held_out_chips(tmp_path):
    chips = [_chip(tmp_path, f"c{i}", 2400, i) for i in range(4)]
    train_idx = np.array([0, 1, 2])
    baseline = fold_params(chips, train_idx, ("unmix",), None).endmembers

    _write(chips[3]["mask"], np.zeros((1, 64, 64), dtype=np.uint8))
    held_out_changed = fold_params(chips, train_idx, ("unmix",), None).endmembers
    assert np.array_equal(baseline, held_out_changed)

    _write(chips[0]["mask"], np.zeros((1, 64, 64), dtype=np.uint8))
    with_train_label_changed = None
    try:
        with_train_label_changed = fold_params(chips, train_idx, ("unmix",), None).endmembers
    except ValueError:
        pass
    assert with_train_label_changed is None or not np.array_equal(baseline, with_train_label_changed)
