from pathlib import Path

import numpy as np
import pandas as pd

from .chipio import ChipStore, read_mask, read_raster
from .features_af import DEFAULT_GROUPS, af_feature_names, af_features, af_valid_mask
from .parallel import parallel_map


def _compute_chip(args: tuple[dict[str, Path], tuple[str, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths, groups = args
    viirs = read_raster(paths["viirs"])
    aux = read_raster(paths["aux"])
    features = af_features(viirs, aux, groups)
    valid = af_valid_mask(viirs)
    if "mask" in paths:
        truth = read_mask(paths["mask"]).astype(np.uint8)
    else:
        truth = np.zeros(valid.shape, dtype=np.uint8)
    return features, valid, truth


def build_af_cache(store: ChipStore, chip_ids: list[str], cache_dir: Path, groups: tuple[str, ...] = DEFAULT_GROUPS, cache_name: str = "af_x.npy") -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(chip_ids)
    n_features = len(af_feature_names(groups))
    x = np.lib.format.open_memmap(cache_dir / cache_name, mode="w+", dtype=np.float32, shape=(n, n_features, 256, 256))
    valid = np.zeros((n, 256, 256), dtype=bool)
    truth = np.zeros((n, 256, 256), dtype=np.uint8)
    results = parallel_map(_compute_chip, [(store.files(cid).paths, groups) for cid in chip_ids])
    for i, (features, v, t) in enumerate(results):
        x[i] = features
        valid[i] = v
        truth[i] = t
    x.flush()
    np.save(cache_dir / "af_valid.npy", valid)
    np.save(cache_dir / "af_truth.npy", truth)
    pd.Series(chip_ids).to_csv(cache_dir / "af_ids.csv", index=False, header=False)
