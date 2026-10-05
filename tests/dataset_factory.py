import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from kosmofire.features_af import af_feature_names
from kosmofire.features_bs import bs_feature_names
from kosmofire.lgbm_utils import fit_binary
from tests.synthetic import make_af_chip, make_bs_chip


def _write(path: Path, array: np.ndarray, epsg: int = 32637) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=f"EPSG:{epsg}",
        transform=Affine(20.0, 0.0, 400000.0, 0.0, -20.0, 5300000.0),
    ) as dst:
        dst.write(array)


def build_test_dataset(root: Path, n_af: int = 3, n_bs: int = 2, bs_size: int = 512, af_size: int = 256) -> Path:
    rows = []
    for i in range(1, n_af + 1):
        cid = f"AF_te_{i:06d}"
        viirs, aux = make_af_chip(size=af_size, seed=i)
        if i == 1:
            viirs[3, 100, 100] = 340.0
        _write(root / "af" / "viirs" / f"{cid}_VIIRS_I1-I5.tif", viirs)
        _write(root / "af" / "aux" / f"{cid}_AUX.tif", aux)
        rows.append((cid, 1))
    for i in range(1, n_bs + 1):
        cid = f"BS_te_{i:06d}"
        s2_pre, s2_post, s1_pre, s1_post, aux = make_bs_chip(size=bs_size, seed=i)
        if i == 2:
            s2_pre[9] = 9
            s2_post[9] = 9
        _write(root / "bs" / "sentinel2_pre" / f"{cid}_Sentinel-2_pre.tif", s2_pre)
        _write(root / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif", s2_post)
        _write(root / "bs" / "sentinel1_pre" / f"{cid}_Sentinel-1_pre.tif", s1_pre)
        _write(root / "bs" / "sentinel1_post" / f"{cid}_Sentinel-1_post.tif", s1_post)
        _write(root / "bs" / "aux" / f"{cid}_AUX.tif", aux)
        rows += [(cid, 1), (cid, 2), (cid, 3)]
    pd.DataFrame(rows, columns=["chip_id", "class_id"]).assign(rle="").to_csv(root / "sample_submission.csv", index=False)
    return root


def build_test_models(model_dir: Path, gate_threshold: float | None = None, af_prefilter: dict | None = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    af_names = af_feature_names(())
    x = rng.normal(size=(400, len(af_names))).astype(np.float32)
    y = (x[:, af_names.index("i4_z_7") if "i4_z_7" in af_names else 0] > 0.5).astype(np.float32)
    fit_binary(x, y, 10, af_names, num_leaves=7, min_data_in_leaf=5).save_model(str(model_dir / "af_lgbm.txt"))
    af_meta = {"threshold": 0.5, "groups": []}
    if af_prefilter:
        af_meta["prefilter"] = af_prefilter
    (model_dir / "af_meta.json").write_text(json.dumps(af_meta))
    bs_names = bs_feature_names(())
    xb = rng.normal(size=(600, len(bs_names))).astype(np.float32)
    dn = bs_names.index("dnbr")
    for k, thr in zip((1, 2, 3), (0.0, 0.5, 1.0)):
        yb = (xb[:, dn] > thr).astype(np.float32)
        fit_binary(xb, yb, 10, bs_names, num_leaves=7, min_data_in_leaf=5).save_model(str(model_dir / f"bs_ge{k}.txt"))
    bs_meta = {"burn_threshold": 0.5}
    if gate_threshold is not None:
        bs_meta["gate"] = {"threshold": gate_threshold}
        fit_binary(xb, (xb[:, dn] > 0.0).astype(np.float32), 10, bs_names, num_leaves=7, min_data_in_leaf=5).save_model(str(model_dir / "bs_gate.txt"))
    bs_meta["groups"] = []
    (model_dir / "bs_meta.json").write_text(json.dumps(bs_meta))
    (model_dir / "bs_preproc.json").write_text(json.dumps({"endmembers": None, "kndvi_sigma": None}))
    if gate_threshold is None:
        fit_binary(xb, (xb[:, dn] > 0.0).astype(np.float32), 5, bs_names, num_leaves=3, min_data_in_leaf=5).save_model(str(model_dir / "bs_gate.txt"))
    return model_dir
