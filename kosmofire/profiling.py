import json
import time
from pathlib import Path

import numpy as np

from .af_data import _compute_chip
from .bs_data import load_bs_features
from .chipio import ChipStore, read_raster
from .features_af import af_feature_names
from .features_bs import bs_feature_names
from .inference_pipeline import _af_candidates, _bs_candidates
from .lgbm_utils import load_booster
from .unmixing import load_endmembers


def _timed(store: dict, key: str, started: float) -> None:
    store[key] = store.get(key, 0.0) + (time.perf_counter() - started)


def profile_stages(data_dir: Path, model_dir: Path, n_af: int = 10, n_bs: int = 8) -> dict:
    store = ChipStore(data_dir)
    af_meta = json.loads((model_dir / "af_meta.json").read_text())
    bs_meta = json.loads((model_dir / "bs_meta.json").read_text())
    af_groups = tuple(af_meta.get("groups", ()))
    bs_groups = tuple(bs_meta.get("groups", ()))
    af_names = af_feature_names(af_groups)
    bs_names = bs_feature_names(bs_groups)
    af_model = load_booster(model_dir / "af_lgbm.txt")
    bs_models = [load_booster(model_dir / f"bs_ge{k}.txt") for k in (1, 2, 3)]
    gate = load_booster(model_dir / "bs_gate.txt")
    endmembers = load_endmembers(model_dir / "bs_endmembers.json")

    af, bs = {}, {}
    af_ids = store.chip_ids("af")[:n_af]
    bs_ids = store.chip_ids("bs")[:n_bs]
    for cid in af_ids:
        paths = store.files(cid).paths
        t = time.perf_counter()
        for key in ("viirs", "aux"):
            read_raster(paths[key])
        _timed(af, "io_read", t)
        t = time.perf_counter()
        features, valid, _ = _compute_chip((paths, af_groups))
        _timed(af, "io_read_plus_features", t)
        cand = _af_candidates(features, af_names, af_meta.get("prefilter")) & valid
        af["pixels_total"] = af.get("pixels_total", 0) + int(valid.size)
        af["pixels_candidates"] = af.get("pixels_candidates", 0) + int(cand.sum())
        t = time.perf_counter()
        af_model.predict(np.ascontiguousarray(features[:, valid].T), num_threads=1)
        _timed(af, "predict_no_prefilter", t)
        t = time.perf_counter()
        af_model.predict(np.ascontiguousarray(features[:, cand].T), num_threads=1)
        _timed(af, "predict_with_prefilter", t)
    for cid in bs_ids:
        paths = store.files(cid).paths
        t = time.perf_counter()
        for key in ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux"):
            read_raster(paths[key])
        _timed(bs, "io_read", t)
        t = time.perf_counter()
        features, _ = load_bs_features(paths, bs_groups, endmembers)
        _timed(bs, "io_read_plus_features", t)
        flat = np.ascontiguousarray(features.reshape(features.shape[0], -1).T)
        t = time.perf_counter()
        bs_models[0].predict(flat, num_threads=1)
        _timed(bs, "ge1_full_pixels", t)
        t = time.perf_counter()
        gate_prob = gate.predict(flat, num_threads=1)
        _timed(bs, "gate_predict", t)
        keep = gate_prob >= bs_meta["gate"]["threshold"]
        rows = flat[keep]
        bs["pixels_total"] = bs.get("pixels_total", 0) + int(flat.shape[0])
        bs["pixels_after_gate"] = bs.get("pixels_after_gate", 0) + int(keep.sum())
        t = time.perf_counter()
        p1 = bs_models[0].predict(rows, num_threads=1)
        _timed(bs, "ge1_after_gate", t)
        burned = rows[p1 >= bs_meta["burn_threshold"]]
        t = time.perf_counter()
        for m in bs_models[1:]:
            m.predict(burned, num_threads=1)
        _timed(bs, "ge2_ge3_on_burned", t)
        bs["pixels_burned"] = bs.get("pixels_burned", 0) + int(len(burned))
        t = time.perf_counter()
        for m in bs_models[1:]:
            m.predict(flat, num_threads=1)
        _timed(bs, "ge2_ge3_full_pixels", t)

    def per_chip(d: dict, n: int) -> dict:
        return {k: (round(v / n, 3) if isinstance(v, float) else v) for k, v in d.items()}

    return {"af_per_chip_seconds": per_chip(af, len(af_ids)), "bs_per_chip_seconds": per_chip(bs, len(bs_ids)), "chips_profiled": {"af": len(af_ids), "bs": len(bs_ids)}}
