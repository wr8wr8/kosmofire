import json
import sys
import traceback
from pathlib import Path

import numpy as np

from .af_data import _compute_chip
from .bs_data import load_bs_features
from .bs_model import load_models
from .bs_train import load_preprocessing
from .chipio import ChipStore
from .config import DEFAULT_MODEL_DIR
from .decision import decide_severity
from .features_af import af_feature_names
from .features_bs import bs_feature_names
from .lgbm_utils import load_booster
from .parallel import default_workers, parallel_map
from .postprocess import remove_small_components


class ChipFailure(RuntimeError):
    pass


def _chunks(items: list, parts: int) -> list[list]:
    parts = max(1, min(parts, len(items)))
    return [items[i::parts] for i in range(parts)]


def _af_candidates(features: np.ndarray, names: list[str], prefilter: dict | None) -> np.ndarray:
    if not prefilter:
        return np.ones(features.shape[1:], dtype=bool)
    return (features[names.index("i4_minus_bg_7")] > prefilter["i4_minus_bg_7"]) | (features[names.index("delta_mir_tir")] > prefilter["delta_mir_tir"])


def _af_worker(args):
    jobs, model_path, meta, on_error = args
    model = load_booster(Path(model_path))
    groups = tuple(meta.get("groups", ()))
    names = af_feature_names(groups)
    masks, failures = {}, {}
    for chip_id, paths in jobs:
        try:
            features, valid, _ = _compute_chip((paths, groups))
            candidates = _af_candidates(features, names, meta.get("prefilter")) & valid
            mask = np.zeros(valid.shape, dtype=bool)
            if candidates.any():
                prob = model.predict(np.ascontiguousarray(features[:, candidates].T), num_threads=1)
                mask[candidates] = prob >= meta["threshold"]
            masks[chip_id] = remove_small_components(mask, int(meta.get("min_component", 0))).astype(np.uint8)
        except Exception:
            failures[chip_id] = traceback.format_exc(limit=3)
            if on_error == "empty":
                masks[chip_id] = np.zeros((256, 256), dtype=np.uint8)
    return masks, failures


def _bs_worker(args):
    jobs, model_dir, meta, on_error = args
    model_dir = Path(model_dir)
    groups = tuple(meta.get("groups", ()))
    params = load_preprocessing(model_dir, groups)
    full, _ = load_models(model_dir, "", False)
    gate = load_booster(model_dir / "bs_gate.txt") if meta.get("gate") else None
    burn_threshold = meta["burn_threshold"]
    masks, failures = {}, {}
    for chip_id, paths in jobs:
        try:
            features, _ = load_bs_features(paths, params)
            shape = features.shape[1:]
            flat = np.ascontiguousarray(features.reshape(features.shape[0], -1).T)
            if gate is not None:
                keep = gate.predict(flat, num_threads=1) >= meta["gate"]["threshold"]
            else:
                keep = np.ones(flat.shape[0], dtype=bool)
            p1 = np.zeros(flat.shape[0])
            p2 = np.zeros(flat.shape[0])
            p3 = np.zeros(flat.shape[0])
            if keep.any():
                p1[keep] = full[0].predict(flat[keep], num_threads=1)
            burned = keep & (p1 >= burn_threshold)
            if burned.any():
                rows = flat[burned]
                p2[burned] = full[1].predict(rows, num_threads=1)
                p3[burned] = full[2].predict(rows, num_threads=1)
            masks[chip_id] = decide_severity(p1.reshape(shape), p2.reshape(shape), p3.reshape(shape), keep.reshape(shape), burn_threshold, int(meta.get("min_component", 0)))
        except Exception:
            failures[chip_id] = traceback.format_exc(limit=3)
            if on_error == "empty":
                masks[chip_id] = np.zeros((512, 512), dtype=np.uint8)
    return masks, failures


def _merge(parts: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    masks, failures = {}, {}
    for m, f in parts:
        masks.update(m)
        failures.update(f)
    return masks, failures


def predict_af(store: ChipStore, chip_ids: list[str], model_dir: Path = DEFAULT_MODEL_DIR, workers: int | None = None, on_error: str = "fail"):
    meta = json.loads((model_dir / "af_meta.json").read_text())
    jobs = [(c, store.files(c).paths) for c in chip_ids]
    args = [(chunk, str(model_dir / "af_lgbm.txt"), meta, on_error) for chunk in _chunks(jobs, workers or default_workers())]
    return _merge(parallel_map(_af_worker, args, workers=len(args)))


def predict_bs(store: ChipStore, chip_ids: list[str], model_dir: Path = DEFAULT_MODEL_DIR, workers: int | None = None, on_error: str = "fail"):
    meta = json.loads((model_dir / "bs_meta.json").read_text())
    jobs = [(c, store.files(c).paths) for c in chip_ids]
    args = [(chunk, str(model_dir), meta, on_error) for chunk in _chunks(jobs, workers or default_workers())]
    return _merge(parallel_map(_bs_worker, args, workers=len(args)))
