import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .af_data import build_af_cache
from .chipio import ChipStore
from .config import SEED
from .features_af import DEFAULT_GROUPS, af_feature_names
from .lgbm_utils import fit_binary
from .metrics import Counts, f1_from_counts
from .postprocess import remove_small_components
from .validation import af_groups

logger = logging.getLogger("train_af")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not serializable: {type(value)}")

RANDOM_NEGATIVE_RATE = 0.02
HARD_NEGATIVE_Z = 2.0


def _sample_rows(x_chip: np.ndarray, valid: np.ndarray, truth: np.ndarray, names: list[str], rng: np.random.Generator):
    z_idx = names.index("i4_z_7")
    d_idx = names.index("delta_mir_tir")
    positive = valid & (truth == 1)
    hard = valid & (truth == 0) & ((x_chip[z_idx] > HARD_NEGATIVE_Z) | (x_chip[d_idx] > 12.0))
    random_neg = valid & (truth == 0) & (rng.random(valid.shape) < RANDOM_NEGATIVE_RATE)
    keep = positive | hard | random_neg
    rows = x_chip[:, keep].T
    labels = positive[keep].astype(np.float32)
    return rows, labels


def _gather(x: np.ndarray, valid: np.ndarray, truth: np.ndarray, indices: np.ndarray, names: list[str], rng: np.random.Generator):
    row_blocks, label_blocks = [], []
    for i in indices:
        rows, labels = _sample_rows(np.asarray(x[i]), valid[i], truth[i], names, rng)
        row_blocks.append(rows)
        label_blocks.append(labels)
    return np.concatenate(row_blocks), np.concatenate(label_blocks)


def _predict_chips(model, x: np.ndarray, valid: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.zeros((len(indices), 256, 256), dtype=np.float32)
    for j, i in enumerate(indices):
        chip = np.asarray(x[i])
        prob = model.predict(chip.reshape(chip.shape[0], -1).T, num_threads=4)
        out[j] = prob.reshape(256, 256) * valid[i]
    return out


def best_threshold(prob: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    positives = truth == 1
    for t in grid:
        pred = prob >= t
        counts = Counts()
        counts.tp = int(np.count_nonzero(pred & positives))
        counts.fp = int(np.count_nonzero(pred & ~positives))
        counts.fn = int(np.count_nonzero(~pred & positives))
        f1 = f1_from_counts(counts)
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def train_af(train_dir: Path, work_dir: Path, model_dir: Path, rounds: int = 500, folds: int = 5, groups: tuple[str, ...] = DEFAULT_GROUPS, tag: str = "") -> dict:
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("af")
    meta = store.meta()
    names = af_feature_names(groups)

    cache_name = f"af_x{tag}.npy"
    if not (work_dir / cache_name).exists():
        build_af_cache(store, chip_ids, work_dir, groups, cache_name)

    x = np.load(work_dir / cache_name, mmap_mode="r")
    valid = np.load(work_dir / "af_valid.npy")
    truth = np.load(work_dir / "af_truth.npy")

    fold_groups = af_groups(meta[meta["kind"] == "af"], chip_ids)
    oof = np.zeros(truth.shape, dtype=np.float32)
    rng = np.random.default_rng(SEED)

    for fold, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(chip_ids)), groups=fold_groups)):
        xs, ys = _gather(x, valid, truth, tr, names, rng)
        model = fit_binary(xs, ys, rounds, names)
        oof[va] = _predict_chips(model, x, valid, va)
        logger.info("fold %d rows=%d pos=%d", fold, len(ys), int(ys.sum()))

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    threshold, oof_f1 = best_threshold(oof, truth, grid)
    logger.info("oof threshold=%.2f f1=%.4f", threshold, oof_f1)

    all_idx = np.arange(len(chip_ids))
    xs, ys = _gather(x, valid, truth, all_idx, names, rng)
    final = fit_binary(xs, ys, rounds, names)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / f"af_oof{tag}.npy", oof)
    final.save_model(str(model_dir / f"af_lgbm{tag}.txt"))
    report = {"threshold": threshold, "oof_f1": oof_f1, "rounds": rounds, "groups": list(groups), "features": names}
    (model_dir / f"af_meta{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    return report


MIN_COMPONENT_GRID = (0, 2, 3, 5, 8)


def tune_af_postprocess(work_dir: Path, model_dir: Path, tag: str = "") -> dict:
    oof = np.load(work_dir / f"af_oof{tag}.npy")
    truth = np.load(work_dir / "af_truth.npy")
    meta_path = model_dir / f"af_meta{tag}.json"
    meta = json.loads(meta_path.read_text())
    threshold = meta["threshold"]
    results = {}
    for size in MIN_COMPONENT_GRID:
        counts = Counts()
        for i in range(oof.shape[0]):
            pred = remove_small_components(oof[i] >= threshold, size)
            counts.add(truth[i] == 1, pred)
        results[size] = f1_from_counts(counts)
    best = max(results, key=results.get)
    meta["min_component"] = int(best)
    meta["oof_f1_after_postprocess"] = results[best]
    meta["postprocess_grid"] = {str(k): v for k, v in results.items()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
