import json
import logging
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from .af_train import _gather
from .calibration import apply_isotonic, blend, fit_isotonic
from .chipio import ChipStore
from .config import NUM_THREADS, SEED
from .features_af import af_feature_names
from .metrics import Counts, f1_from_counts
from .validation import af_groups

logger = logging.getLogger("af_ensemble")

WEIGHT_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))
THRESHOLD_GRID = tuple(np.round(np.arange(0.05, 0.96, 0.02), 2))


def make_catboost(iterations: int = 400) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=7,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=NUM_THREADS,
        verbose=False,
        allow_writing_files=False,
    )


def _predict_chips(model, x, valid, indices: np.ndarray) -> np.ndarray:
    out = np.zeros((len(indices), 256, 256), dtype=np.float32)
    for j, i in enumerate(indices):
        chip = np.asarray(x[i])
        flat = chip.reshape(chip.shape[0], -1).T
        out[j] = model.predict_proba(flat)[:, 1].reshape(256, 256) * valid[i]
    return out


def af_f1(prob: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    positives = truth == 1
    counts = Counts()
    counts.tp = int(np.count_nonzero(pred & positives))
    counts.fp = int(np.count_nonzero(pred & ~positives))
    counts.fn = int(np.count_nonzero(~pred & positives))
    return f1_from_counts(counts)


def best_threshold_f1(prob: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    scored = [(af_f1(prob, truth, t), t) for t in THRESHOLD_GRID]
    f1, t = max(scored)
    return float(t), float(f1)


def cross_fitted_blend(p_lgbm: np.ndarray, p_cat: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict:
    halves = [np.flatnonzero(groups % 2 == 0), np.flatnonzero(groups % 2 == 1)]
    final_prob = np.zeros_like(p_lgbm)
    chosen = []
    for fit_idx, apply_idx in ((halves[0], halves[1]), (halves[1], halves[0])):
        iso_a = fit_isotonic(p_lgbm[fit_idx], truth[fit_idx])
        iso_b = fit_isotonic(p_cat[fit_idx], truth[fit_idx])
        ca, cb = apply_isotonic(iso_a, p_lgbm[fit_idx]), apply_isotonic(iso_b, p_cat[fit_idx])
        best = max(((best_threshold_f1(blend(ca, cb, w), truth[fit_idx])[1], w) for w in WEIGHT_GRID))
        weight = best[1]
        fitted_blend = blend(ca, cb, weight)
        threshold, _ = best_threshold_f1(fitted_blend, truth[fit_idx])
        final_prob[apply_idx] = blend(apply_isotonic(iso_a, p_lgbm[apply_idx]), apply_isotonic(iso_b, p_cat[apply_idx]), weight)
        chosen.append({"weight_lgbm": float(weight), "threshold": threshold})
    threshold = float(np.mean([c["threshold"] for c in chosen]))
    return {"prob": final_prob, "chosen": chosen, "threshold": threshold, "f1": af_f1(final_prob, truth, threshold)}


def run_af_ensemble(train_dir: Path, work_dir: Path, tag: str = "", folds: int = 5, iterations: int = 400, groups: tuple[str, ...] = ("onehot",)) -> dict:
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("af")
    meta = store.meta()
    names = af_feature_names(groups)
    x = np.load(work_dir / f"af_x{tag}.npy", mmap_mode="r")
    valid = np.load(work_dir / "af_valid.npy")
    truth = np.load(work_dir / "af_truth.npy")
    fold_groups = af_groups(meta[meta["kind"] == "af"], chip_ids)
    rng = np.random.default_rng(SEED)
    cat_oof = np.zeros(truth.shape, dtype=np.float32)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(chip_ids)), groups=fold_groups)):
        rows, labels = _gather(x, valid, truth, tr, names, rng)
        model = make_catboost(iterations)
        model.fit(rows, labels)
        cat_oof[va] = _predict_chips(model, x, valid, va)
        logger.info("catboost fold %d done", fold)
    np.save(work_dir / f"af_oof_cat{tag}.npy", cat_oof)
    lgbm_oof = np.load(work_dir / f"af_oof{tag}.npy")
    result = cross_fitted_blend(lgbm_oof, cat_oof, truth, fold_groups)
    report = {
        "f1_lgbm": best_threshold_f1(lgbm_oof, truth)[1],
        "f1_catboost": best_threshold_f1(cat_oof, truth)[1],
        "f1_blend_cross_fitted": result["f1"],
        "blend_choices": result["chosen"],
    }
    (work_dir / f"af_ensemble_report{tag}.json").write_text(json.dumps(report, indent=2))
    return report
