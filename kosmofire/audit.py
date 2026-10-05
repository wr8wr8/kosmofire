import json
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import kosmofire
from .chipio import ChipStore
from .validation import af_groups, bs_groups

FORBIDDEN_COLUMNS = ("n_fire_px", "burn_area_ha", "sev1_px", "sev2_px", "sev3_px", "landcover_top")
DATE_TOKENS = ("date_pre", "date_post", "s1_date", "acq_datetime", "day_of_year", "doy")
MODEL_SOURCES = (
    "features_af.py", "features_bs.py", "af_data.py", "bs_data.py", "af_train.py", "bs_train.py",
    "inference_pipeline.py", "dozier.py", "unmixing.py", "ordinal.py", "bs_model.py", "bs_gate.py", "landcover.py",
)


def fold_overlap_report(train_dir: Path, folds: int = 5) -> dict:
    store = ChipStore(train_dir)
    meta = store.meta()
    report = {}
    for kind, group_fn in (("af", af_groups), ("bs", bs_groups)):
        ids = store.chip_ids(kind)
        frame = meta[meta["kind"] == kind]
        groups = group_fn(frame, ids)
        overlaps, val_sizes = [], []
        for train_idx, val_idx in GroupKFold(n_splits=folds).split(np.arange(len(ids)), groups=groups):
            overlap = set(groups[train_idx]) & set(groups[val_idx])
            overlaps.append(len(overlap))
            val_sizes.append(len(val_idx))
            assert not overlap, f"{kind}: {len(overlap)} groups shared between train and validation folds"
            assert not (set(train_idx) & set(val_idx))
        report[kind] = {"group_count": int(len(set(groups))), "overlap_per_fold": overlaps, "validation_chips_per_fold": val_sizes}
    bs_frame = meta[meta["kind"] == "bs"]
    report["bs_unique_fire_events"] = int(bs_frame["fire_event_id"].nunique())
    report["af_fire_event_id_available"] = bool(meta[meta["kind"] == "af"]["fire_event_id"].notna().any())
    return report


def grep_forbidden_columns() -> dict:
    root = Path(kosmofire.__file__).parent
    hits = {"forbidden_columns": {}, "date_columns": {}}
    for name in MODEL_SOURCES:
        text = (root / name).read_text()
        for token in FORBIDDEN_COLUMNS:
            if re.search(rf"\b{token}\b", text):
                hits["forbidden_columns"].setdefault(name, []).append(token)
        for token in DATE_TOKENS:
            if re.search(rf"\b{token}\b", text):
                hits["date_columns"].setdefault(name, []).append(token)
    return hits


def af_confusion(oof: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    pred = oof >= threshold
    positives = truth == 1
    tp = int(np.count_nonzero(pred & positives))
    fp = int(np.count_nonzero(pred & ~positives))
    fn = int(np.count_nonzero(~pred & positives))
    return {
        "tp": tp, "fp": fp, "fn": fn, "positives": int(positives.sum()),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def single_feature_rule_baseline(x, truth: np.ndarray, valid: np.ndarray, feature_index: int, thresholds) -> dict:
    best = {"f1": 0.0}
    values = np.concatenate([np.asarray(x[i])[feature_index][valid[i]] for i in range(truth.shape[0])])
    labels = np.concatenate([truth[i][valid[i]] for i in range(truth.shape[0])]) == 1
    for t in thresholds:
        pred = values > t
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return best


def run_audit(train_dir: Path, work_dir: Path, model_dir: Path, tag: str = "_oh") -> dict:
    from .features_af import af_feature_names

    meta = json.loads((model_dir / "af_meta.json").read_text())
    oof = np.load(work_dir / f"af_oof{tag}.npy")
    truth = np.load(work_dir / "af_truth.npy")
    valid = np.load(work_dir / "af_valid.npy")
    names = af_feature_names(tuple(meta.get("groups", ())))
    x = np.load(work_dir / f"af_x{tag}.npy", mmap_mode="r")
    return {
        "af_confusion_at_threshold": af_confusion(oof, truth, meta["threshold"]),
        "folds": fold_overlap_report(train_dir),
        "source_grep": grep_forbidden_columns(),
        "single_feature_baselines": {
            "delta_mir_tir": single_feature_rule_baseline(x, truth, valid, names.index("delta_mir_tir"), np.arange(5, 40, 1.0)),
            "i4": single_feature_rule_baseline(x, truth, valid, names.index("I4"), np.arange(300, 367, 2.0)),
        },
    }
