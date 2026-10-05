import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .bs_data import BSParams, fit_endmembers, fit_kndvi_sigma, load_bs_features, sample_bs_training_rows
from .bs_model import fit_cascade, predict_cascade, save_models
from .chipio import ChipStore, read_mask
from .config import SEED
from .decision import decide_severity
from .features_bs import DEFAULT_GROUPS, bs_feature_names
from .lgbm_utils import fit_binary
from .metrics import ScoreAccumulator
from .validation import bs_groups

logger = logging.getLogger("train_bs")

BURN_THRESHOLDS = tuple(np.round(np.arange(0.25, 0.71, 0.05), 2))
GATE_THRESHOLDS = (None, 0.02, 0.05, 0.1)
MIN_COMPONENTS = (0, 10, 25, 50, 100)
GATE_ROUNDS = 60
GATE_LEAVES = 15


def _oof_path(work_dir: Path, tag: str, index: int) -> Path:
    return work_dir / f"bs_oof{tag}" / f"{index:04d}.npz"


def load_preprocessing(model_dir: Path, groups: tuple[str, ...], tag: str = "") -> BSParams:
    data = json.loads((model_dir / f"bs_preproc{tag}.json").read_text())
    endmembers = np.array(data["endmembers"], dtype=np.float64) if data["endmembers"] is not None else None
    return BSParams(groups=groups, endmembers=endmembers, kndvi_sigma=data["kndvi_sigma"])


def fold_params(paths, train_idx, groups, sigma) -> BSParams:
    endmembers = fit_endmembers([paths[i] for i in train_idx], SEED) if "unmix" in groups else None
    return BSParams(groups=groups, endmembers=endmembers, kndvi_sigma=sigma)


def run_cross_validation(paths, group_ids, names, groups, sigma, rounds, folds, monotone, weighted, work_dir: Path, tag: str) -> None:
    (work_dir / f"bs_oof{tag}").mkdir(parents=True, exist_ok=True)
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(paths)), groups=group_ids)):
        params = fold_params(paths, tr, groups, sigma)
        rows, labels, weights, _ = sample_bs_training_rows([paths[i] for i in tr], SEED + 1000 * fold, params)
        sample_weight = weights if weighted else None
        full = fit_cascade(rows, labels, rounds, names, monotone, sample_weight)
        gate = fit_binary(rows, (labels >= 1).astype(np.float32), GATE_ROUNDS, names, weight=sample_weight, num_leaves=GATE_LEAVES)
        for i in va:
            features, _ = load_bs_features(paths[i], params)
            flat = features.reshape(features.shape[0], -1).T
            payload = {
                "full": predict_cascade(full, features).astype(np.float16),
                "gate": gate.predict(flat, num_threads=4).reshape(features.shape[1:]).astype(np.float16),
            }
            np.savez(_oof_path(work_dir, tag, i), **payload)
        logger.info("fold %d done rows=%d", fold, len(labels))


def load_oof(work_dir: Path, tag: str, index: int) -> dict[str, np.ndarray]:
    with np.load(_oof_path(work_dir, tag, index)) as data:
        return {k: data[k].astype(np.float32) for k in data.files}


def decode_oof(oof: dict, burn_threshold: float, gate_threshold: float | None, min_component: int) -> np.ndarray:
    keep = None if gate_threshold is None else oof["gate"] >= gate_threshold
    probs = oof["full"]
    return decide_severity(probs[0], probs[1], probs[2], keep, burn_threshold, min_component)


def score_oof(work_dir: Path, tag: str, truths, burn_threshold: float, gate_threshold, min_component: int):
    acc = ScoreAccumulator()
    for i, truth in enumerate(truths):
        acc.add_bs(truth, decode_oof(load_oof(work_dir, tag, i), burn_threshold, gate_threshold, min_component))
    return acc.result()


def _value(result) -> float:
    return 0.35 * result.iou_burn + 0.30 * result.miou_sev


def select_postprocessing(work_dir: Path, tag: str, truths) -> dict:
    best = None
    for threshold in BURN_THRESHOLDS:
        result = score_oof(work_dir, tag, truths, float(threshold), None, 0)
        logger.info("thr=%.2f burn_iou=%.4f miou=%.4f", threshold, result.iou_burn, result.miou_sev)
        if best is None or _value(result) > best[0]:
            best = (_value(result), float(threshold))
    threshold = best[1]
    best_size = (0, -1.0)
    for size in MIN_COMPONENTS:
        result = score_oof(work_dir, tag, truths, threshold, None, size)
        logger.info("min_component=%d value=%.4f", size, _value(result))
        if _value(result) > best_size[1]:
            best_size = (size, _value(result))
    reference = score_oof(work_dir, tag, truths, threshold, None, best_size[0])
    gate_table = {}
    chosen_gate = None
    for gate in GATE_THRESHOLDS:
        result = score_oof(work_dir, tag, truths, threshold, gate, best_size[0])
        gate_table[str(gate)] = {"iou_burn": result.iou_burn, "miou_sev": result.miou_sev}
        logger.info("gate=%s iou_burn=%.4f miou=%.4f", gate, result.iou_burn, result.miou_sev)
    for gate in sorted(g for g in GATE_THRESHOLDS if g is not None):
        entry = gate_table[str(gate)]
        if _value(reference) - (0.35 * entry["iou_burn"] + 0.30 * entry["miou_sev"]) <= 0.002:
            chosen_gate = gate
    final = score_oof(work_dir, tag, truths, threshold, chosen_gate, best_size[0])
    return {
        "burn_threshold": threshold,
        "min_component": best_size[0],
        "gate": {"threshold": chosen_gate} if chosen_gate is not None else None,
        "gate_table": gate_table,
        "oof_iou_burn": final.iou_burn,
        "oof_miou_sev": final.miou_sev,
        "oof_iou_by_class": list(final.iou_by_class),
        "oof_without_gate": {"iou_burn": reference.iou_burn, "miou_sev": reference.miou_sev},
    }


def train_bs(
    train_dir: Path,
    work_dir: Path,
    model_dir: Path,
    rounds: int = 400,
    folds: int = 5,
    groups: tuple[str, ...] = DEFAULT_GROUPS,
    tag: str = "",
    monotone: bool = True,
    weighted: bool = True,
    skip_cv: bool = False,
) -> dict:
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("bs")
    meta = store.meta()
    names = bs_feature_names(groups)
    paths = [store.files(cid).paths for cid in chip_ids]
    group_ids = bs_groups(meta[meta["kind"] == "bs"], chip_ids)
    sigma = fit_kndvi_sigma(paths) if "indices2" in groups else None

    if not skip_cv:
        run_cross_validation(paths, group_ids, names, groups, sigma, rounds, folds, monotone, weighted, work_dir, tag)
    truths = [read_mask(p["mask"]).astype(np.uint8) for p in paths]
    selection = select_postprocessing(work_dir, tag, truths)

    final_params = fold_params(paths, np.arange(len(paths)), groups, sigma)
    rows, labels, weights, _ = sample_bs_training_rows(paths, SEED, final_params)
    sample_weight = weights if weighted else None
    full = fit_cascade(rows, labels, rounds, names, monotone, sample_weight)
    gate = fit_binary(rows, (labels >= 1).astype(np.float32), GATE_ROUNDS, names, weight=sample_weight, num_leaves=GATE_LEAVES)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_models(model_dir, tag, full, None)
    gate.save_model(str(model_dir / f"bs_gate{tag}.txt"))
    save_preprocessing_tagged(model_dir, tag, final_params.endmembers, sigma)
    report = {
        **selection,
        "rounds": rounds,
        "groups": list(groups),
        "monotone": monotone,
        "weighted_sampling": weighted,
        "features": names,
    }
    (model_dir / f"bs_meta{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    return report


def save_preprocessing_tagged(model_dir: Path, tag: str, endmembers, sigma) -> None:
    payload = {"endmembers": endmembers.tolist() if endmembers is not None else None, "kndvi_sigma": sigma}
    (model_dir / f"bs_preproc{tag}.json").write_text(json.dumps(payload))
