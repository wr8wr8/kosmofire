import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .bs_data import load_bs_features, sample_bs_training_rows
from .bs_train import decode_oof, load_oof
from .chipio import ChipStore, read_mask
from .config import SEED
from .features_bs import bs_feature_names
from .lgbm_utils import fit_binary
from .metrics import ScoreAccumulator
from .unmixing import load_endmembers
from .validation import bs_groups

logger = logging.getLogger("bs_gate")

GATE_ROUNDS = 60
GATE_LEAVES = 15
GATE_THRESHOLDS = (0.003, 0.01, 0.02, 0.05, 0.1)


def train_bs_gate(train_dir: Path, work_dir: Path, model_dir: Path, oof_tag: str = "_v2", folds: int = 5) -> dict:
    meta = json.loads((model_dir / "bs_meta.json").read_text())
    groups = tuple(meta["groups"])
    names = bs_feature_names(groups)
    endmembers = load_endmembers(model_dir / "bs_endmembers.json") if "unmix" in groups else None
    store = ChipStore(train_dir)
    chip_ids = store.chip_ids("bs")
    frame = store.meta()
    group_ids = bs_groups(frame[frame["kind"] == "bs"], chip_ids)
    paths = [store.files(c).paths for c in chip_ids]

    rows, labels, owners = sample_bs_training_rows(paths, SEED, groups, endmembers)
    burned = (labels >= 1).astype(np.float32)
    gate_prob = {}
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(np.arange(len(paths)), groups=group_ids)):
        selected = np.isin(owners, tr)
        model = fit_binary(rows[selected], burned[selected], GATE_ROUNDS, names, num_leaves=GATE_LEAVES)
        for i in va:
            features, _ = load_bs_features(paths[i], groups, endmembers)
            flat = features.reshape(features.shape[0], -1).T
            gate_prob[i] = model.predict(flat, num_threads=4).reshape(features.shape[1:])
        logger.info("gate fold %d done", fold)

    truths = [read_mask(p["mask"]).astype(np.uint8) for p in paths]
    results = {}
    for threshold in GATE_THRESHOLDS:
        acc = ScoreAccumulator()
        kept = total = 0
        for i, truth in enumerate(truths):
            pred = decode_oof(load_oof(work_dir, oof_tag, i), meta["burn_threshold"], 0.0, int(meta.get("min_component", 0)))
            keep = gate_prob[i] >= threshold
            kept += int(keep.sum())
            total += keep.size
            acc.add_bs(truth, np.where(keep, pred, 0))
        r = acc.result()
        results[str(threshold)] = {"kept": kept / total, "iou_burn": r.iou_burn, "miou_sev": r.miou_sev}
        logger.info("gate %.3f kept %.3f iou_burn %.4f miou %.4f", threshold, kept / total, r.iou_burn, r.miou_sev)

    final = fit_binary(rows, burned, GATE_ROUNDS, names, num_leaves=GATE_LEAVES)
    final.save_model(str(model_dir / "bs_gate.txt"))
    (work_dir / "bs_gate_report.json").write_text(json.dumps(results, indent=2))
    return results
