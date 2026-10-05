import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .bs_train import decode_oof, load_oof
from .chipio import ChipStore, read_mask, read_raster
from .decision import decide_severity
from .features_af import af_feature_names
from .metrics import ScoreAccumulator

LANDCOVER_STRATA = {10: "лес", 20: "кустарник", 30: "степь", 40: "пашня", 60: "открытый грунт", 90: "пойма/болото"}
CLOUD_BINS = (("низкая (<2%)", 0.0, 0.02), ("средняя (2-10%)", 0.02, 0.10), ("высокая (>=10%)", 0.10, 1.01))
SIZE_BINS = ((1, 10), (10, 25), (25, 50), (50, 100), (100, 400), (400, 10**9))


def confusion_matrix(truths: list[np.ndarray], preds: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for t, p in zip(truths, preds):
        idx = t.astype(np.int64) * 4 + p.astype(np.int64)
        matrix += np.bincount(idx.ravel(), minlength=16).reshape(4, 4)
    return matrix


def confusion_summary(matrix: np.ndarray) -> dict:
    rows = matrix.sum(axis=1, keepdims=True)
    normalised = matrix / np.maximum(rows, 1)
    burned_truth = matrix[1:].sum(axis=1)
    adjacent = matrix[1, 2] + matrix[2, 1] + matrix[2, 3] + matrix[3, 2]
    far = matrix[1, 3] + matrix[3, 1]
    to_background = matrix[1:, 0].sum()
    from_background = matrix[0, 1:].sum()
    burned_errors = adjacent + far + to_background + from_background
    return {
        "matrix": matrix.tolist(),
        "row_normalised": np.round(normalised, 4).tolist(),
        "adjacent_severity_confusions_px": int(adjacent),
        "far_severity_confusions_px": int(far),
        "burned_missed_as_background_px": int(to_background),
        "background_predicted_as_burn_px": int(from_background),
        "share_of_all_errors": {
            "adjacent_1_2_and_2_3": float(adjacent / max(burned_errors, 1)),
            "far_1_3": float(far / max(burned_errors, 1)),
            "missed_to_background": float(to_background / max(burned_errors, 1)),
            "false_burn_from_background": float(from_background / max(burned_errors, 1)),
        },
        "true_burned_px": burned_truth.tolist(),
    }


def removed_component_analysis(oof_list: list[dict], truths: list[np.ndarray], burn_threshold: float, gate_threshold, min_component: int) -> dict:
    structure = np.ones((3, 3), dtype=bool)
    stats = {f"{lo}-{hi}": {"components": 0, "removed_tp_px": 0, "removed_fp_px": 0} for lo, hi in SIZE_BINS}
    for oof, truth in zip(oof_list, truths):
        keep = None if gate_threshold is None else oof["gate"] >= gate_threshold
        unfiltered = decide_severity(oof["full"][0], oof["full"][1], oof["full"][2], keep, burn_threshold, 0)
        labels, count = ndi.label(unfiltered >= 1, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        burned_truth = truth >= 1
        tp_per = np.bincount(labels.ravel(), weights=burned_truth.ravel(), minlength=count + 1)
        for component in range(1, count + 1):
            size = int(sizes[component])
            for lo, hi in SIZE_BINS:
                if lo <= size < hi:
                    entry = stats[f"{lo}-{hi}"]
                    entry["components"] += 1
                    if size < min_component:
                        entry["removed_tp_px"] += int(tp_per[component])
                        entry["removed_fp_px"] += int(size - tp_per[component])
                    break
    return stats


def gate_recall(oof_list: list[dict], truths: list[np.ndarray], gate_threshold, burn_threshold: float) -> dict:
    if gate_threshold is None:
        return {"gate": None}
    kept_by_class = np.zeros(4)
    total_by_class = np.zeros(4)
    pred_lost = 0
    pred_total = 0
    for oof, truth in zip(oof_list, truths):
        keep = oof["gate"] >= gate_threshold
        for k in range(4):
            mask = truth == k
            total_by_class[k] += mask.sum()
            kept_by_class[k] += (mask & keep).sum()
        raw = oof["full"][0] >= burn_threshold
        pred_total += int(raw.sum())
        pred_lost += int((raw & ~keep).sum())
    return {
        "gate_threshold": gate_threshold,
        "recall_of_true_burn_pixels": float(kept_by_class[1:].sum() / max(total_by_class[1:].sum(), 1)),
        "recall_by_severity": {str(k): float(kept_by_class[k] / max(total_by_class[k], 1)) for k in (1, 2, 3)},
        "fraction_of_background_pixels_kept": float(kept_by_class[0] / max(total_by_class[0], 1)),
        "raw_predicted_burn_pixels_removed_by_gate": float(pred_lost / max(pred_total, 1)),
    }


def evaluate(train_dir: Path, work_dir: Path, model_dir: Path, af_tag: str = "_oh", bs_tag: str = "_v3") -> dict:
    store = ChipStore(train_dir)
    af_ids = store.chip_ids("af")
    bs_ids = store.chip_ids("bs")
    meta = store.meta().set_index("chip_id")
    af_meta = json.loads((model_dir / "af_meta.json").read_text())
    bs_meta = json.loads((model_dir / f"bs_meta{bs_tag}.json").read_text())
    gate_threshold = (bs_meta.get("gate") or {}).get("threshold")

    af_prob = np.load(work_dir / f"af_oof{af_tag}.npy")
    af_truth = np.load(work_dir / "af_truth.npy")
    af_valid = np.load(work_dir / "af_valid.npy")
    af_x = np.load(work_dir / f"af_x{af_tag}.npy", mmap_mode="r")
    names = af_feature_names(tuple(af_meta.get("groups", ())))
    pre = af_meta.get("prefilter") or {"i4_minus_bg_7": -1e9, "delta_mir_tir": -1e9}
    i_bg, i_dm = names.index("i4_minus_bg_7"), names.index("delta_mir_tir")

    acc = ScoreAccumulator()
    af_tp = af_fp = af_fn = 0
    for i in range(len(af_ids)):
        chip = np.asarray(af_x[i])
        candidates = ((chip[i_bg] > pre["i4_minus_bg_7"]) | (chip[i_dm] > pre["delta_mir_tir"])) & af_valid[i]
        pred = ((af_prob[i] >= af_meta["threshold"]) & candidates).astype(np.uint8)
        acc.add_af(af_truth[i], pred)
        truth_pos = af_truth[i] == 1
        af_tp += int((pred.astype(bool) & truth_pos).sum())
        af_fp += int((pred.astype(bool) & ~truth_pos).sum())
        af_fn += int((~pred.astype(bool) & truth_pos).sum())

    oof_list = [load_oof(work_dir, bs_tag, i) for i in range(len(bs_ids))]
    truths = [read_mask(store.files(c).paths["mask"]).astype(np.uint8) for c in bs_ids]
    preds = [decode_oof(o, bs_meta["burn_threshold"], gate_threshold, int(bs_meta.get("min_component", 0))) for o in oof_list]
    for truth, pred in zip(truths, preds):
        acc.add_bs(truth, pred)
    result = acc.result()

    strata = {name: ScoreAccumulator() for name in LANDCOVER_STRATA.values()}
    cloud_acc = {label: ScoreAccumulator() for label, _, _ in CLOUD_BINS}
    cloud_counts = {label: 0 for label, _, _ in CLOUD_BINS}
    for i, cid in enumerate(bs_ids):
        landcover = read_raster(store.files(cid).paths["aux"])[2]
        for code, name in LANDCOVER_STRATA.items():
            region = landcover == code
            if region.any():
                strata[name].add_bs(np.where(region, truths[i], 0), np.where(region, preds[i], 0))
        frac = float(meta.loc[cid, "cloud_frac"])
        for label, lo, hi in CLOUD_BINS:
            if lo <= frac < hi:
                cloud_acc[label].add_bs(truths[i], preds[i])
                cloud_counts[label] += 1

    def block(a: ScoreAccumulator) -> dict:
        r = a.result()
        return {"iou_burn": r.iou_burn, "miou_sev": r.miou_sev, "iou_by_class": list(r.iou_by_class)}

    return {
        "score": result.score,
        "f1_af": result.f1_af,
        "af_counts": {"tp": af_tp, "fp": af_fp, "fn": af_fn},
        "iou_burn": result.iou_burn,
        "iou_by_class": list(result.iou_by_class),
        "miou_sev": result.miou_sev,
        "bs_selected_params": {"burn_threshold": bs_meta["burn_threshold"], "gate": bs_meta.get("gate"), "min_component": bs_meta.get("min_component")},
        "confusion": confusion_summary(confusion_matrix(truths, preds)),
        "by_landcover": {name: block(a) for name, a in strata.items() if a.burn.tp + a.burn.fn > 0},
        "by_cloud_fraction": {label: {"chips": cloud_counts[label], **block(cloud_acc[label])} for label, _, _ in CLOUD_BINS if cloud_counts[label]},
        "removed_components_by_size": removed_component_analysis(oof_list, truths, bs_meta["burn_threshold"], gate_threshold, int(bs_meta.get("min_component", 0))),
        "gate_recall": gate_recall(oof_list, truths, gate_threshold, bs_meta["burn_threshold"]),
    }
