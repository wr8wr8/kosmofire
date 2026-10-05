from pathlib import Path

import numpy as np
import pandas as pd

from .config import AF_CHIP_SIZE, BS_CHIP_SIZE
from .rle import decode_rle, encode_rle


def read_template(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"chip_id": str}, keep_default_na=False)
    frame["class_id"] = frame["class_id"].astype(int)
    return frame[["chip_id", "class_id"]]


def chip_shape(chip_id: str) -> tuple[int, int]:
    size = AF_CHIP_SIZE if chip_id.startswith("AF_") else BS_CHIP_SIZE
    return size, size


def build_rows(template: pd.DataFrame, af_masks: dict[str, np.ndarray], bs_masks: dict[str, np.ndarray]) -> pd.DataFrame:
    records = []
    for chip_id, class_id in zip(template["chip_id"], template["class_id"]):
        if chip_id.startswith("AF_"):
            mask = af_masks[chip_id] == 1
        else:
            mask = bs_masks[chip_id] == class_id
        records.append((chip_id, int(class_id), encode_rle(mask)))
    return pd.DataFrame(records, columns=["chip_id", "class_id", "rle"])


def write_submission(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def validate_submission(submission_path: Path, template_path: Path) -> list[str]:
    problems: list[str] = []
    frame = pd.read_csv(submission_path, dtype={"chip_id": str, "rle": str}, keep_default_na=False)
    template = read_template(template_path)

    if list(frame.columns) != ["chip_id", "class_id", "rle"]:
        problems.append(f"unexpected columns {list(frame.columns)}")
        return problems
    if len(frame) != len(template):
        problems.append(f"row count {len(frame)} != {len(template)}")
    got = set(zip(frame["chip_id"], frame["class_id"].astype(int)))
    want = set(zip(template["chip_id"], template["class_id"]))
    if got != want:
        problems.append(f"pair mismatch: missing={len(want - got)} extra={len(got - want)}")
    if frame.duplicated(["chip_id", "class_id"]).any():
        problems.append("duplicate (chip_id, class_id) rows")

    by_chip: dict[str, list[np.ndarray]] = {}
    for chip_id, class_id, rle in zip(frame["chip_id"], frame["class_id"], frame["rle"]):
        try:
            mask = decode_rle(rle, chip_shape(chip_id))
        except ValueError as exc:
            problems.append(f"{chip_id}/{class_id}: {exc}")
            continue
        if chip_id.startswith("BS_"):
            by_chip.setdefault(chip_id, []).append(mask)
    for chip_id, masks in by_chip.items():
        if np.stack(masks).sum(axis=0).max() > 1:
            problems.append(f"{chip_id}: class masks overlap")
    return problems
