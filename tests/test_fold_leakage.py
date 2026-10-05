import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from kosmofire.audit import FORBIDDEN_COLUMNS, grep_forbidden_columns
from kosmofire.validation import af_groups, bs_groups


def _meta(n=120, kind="af"):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "chip_id": [f"{kind.upper()}_tr_{i:06d}" for i in range(n)],
            "kind": kind,
            "epsg": rng.choice([32637, 32638], n),
            "x_min": rng.uniform(300000, 700000, n),
            "y_min": rng.uniform(5000000, 5600000, n),
            "acq_datetime": pd.to_datetime(rng.choice(pd.date_range("2019-04-01", "2024-10-01", freq="D"), n), utc=True).astype(str),
            "fire_event_id": [f"FE{i % 40:05d}" for i in range(n)],
        }
    )
    return frame


def test_af_folds_share_no_groups():
    frame = _meta()
    groups = af_groups(frame, list(frame["chip_id"]))
    for tr, va in GroupKFold(n_splits=5).split(np.arange(len(frame)), groups=groups):
        assert len(set(groups[tr]) & set(groups[va])) == 0


def test_bs_chips_of_one_fire_event_stay_in_one_fold():
    frame = _meta(kind="bs")
    groups = bs_groups(frame, list(frame["chip_id"]))
    events = frame["fire_event_id"].to_numpy()
    for tr, va in GroupKFold(n_splits=5).split(np.arange(len(frame)), groups=groups):
        assert len(set(events[tr]) & set(events[va])) == 0


def test_model_sources_do_not_reference_leaky_or_date_columns():
    hits = grep_forbidden_columns()
    assert hits["forbidden_columns"] == {}
    assert hits["date_columns"] == {}
    assert "n_fire_px" in FORBIDDEN_COLUMNS


def test_blank_and_nan_event_ids_fall_back_to_spatial_blocks_and_numeric_ids_are_kept():
    frame = _meta(n=12, kind="bs")
    frame["fire_event_id"] = ["", None, np.nan, "nan", "  ", 7, 7, "FE1", "FE1", "FE2", "", ""]
    groups = bs_groups(frame, list(frame["chip_id"]))
    assert groups[5] == groups[6]
    assert groups[7] == groups[8]
    assert groups[5] != groups[7]
    assert len({groups[0], groups[1]}) >= 1
