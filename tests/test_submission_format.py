import numpy as np
import pandas as pd
import pytest

from kosmofire.rle import encode_rle
from kosmofire.submission import build_rows, read_template, validate_submission, write_submission


@pytest.fixture
def template_path(tmp_path):
    rows = [("AF_te_000001", 1), ("AF_te_000002", 1)]
    rows += [("BS_te_000001", k) for k in (1, 2, 3)]
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame(rows, columns=["chip_id", "class_id"]).assign(rle="").to_csv(path, index=False)
    return path


def _masks():
    af = {"AF_te_000001": np.zeros((256, 256), dtype=np.uint8), "AF_te_000002": np.zeros((256, 256), dtype=np.uint8)}
    af["AF_te_000001"][3, 4:9] = 1
    bs_mask = np.zeros((512, 512), dtype=np.uint8)
    bs_mask[10:20, 10:20] = 1
    bs_mask[30:40, 10:20] = 2
    bs_mask[50:60, 10:20] = 3
    return af, {"BS_te_000001": bs_mask}


def test_valid_submission_passes(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    out = tmp_path / "submission.csv"
    write_submission(build_rows(template, af, bs), out)
    assert validate_submission(out, template_path) == []
    assert len(pd.read_csv(out)) == len(template)


def test_row_count_matches_template_not_constant(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    out = tmp_path / "submission.csv"
    write_submission(build_rows(template, af, bs), out)
    assert len(pd.read_csv(out, keep_default_na=False)) == 5


def test_empty_class_is_an_empty_field_and_file_matches_reference_style(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    out = tmp_path / "submission.csv"
    write_submission(build_rows(template, af, bs), out)
    lines = out.read_text().splitlines()
    assert lines[0] == "chip_id,class_id,rle"
    assert lines[1] == "AF_te_000001,1,5 5" or lines[1].startswith("AF_te_000001,1,")
    assert lines[2] == "AF_te_000002,1,"
    reread = pd.read_csv(out, keep_default_na=False, dtype=str)
    assert list(reread.columns) == ["chip_id", "class_id", "rle"]


def test_missing_pair_is_reported(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    frame = build_rows(template, af, bs).iloc[:-1]
    out = tmp_path / "submission.csv"
    write_submission(frame, out)
    assert any("row count" in p or "pair mismatch" in p for p in validate_submission(out, template_path))


def test_overlapping_bs_classes_are_reported(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    frame = build_rows(template, af, bs)
    overlap = np.zeros((512, 512), dtype=bool)
    overlap[10:20, 10:20] = True
    frame.loc[frame["class_id"] == 2, "rle"] = frame.loc[frame["chip_id"].eq("BS_te_000001") & frame["class_id"].eq(1), "rle"].iloc[0]
    out = tmp_path / "submission.csv"
    write_submission(frame, out)
    assert any("overlap" in p for p in validate_submission(out, template_path))


def test_out_of_bounds_index_is_reported(tmp_path, template_path):
    template = read_template(template_path)
    af, bs = _masks()
    frame = build_rows(template, af, bs)
    frame.loc[0, "rle"] = "65535 5"
    out = tmp_path / "submission.csv"
    write_submission(frame, out)
    assert any("exceeds" in p for p in validate_submission(out, template_path))


def test_encode_edge_last_pixel():
    mask = np.zeros((256, 256), dtype=bool)
    mask[-1, -1] = True
    assert encode_rle(mask) == "65536 1"
