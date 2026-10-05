import json
import subprocess
import sys
from pathlib import Path

from kosmofire.bundle import verify_bundle
from tests.dataset_factory import build_test_dataset, build_test_models

ROOT = Path(__file__).resolve().parents[1]


def _run(data, models, output, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--data-dir", str(data), "--output", str(output), "--model-dir", str(models), "--workers", "2", "--skip-bundle-check", *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _corrupt_one_bs_chip(data: Path) -> None:
    target = next((data / "bs" / "sentinel2_pre").glob("BS_te_000001_*.tif"))
    target.write_bytes(b"not a raster")


def test_failed_chip_stops_run_by_default_and_is_listed_in_manifest(tmp_path):
    data = build_test_dataset(tmp_path / "data")
    models = build_test_models(tmp_path / "models")
    _corrupt_one_bs_chip(data)
    out = tmp_path / "submission.csv"
    result = _run(data, models, out)
    assert result.returncode == 2
    assert not out.exists()
    manifest = json.loads((tmp_path / "submission.csv.manifest.json").read_text())
    assert manifest["chips_failed"] == 1
    assert "BS_te_000001" in manifest["failures"]


def test_explicit_empty_policy_writes_file_but_reports_failure(tmp_path):
    data = build_test_dataset(tmp_path / "data")
    models = build_test_models(tmp_path / "models")
    _corrupt_one_bs_chip(data)
    out = tmp_path / "submission.csv"
    result = _run(data, models, out, "--on-error", "empty")
    assert result.returncode == 0
    assert "warning" in result.stderr
    manifest = json.loads((tmp_path / "submission.csv.manifest.json").read_text())
    assert manifest["chips_failed"] == 1 and manifest["on_error"] == "empty"


def test_bundle_detects_tampered_weights(tmp_path):
    from kosmofire.bundle import build_bundle

    models = build_test_models(tmp_path / "models")
    (models / "af_meta.json").write_text(json.dumps({"threshold": 0.5, "groups": []}))
    for name in ("bs_ge1.txt", "bs_ge2.txt", "bs_ge3.txt", "bs_gate.txt"):
        assert (models / name).exists()
    build_bundle(models)
    assert verify_bundle(models) == []
    (models / "bs_ge1.txt").write_text((models / "bs_ge1.txt").read_text() + "\n")
    assert any("bs_ge1.txt" in p for p in verify_bundle(models))


def test_missing_bundle_is_reported(tmp_path):
    assert verify_bundle(tmp_path)
