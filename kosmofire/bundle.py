import hashlib
import json
from pathlib import Path

import lightgbm
import numpy
import scipy
import sklearn

from .features_af import af_feature_names
from .features_bs import bs_feature_names

BUNDLE_NAME = "bundle.json"
SCHEMA_VERSION = 1
FILES = ("af_lgbm.txt", "af_meta.json", "bs_ge1.txt", "bs_ge2.txt", "bs_ge3.txt", "bs_meta.json", "bs_gate.txt", "bs_preproc.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _components(model_dir: Path) -> dict:
    af_meta = json.loads((model_dir / "af_meta.json").read_text())
    bs_meta = json.loads((model_dir / "bs_meta.json").read_text())
    return {
        "af": {
            "groups": af_meta.get("groups", []),
            "threshold": af_meta["threshold"],
            "prefilter": af_meta.get("prefilter"),
            "min_component": af_meta.get("min_component", 0),
            "feature_hash": _names_hash(af_feature_names(tuple(af_meta.get("groups", ())))),
        },
        "bs": {
            "groups": bs_meta.get("groups", []),
            "burn_threshold": bs_meta["burn_threshold"],
            "gate": bs_meta.get("gate"),
            "min_component": bs_meta.get("min_component", 0),
            "feature_hash": _names_hash(bs_feature_names(tuple(bs_meta.get("groups", ())))),
        },
    }


def build_bundle(model_dir: Path, extra: dict | None = None) -> dict:
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "files": {name: _sha256(model_dir / name) for name in FILES},
        "components": _components(model_dir),
        "libraries": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "extra": extra or {},
    }
    (model_dir / BUNDLE_NAME).write_text(json.dumps(bundle, indent=2, sort_keys=True))
    return bundle


def verify_bundle(model_dir: Path) -> list[str]:
    path = model_dir / BUNDLE_NAME
    if not path.exists():
        return [f"{BUNDLE_NAME} is missing in {model_dir}"]
    bundle = json.loads(path.read_text())
    problems = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"bundle schema {bundle.get('schema_version')} != {SCHEMA_VERSION}")
    for name, digest in bundle["files"].items():
        target = model_dir / name
        if not target.exists():
            problems.append(f"{name} is missing")
        elif _sha256(target) != digest:
            problems.append(f"{name} does not match the bundle checksum")
    current = _components(model_dir)
    for kind in ("af", "bs"):
        if current[kind]["feature_hash"] != bundle["components"][kind]["feature_hash"]:
            problems.append(f"{kind}: feature order in code differs from the bundle")
    if bundle["libraries"]["lightgbm"] != lightgbm.__version__:
        problems.append(f"lightgbm {lightgbm.__version__} differs from bundle {bundle['libraries']['lightgbm']}")
    return problems
