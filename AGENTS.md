# Agent Guide

## Project scope

KosmoFire is a Python remote-sensing and geospatial machine-learning project. It contains two raster classification paths: AF for active-fire pixels from VIIRS and BS for burned-area severity from before/after Sentinel-1 and Sentinel-2 data. It also contains submission utilities and a FastAPI service for prepared geospatial products.

## Snapshot limitations

The distributed snapshot includes AF weights only. BS weights, BS metadata, BS preprocessing parameters, `models/bundle.json`, training rasters, evaluation rasters, and prepared service GeoPackages are not included. Do not claim that full AF+BS inference or service-data generation has been verified unless those artifacts are supplied and the corresponding workflow succeeds.

## Source of truth

- `kosmofire/config.py` defines paths, seeds, chip dimensions, scales, and band names.
- `kosmofire/features_af.py` and `kosmofire/features_bs.py` define model feature names and ordering.
- `kosmofire/inference_pipeline.py` defines production prediction orchestration.
- `kosmofire/decision.py` and `kosmofire/ordinal.py` define severity decoding and postprocessing shared by training evaluation and inference.
- `kosmofire/bundle.py` defines model-bundle contents, checksums, and feature-order validation.

Keep these paths consistent. A feature-order change requires compatible model artifacts and a newly generated bundle.

## Data and leakage safeguards

- Do not use target masks, target-derived values, chip IDs, dates, or post-event labels as model features.
- Preserve group-based validation by event or geography; do not replace it with random pixel or chip splits.
- Fit any learned preprocessing or spectral reference values within each training fold. Do not derive them from validation labels or held-out target chips.
- Keep inference postprocessing equivalent to the path used for OOF scoring.
- Treat raster metadata and external datasets as inputs that require schema and validity checks.

## Code and artifact changes

- Keep Python code compatible with the versions and Python requirement in `pyproject.toml` and `requirements.txt`.
- Preserve deterministic seeds and explicit worker/thread limits unless a change is justified and documented.
- Keep source code free of inline comments unless the maintainer explicitly requests them.
- Do not silently convert failed-chip predictions into empty masks. The default inference policy is to fail; the explicit `--on-error empty` option must remain opt-in and manifest the affected chips.
- Do not commit private datasets, credentials, generated caches, large OOF arrays, or service GeoPackages without an explicit distribution need.
- Keep model weights, model metadata, preprocessing parameters, tags, feature order, and bundle checksums from the same training run together.

## Development commands

Create an environment and install core dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train with the default grouped-validation settings:

```bash
python train_af.py --train-dir data/raw/train
python train_bs.py --train-dir data/raw/train
```

After all required AF and BS artifacts exist:

```bash
python make_bundle.py
python inference.py --data-dir /path/to/test-data --output outputs/submission.csv
```

Run tests with:

```bash
pytest -q
```

Tests requiring external datasets or model artifacts may not be runnable from the source snapshot alone. Report exactly which inputs are missing instead of fabricating validation results.

## Documentation and user-facing claims

- Keep the README in English, Russian, then Chinese, in that order, unless the maintainer requests otherwise.
- State clearly when required models or datasets are absent.
- Do not invent benchmark scores, dataset provenance, deployment status, or supported features.
- Use the existing `assets/kosmofire-cover.jpg` artwork for the README banner; preserve its filename or update references when replacing it.
