from pathlib import Path

import kosmofire
from kosmofire.features_af import af_feature_names
from kosmofire.features_bs import bs_feature_names

FORBIDDEN = ("n_fire_px", "burn_area_ha", "sev1_px", "sev2_px", "sev3_px", "landcover_top")
DATE_TOKENS = ("date_pre", "date_post", "day_of_year", "doy", "acq_datetime", "s1_date")
SOURCES = [
    "features_af.py",
    "features_bs.py",
    "inference_pipeline.py",
    "bs_data.py",
    "af_data.py",
    "submission.py",
]


def _source(name: str) -> str:
    return (Path(kosmofire.__file__).parent / name).read_text()


def test_feature_names_do_not_reference_leaky_columns():
    names = " ".join(af_feature_names() + bs_feature_names())
    assert not any(token in names for token in FORBIDDEN)


def test_inference_sources_never_touch_leaky_columns():
    for name in ("features_af.py", "features_bs.py", "inference_pipeline.py", "bs_data.py", "af_data.py"):
        text = _source(name)
        assert not any(token in text for token in FORBIDDEN), name


def test_inference_sources_never_use_date_columns():
    for name in ("features_af.py", "features_bs.py", "inference_pipeline.py", "bs_data.py", "af_data.py"):
        text = _source(name)
        assert not any(token in text for token in DATE_TOKENS), name
