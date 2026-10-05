from kosmofire.features_af import af_feature_names
from kosmofire.features_bs import bs_feature_names

DATE_LIKE = ("date", "doy", "day_of", "season", "interval", "month", "year")


def test_no_feature_is_derived_from_acquisition_dates():
    for name in af_feature_names() + bs_feature_names():
        assert not any(token in name for token in DATE_LIKE), name
