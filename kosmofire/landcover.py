import numpy as np

WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
UNKNOWN_INDEX = len(WORLDCOVER_CODES)


def landcover_index(codes: np.ndarray) -> np.ndarray:
    lookup = np.full(256, UNKNOWN_INDEX, dtype=np.float32)
    for index, code in enumerate(WORLDCOVER_CODES):
        lookup[code] = index
    clipped = np.clip(np.nan_to_num(codes, nan=0.0), 0, 255).astype(np.int64)
    return lookup[clipped]


def landcover_one_hot(codes: np.ndarray) -> np.ndarray:
    index = landcover_index(codes).astype(np.int64)
    return np.eye(UNKNOWN_INDEX + 1, dtype=np.float32)[index][..., : len(WORLDCOVER_CODES)].transpose(2, 0, 1)


def one_hot_names() -> list[str]:
    return [f"lc_{code}" for code in WORLDCOVER_CODES]
