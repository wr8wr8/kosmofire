import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).ravel()
    if not flat.any():
        return ""
    padded = np.concatenate(([False], flat, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2] + 1
    lengths = changes[1::2] - changes[0::2]
    return " ".join(f"{s} {l}" for s, l in zip(starts.tolist(), lengths.tolist()))


def decode_rle(rle: str, shape: tuple[int, int]) -> np.ndarray:
    total = shape[0] * shape[1]
    flat = np.zeros(total, dtype=bool)
    if rle is None or not str(rle).strip():
        return flat.reshape(shape)
    values = [int(v) for v in str(rle).split()]
    if len(values) % 2:
        raise ValueError("RLE must contain an even number of integers")
    previous_end = 0
    for start, length in zip(values[0::2], values[1::2]):
        if start < 1 or length < 1:
            raise ValueError("RLE runs must have positive start and length")
        if start <= previous_end:
            raise ValueError("RLE runs must be ascending and must not touch or overlap")
        end = start - 1 + length
        if end > total:
            raise ValueError("RLE run exceeds chip bounds")
        flat[start - 1 : end] = True
        previous_end = end
    return flat.reshape(shape)
