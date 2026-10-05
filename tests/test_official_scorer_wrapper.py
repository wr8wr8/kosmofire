import numpy as np

from kosmofire.metrics import ScoreAccumulator


def _reference(truths, preds, classes):
    out = {}
    for k in classes:
        tp = sum(int(((t == k) & (p == k)).sum()) for t, p in zip(truths, preds))
        fp = sum(int(((t != k) & (p == k)).sum()) for t, p in zip(truths, preds))
        fn = sum(int(((t == k) & (p != k)).sum()) for t, p in zip(truths, preds))
        out[k] = (tp, fp, fn)
    return out


def test_micro_average_equals_direct_pooled_computation():
    rng = np.random.default_rng(3)
    truths = [rng.integers(0, 4, (32, 32)).astype(np.uint8) for _ in range(5)]
    preds = [np.where(rng.random((32, 32)) > 0.3, t, rng.integers(0, 4, (32, 32))).astype(np.uint8) for t in truths]

    acc = ScoreAccumulator()
    for t, p in zip(truths, preds):
        acc.add_bs(t, p)
    result = acc.result()

    ref = _reference(truths, preds, (1, 2, 3))
    ious = [tp / (tp + fp + fn) for tp, fp, fn in ref.values()]
    assert np.allclose(result.iou_by_class, ious)
    assert abs(result.miou_sev - np.mean(ious)) < 1e-12

    burned_t = [t >= 1 for t in truths]
    burned_p = [p >= 1 for p in preds]
    tp = sum(int((a & b).sum()) for a, b in zip(burned_t, burned_p))
    fp = sum(int((~a & b).sum()) for a, b in zip(burned_t, burned_p))
    fn = sum(int((a & ~b).sum()) for a, b in zip(burned_t, burned_p))
    assert abs(result.iou_burn - tp / (tp + fp + fn)) < 1e-12
