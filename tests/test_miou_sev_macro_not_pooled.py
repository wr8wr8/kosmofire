import numpy as np

from kosmofire.metrics import ScoreAccumulator


def test_miou_is_mean_of_per_class_iou_not_pooled_counts():
    acc = ScoreAccumulator()
    truth = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 100), dtype=np.uint8)
    truth[0, :90] = 1
    pred[0, :90] = 1
    truth[1, :10] = 2
    pred[1, :5] = 2
    truth[2, :10] = 3
    pred[3, :10] = 3
    acc.add_bs(truth, pred)
    result = acc.result()
    ious = (90 / 90, 5 / 10, 0.0)
    assert result.iou_by_class == ious
    macro = sum(ious) / 3
    pooled_tp = 90 + 5
    pooled_fp = 0 + 0 + 10
    pooled_fn = 0 + 5 + 10
    pooled = pooled_tp / (pooled_tp + pooled_fp + pooled_fn)
    assert abs(result.miou_sev - macro) < 1e-12
    assert abs(result.miou_sev - pooled) > 0.05
