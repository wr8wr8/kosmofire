import numpy as np

from kosmofire.metrics import ScoreAccumulator


def test_class_absent_in_truth_and_prediction_scores_one():
    acc = ScoreAccumulator()
    acc.add_af(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    acc.add_bs(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
    result = acc.result()
    assert result.f1_af == 1.0
    assert result.iou_burn == 1.0
    assert result.score == 1.0


def test_class_present_in_truth_and_missing_in_prediction_scores_zero():
    acc = ScoreAccumulator()
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[0, 0] = 1
    acc.add_af(truth, np.zeros((4, 4), dtype=np.uint8))
    assert acc.result().f1_af == 0.0


def test_micro_average_pools_pixels_across_chips():
    acc = ScoreAccumulator()
    a_truth = np.zeros((10, 10), dtype=np.uint8)
    a_truth[0, :4] = 1
    a_pred = a_truth.copy()
    b_truth = np.zeros((10, 10), dtype=np.uint8)
    b_truth[0, :1] = 1
    b_pred = np.zeros((10, 10), dtype=np.uint8)
    acc.add_af(a_truth, a_pred)
    acc.add_af(b_truth, b_pred)
    expected_precision = 4 / 4
    expected_recall = 4 / 5
    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert abs(acc.result().f1_af - expected_f1) < 1e-12


def test_severity_miou_averages_three_classes():
    acc = ScoreAccumulator()
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[0, 0], truth[0, 1], truth[0, 2] = 1, 2, 3
    pred = truth.copy()
    pred[0, 2] = 2
    acc.add_bs(truth, pred)
    result = acc.result()
    assert result.iou_by_class == (1.0, 0.5, 0.0)
    assert abs(result.miou_sev - 0.5) < 1e-12
    assert result.iou_burn == 1.0
