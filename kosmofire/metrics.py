from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreBreakdown:
    f1_af: float
    iou_burn: float
    miou_sev: float
    iou_by_class: tuple[float, float, float]
    score: float


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, truth: np.ndarray, pred: np.ndarray) -> None:
        truth = truth.astype(bool)
        pred = pred.astype(bool)
        self.tp += int(np.count_nonzero(truth & pred))
        self.fp += int(np.count_nonzero(~truth & pred))
        self.fn += int(np.count_nonzero(truth & ~pred))


def f1_from_counts(c: Counts) -> float:
    if c.tp + c.fp + c.fn == 0:
        return 1.0
    if c.tp == 0:
        return 0.0
    precision = c.tp / (c.tp + c.fp)
    recall = c.tp / (c.tp + c.fn)
    return 2 * precision * recall / (precision + recall)


def iou_from_counts(c: Counts) -> float:
    denominator = c.tp + c.fp + c.fn
    if denominator == 0:
        return 1.0
    return c.tp / denominator


class ScoreAccumulator:
    def __init__(self) -> None:
        self.af = Counts()
        self.burn = Counts()
        self.sev = {1: Counts(), 2: Counts(), 3: Counts()}

    def add_af(self, truth: np.ndarray, pred: np.ndarray) -> None:
        self.af.add(truth == 1, pred == 1)

    def add_bs(self, truth: np.ndarray, pred: np.ndarray) -> None:
        self.burn.add(truth >= 1, pred >= 1)
        for k in (1, 2, 3):
            self.sev[k].add(truth == k, pred == k)

    def result(self) -> ScoreBreakdown:
        f1 = f1_from_counts(self.af)
        iou_burn = iou_from_counts(self.burn)
        ious = tuple(iou_from_counts(self.sev[k]) for k in (1, 2, 3))
        miou = float(sum(ious) / 3)
        score = 0.35 * f1 + 0.35 * iou_burn + 0.30 * miou
        return ScoreBreakdown(f1, iou_burn, miou, ious, score)
