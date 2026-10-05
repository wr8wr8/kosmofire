import numpy as np

from kosmofire.decision import decide_severity
from kosmofire.ordinal import decode_severity
from kosmofire.postprocess import clean_severity


def test_gate_then_clean_differs_from_clean_then_gate():
    p = np.zeros((40, 40))
    p[5:15, 5:15] = 0.9
    keep = np.zeros((40, 40), dtype=bool)
    keep[5:9, 5:10] = True
    keep[11:15, 10:15] = True
    q = np.stack([p, p * 0.5, p * 0.1])
    gate_first = decide_severity(q[0], q[1], q[2], keep, 0.5, 30)
    severity = decode_severity(q[0], q[1], q[2], 0.5)
    clean_first = np.where(keep, clean_severity(severity, 30), 0)
    assert int((gate_first > 0).sum()) != int((clean_first > 0).sum())


def test_evaluator_and_inference_call_the_same_function():
    import inspect

    import kosmofire.bs_train as train
    import kosmofire.inference_pipeline as inference

    assert "decide_severity" in inspect.getsource(train)
    assert "decide_severity" in inspect.getsource(inference)
