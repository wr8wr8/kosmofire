import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kosmofire.nn_bs import SiameseUNet, prepare_inputs, seed_everything
from tests.synthetic import make_bs_chip


def _predict(seed: int) -> np.ndarray:
    seed_everything(seed, threads=1)
    model = SiameseUNet(base=8, depth=3)
    model.eval()
    pre, post, static = prepare_inputs(*make_bs_chip(size=64))
    with torch.no_grad():
        logits = model(torch.from_numpy(pre)[None], torch.from_numpy(post)[None], torch.from_numpy(static)[None])
    return logits.argmax(1).numpy()


def test_cpu_inference_is_deterministic_across_runs():
    assert np.array_equal(_predict(0), _predict(0))


def test_model_output_shape_matches_chip():
    seed_everything(0, threads=1)
    model = SiameseUNet(base=8, depth=3)
    pre, post, static = prepare_inputs(*make_bs_chip(size=64))
    out = model(torch.from_numpy(pre)[None], torch.from_numpy(post)[None], torch.from_numpy(static)[None])
    assert out.shape == (1, 4, 64, 64)
