import pytest

torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from kosmofire.nn_bs import ordinal_loss


def _logits(true_class: int, predicted: int) -> torch.Tensor:
    logits = torch.full((1, 4, 2, 2), -5.0)
    logits[:, predicted] = 5.0
    return logits


def test_neighbour_error_penalised_less_than_opposite_class():
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    near = ordinal_loss(_logits(0, 1), target)
    far = ordinal_loss(_logits(0, 3), target)
    assert near < far


def test_plain_cross_entropy_treats_them_equally():
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    near = F.cross_entropy(_logits(0, 1), target)
    far = F.cross_entropy(_logits(0, 3), target)
    assert torch.isclose(near, far)
