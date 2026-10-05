import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import REFLECTANCE_SCALE, SAR_DB_SCALE, SCL_CLOUD_CLASSES
from .landcover import UNKNOWN_INDEX, landcover_index

OPTICAL_MEAN = np.array([0.04, 0.06, 0.06, 0.10, 0.20, 0.24, 0.26, 0.24, 0.16], dtype=np.float32)
OPTICAL_STD = np.array([0.03, 0.03, 0.04, 0.05, 0.08, 0.09, 0.10, 0.09, 0.08], dtype=np.float32)
SAR_MEAN = np.array([-12.0, -19.0], dtype=np.float32)
SAR_STD = np.array([4.0, 4.0], dtype=np.float32)
BRANCH_CHANNELS = 9 + 2 + 1
AUX_CHANNELS = 2 + UNKNOWN_INDEX + 1


def prepare_inputs(s2_pre, s2_post, s1_pre, s1_post, aux) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def branch(s2, s1):
        optical = (s2[:9].astype(np.float32) / REFLECTANCE_SCALE - OPTICAL_MEAN[:, None, None]) / OPTICAL_STD[:, None, None]
        sar = (s1.astype(np.float32) / SAR_DB_SCALE - SAR_MEAN[:, None, None]) / SAR_STD[:, None, None]
        cloud = np.isin(s2[9], SCL_CLOUD_CLASSES).astype(np.float32)[None]
        return np.concatenate([optical, sar, cloud], axis=0)

    dem = (aux[0].astype(np.float32) - 100.0) / 100.0
    slope = aux[1].astype(np.float32) / 10.0
    index = landcover_index(aux[2]).astype(np.int64)
    one_hot = np.eye(UNKNOWN_INDEX + 1, dtype=np.float32)[index].transpose(2, 0, 1)
    static = np.concatenate([dem[None], slope[None], one_hot], axis=0)
    return branch(s2_pre, s1_pre), branch(s2_post, s1_post), static


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    def __init__(self, base: int = 16, depth: int = 4, classes: int = 4):
        super().__init__()
        widths = [base * 2**i for i in range(depth)]
        self.encoders = nn.ModuleList()
        cin = BRANCH_CHANNELS
        for w in widths:
            self.encoders.append(_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.fuse = nn.ModuleList([nn.Conv2d(2 * w, w, 1) for w in widths])
        self.static = nn.Conv2d(AUX_CHANNELS, widths[-1], 1)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(_block(2 * widths[i - 1], widths[i - 1]))
        self.head = nn.Conv2d(widths[0], classes, 1)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats

    def forward(self, pre: torch.Tensor, post: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        f_pre = self._encode(pre)
        f_post = self._encode(post)
        merged = [fuse(torch.cat([b, torch.abs(b - a)], dim=1)) for fuse, a, b in zip(self.fuse, f_pre, f_post)]
        x = merged[-1] + self.static(F.adaptive_avg_pool2d(static, merged[-1].shape[-2:]))
        for up, dec, skip in zip(self.up, self.dec, reversed(merged[:-1])):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def ordinal_loss(logits: torch.Tensor, target: torch.Tensor, distance_weight: float = 1.0, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights)
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(probs.shape[1], device=logits.device).view(1, -1, 1, 1)
    distance = torch.abs(classes - target.unsqueeze(1)).to(probs.dtype)
    return ce + distance_weight * (probs * distance).sum(dim=1).mean()


def seed_everything(seed: int, threads: int = 4) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
