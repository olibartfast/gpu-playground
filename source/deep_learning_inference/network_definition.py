"""
Neural network definition (PyTorch).
Matches the CUDA implementation in cuda/inference.cpp.

Network architecture:
  Input: RGBA uint8 image (HWC) → output: grayscale uint8 image (HW)
  Internally operates on NCHW float32 in [0, 1].

  1.  cast uint8 → float32
  2.  / 255                              → [0, 1]
  3.  Conv2d(4→32, 3×3, replicate pad)
  4.  ReLU
  5.  Conv2d(32→32, 1×1)
  6.  ReLU
  7.  AvgPool2d(kernel=2, stride=4)      → H/4, W/4
  8.  ExoticConv1x1(32→256)             (custom gamma activation)
  9.  ReLU
  10. Conv2d(256→16, 1×1)
  11. ReLU
  12. Clamp[0, 1]
  13. PixelShuffle(4)                    → H, W, 1 channel
"""

import struct
import numpy as np
import torch
import torch.nn as nn


class ExoticConv1x1(nn.Module):
    """
    1×1 convolution with a custom gamma activation instead of a plain dot product.

    Output[n, o, h, w] = sum_c  gamma( input[n, c, h, w], weight[o, c] )

    gamma(x, y) = (b*y + 1) * x   if x*y > 0
                  exp(b*y) * x     otherwise
    where b = beta^2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.beta   = nn.Parameter(torch.empty(1))
        self._init_params()

    def _init_params(self):
        std = 1.0 / np.sqrt(self.weight.size(1))
        nn.init.normal_(self.weight, std=std)
        nn.init.normal_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        w  = self.weight.view(1, self.weight.shape[0], self.weight.shape[1], 1, 1)  # (1, O, C, 1, 1)
        x_ = x.unsqueeze(1)                                                          # (N, 1, C, H, W)

        b = self.beta.square()
        mask          = (x_ * w) > 0
        positive_case = b * w + 1          # (1, O, C, 1, 1)
        negative_case = torch.exp(b * w)   # (1, O, C, 1, 1)
        final_weight  = torch.where(mask, positive_case, negative_case)

        return torch.einsum("noihw,noihw->nohw", x_, final_weight)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1, padding_mode="replicate", bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=4),
            ExoticConv1x1(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Hardtanh(min_val=0.0, max_val=1.0),  # clamp [0, 1]
            nn.PixelShuffle(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

    # ------------------------------------------------------------------
    # Weight I/O — binary layout matches NetworkWeights in inference.h
    # ------------------------------------------------------------------

    def export_weights(self, path: str):
        mods = dict(self.seq.named_children())
        layer3:  nn.Conv2d      = mods["0"]
        layer5:  nn.Conv2d      = mods["2"]
        layer8:  ExoticConv1x1  = mods["5"]
        layer10: nn.Conv2d      = mods["7"]

        with open(path, "wb") as f:
            def write(t: torch.Tensor):
                f.write(t.detach().cpu().contiguous().numpy().astype(np.float32).tobytes())

            write(layer3.weight)         # [32, 4, 3, 3]
            write(layer5.weight)         # [32, 32]  (stored as [32, 32, 1, 1] but only 32*32 floats)
            write(layer8.weight)         # [256, 32]
            write(layer8.beta)           # [1]
            write(layer10.weight)        # [16, 256]

    def import_weights(self, path: str):
        mods = dict(self.seq.named_children())
        layer3:  nn.Conv2d      = mods["0"]
        layer5:  nn.Conv2d      = mods["2"]
        layer8:  ExoticConv1x1  = mods["5"]
        layer10: nn.Conv2d      = mods["7"]

        with open(path, "rb") as f:
            def load(shape):
                n = 1
                for s in shape: n *= s
                return torch.from_numpy(
                    np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape).copy()
                )

            layer3.weight.data.copy_(load(layer3.weight.shape))
            layer5.weight.data.copy_(load(layer5.weight.shape))
            layer8.weight.data.copy_(load(layer8.weight.shape))
            layer8.beta.data.copy_(load(layer8.beta.shape))
            layer10.weight.data.copy_(load(layer10.weight.shape))
