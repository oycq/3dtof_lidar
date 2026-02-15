#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

网络结构（3x3 卷积）：
- 距离头：64x30x40 -> 3x3 Conv -> 64x30x40（输出 64-bin logits）
- 距离输出：argmax(bin) 后查表还原为距离（米）
- 概率头：64x30x40 -> 3x3 Conv -> 1x30x40 + Sigmoid（输出误差在阈值内的概率）
"""

from __future__ import annotations

import torch
import torch.nn as nn

KERNEL_SIZE = 1
PADDING = KERNEL_SIZE // 2

class Network(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
        )
        # 预计算 64 个 bin 对应距离，前向中用索引查表避免 pow 运算。
        dist_lut = torch.tensor([1.06**i for i in range(64)], dtype=torch.float32)
        self.register_buffer("dist_lut", dist_lut, persistent=True)

        self.prob = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=KERNEL_SIZE, padding=PADDING, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """正式推理接口（用于 ONNX 导出）：只输出距离和概率。"""
        logits = self.cls(x)  # (B,64,H,W), raw logits
        idx = torch.argmax(logits, dim=1)  # (B,H,W), int64
        dist = self.dist_lut[idx].unsqueeze(1).contiguous()  # (B,1,H,W)
        conf = self.prob(x).contiguous()  # (B,1,H,W)
        return dist, conf

    @torch.jit.ignore
    def forward_train(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """训练接口：输出 bin logits、距离与概率，不用于 ONNX 导出。"""
        logits = self.cls(x)  # (B,64,H,W)
        idx = torch.argmax(logits, dim=1)  # (B,H,W), int64
        dist = self.dist_lut[idx].unsqueeze(1).contiguous()  # (B,1,H,W)
        conf = self.prob(x).contiguous()  # (B,1,H,W)
        return {
            "bin_logits": logits,
            "dist": dist,
            "conf": conf,
        }


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf = net(inp)
    out_train = net.forward_train(inp)
    print(dist.shape, conf.shape, out_train["dist"].shape)


