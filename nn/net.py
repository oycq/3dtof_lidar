#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

轻量网络（不做 downsample，适配 30x40 小分辨率）：
- 输入：ToF 直方图  (B, 64, H, W) 其中 H=30, W=40
- 输出：            (B,  2, H, W)
  - out[:,0] = 距离倒数（inv_depth，>=0）
  - out[:,1] = 概率（prob，0~1），表示“当前距离预测与真值相差在 ±5% 内”的概率

训练 loss（在 train.py 实现）：
- L2(inv_depth)  占 90%
- 熵（BCE(prob)）占 10%
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()

        # 1x1 卷积逐步降通道：64 -> ... -> 2
        # 不下采样，保持 (H,W) 不变
        channels = [in_channels, 48, 32, 24, 16, 8, 4, 2]
        layers: list[nn.Module] = []
        for i in range(len(channels) - 2):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=1, padding=0, bias=True)
            )
            layers.append(nn.ReLU(inplace=True))

        # 最后一层输出 2 通道（inv_depth 与 prob 的 logits）
        layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1, padding=0, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,64,H,W)
        return: (B,2,H,W)，其中
          - out[:,0] 已经过 softplus，保证非负
          - out[:,1] 已经过 sigmoid，落在 0~1
        """
        y = self.net(x)  # (B,2,H,W)
        inv_depth = F.softplus(y[:, 0:1, :, :])
        prob = torch.sigmoid(y[:, 1:2, :, :])
        return torch.cat([inv_depth, prob], dim=1)


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    out = net(inp)
    print(out.shape)


