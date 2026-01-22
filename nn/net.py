#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

轻量网络（不做 downsample，适配 30x40 小分辨率）：
- 输入：ToF 直方图  (B, 64, H, W) 其中 H=30, W=40
- 输出：            (B,  2, H, W)
  - out[:,0] = 距离倒数（inv_depth，>=0）
  - out[:,1] = 高斯分布尺度（sigma，>0），表示以 inv_depth 为均值的观测噪声尺度

训练 loss（在 train.py 实现）：
- GT(inv_depth) 在 N(mu=pred_inv_depth, sigma) 下的负对数似然（-log 概率 / “熵”）
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

        # 最后一层输出 2 通道（inv_depth 与 sigma 的 raw 输出）
        layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1, padding=0, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,64,H,W)
        return: (B,2,H,W)，其中
          - out[:,0] 已经过 softplus，保证非负
          - out[:,1] 已经过 softplus，保证为正（sigma）
        """
        y = self.net(x)  # (B,2,H,W)
        inv_depth = F.softplus(y[:, 0:1, :, :])
        sigma = F.softplus(y[:, 1:2, :, :])
        return torch.cat([inv_depth, sigma], dim=1)


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    out = net(inp)
    print(out.shape)


