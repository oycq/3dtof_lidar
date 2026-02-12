#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

轻量网络（不做 downsample，适配 30x40 小分辨率）：
- 输入：ToF 直方图  (B, 64, H, W) 其中 H=30, W=40
- 输出：            (B, out_bins, H, W) 概率（每像素 64 类 softmax 后的概率）
  - 推荐 out_bins=64：
    - 通道 0..62：有效距离类别
    - 通道 63：无效类别（GT 超出量程 / NA 等）

训练 loss（在 train.py 实现）：
- per-pixel 交叉熵（等价于只取 GT bin 的概率做 -log(prob)，“熵”）
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_channels: int = 64, out_bins: int = 64):
        super().__init__()

        # 1x1 卷积逐步降通道：64 -> ... -> out_bins
        # 不下采样，保持 (H,W) 不变
        if out_bins <= 0:
            raise ValueError(f"out_bins must be positive, got {out_bins}")
        channels = [in_channels, 64, 64, 64, int(out_bins)]
        layers: list[nn.Module] = []
        for i in range(len(channels) - 2):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=1, padding=0, bias=True)
            )
            layers.append(nn.ReLU(inplace=True))

        # 最后一层输出 out_bins 通道 logits
        layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1, padding=0, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,64,H,W)
        return: (B,out_bins,H,W) probs（内部已做 softmax）
        """
        logits = self.net(x)
        return torch.softmax(logits, dim=1)


if __name__ == "__main__":
    net = Network(out_bins=64)
    inp = torch.randn(1, 64, 30, 40)
    out = net(inp)
    print(out.shape)


