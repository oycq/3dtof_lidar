#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

规则网络，直接由 64-bin 直方图输出：
- dist: 距离（米）
- snr: 信噪比
- reflectance: 反射率
- conf: 置信度（0/1）

流程：
1) 先计算最后两个 bin 的饱和值：
   sat_value = bin64 * 1024 + bin63
   若 sat_value == 0，则赋值为 50000
2) 将最后两个 bin 置 0 后，基于 64 个 bin 计算：
   mean / std / max / argmax
3) argmax clip 到 [1, 60]
4) 距离 = argmax[-1, 0, 1] 三个 bin 的重心 * 0.6m
5) snr = (max - mean) / std
6) 反射率 = dist^2 * max / 156250
   再乘以 50000 / sat_value
7) 若 snr > 4 且 reflectance > 2.5%，则 conf = 1，否则为 0
"""

from __future__ import annotations

import torch
import torch.nn as nn

DIST_SCALE_M = 0.6
TAIL_BASE = 1024.0
PULSES = 50000.0
REFLECT_K = 156250.0
REFLECT_THRESH = 0.025
SNR_THRESH = 4.0
ARGMAX_CLIP_MIN = 1
ARGMAX_CLIP_MAX = 60
DIST_BIAS = 0.4

class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def _caculate_sat_value(self, x):
        raw_bin_63 = x[:, -2:-1, :, :]
        raw_bin_64 = x[:, -1:, :, :]
        sat_value = raw_bin_64 * TAIL_BASE + raw_bin_63
        sat_value[sat_value == 0] = PULSES
        return sat_value

    def _delete_tail(self, x):
        x[:,:, -2:] = 0
        return x

    def _distance(self, x):
        peak_idx = torch.argmax(x, dim=1, keepdim=True)
        peak_idx = torch.clamp(peak_idx, min=ARGMAX_CLIP_MIN, max=ARGMAX_CLIP_MAX)

        a = torch.gather(x, dim=1, index=peak_idx - 1)
        b = torch.gather(x, dim=1, index=peak_idx + 0)
        c = torch.gather(x, dim=1, index=peak_idx + 1)

        centroid = (-1 * a + 0 * b + 1 * c) / (a + b + c) + peak_idx
        dist = centroid * DIST_SCALE_M
        dist = dist + DIST_BIAS
        return dist

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sat_value = self._caculate_sat_value(x)
        x = self._delete_tail(x)
        dist = self._distance(x)
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False)
        vmax = torch.max(x, dim=1, keepdim=True).values

        snr = (vmax - mean) / std
        reflectance = dist * dist * vmax / REFLECT_K * PULSES / sat_value
        conf = ((snr > float(SNR_THRESH)) & (reflectance > float(REFLECT_THRESH))).to(dtype=x.dtype)
        return dist, snr, reflectance, conf


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, snr, reflectance, conf = net(inp)
    print(dist.shape, snr.shape, reflectance.shape, conf.shape)


