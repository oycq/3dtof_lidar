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
1) 先拆分前 62 个有效 bin 与最后两个饱和 bin：
   sat_value = bin64 * 1024 + bin63
   若 sat_value <= 0，则赋值为 50000
2) 仅基于前 62 个有效 bin 计算：
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
REFLECT_K = 156250.0 / 3
REFLECT_THRESH = 0.025
SNR_THRESH = 4.0
ARGMAX_CLIP_MIN = 1
ARGMAX_CLIP_MAX = 60
DIST_BIAS = 0.4

class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def _split_hist_and_tail(self, x):
        hist, raw_bin_63, raw_bin_64 = torch.split(x, [62, 1, 1], dim=1)
        hist[0,0] -= 80   #6321
        return hist, raw_bin_63, raw_bin_64

    def _caculate_sat_value(self, raw_bin_63, raw_bin_64):
        sat_value = raw_bin_63 * TAIL_BASE + raw_bin_64  #6321
        #sat_value = raw_bin_64 * TAIL_BASE + raw_bin_63  #1860
        #sat_value = torch.where(sat_value > 0, sat_value, torch.full_like(sat_value, PULSES)) #1860
        return sat_value

    def _distance(self, x):
        peak_idx = torch.argmax(x, dim=1, keepdim=True)
        peak_idx = torch.clamp(peak_idx, min=ARGMAX_CLIP_MIN, max=ARGMAX_CLIP_MAX)

        a = torch.gather(x, dim=1, index=peak_idx - 1)
        b = torch.gather(x, dim=1, index=peak_idx)
        c = torch.gather(x, dim=1, index=peak_idx + 1)

        centroid = (c - a) / (a + b + c) + peak_idx
        dist = centroid * DIST_SCALE_M
        dist = dist + DIST_BIAS
        return dist

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist, raw_bin_63, raw_bin_64 = self._split_hist_and_tail(x)
        sat_value = self._caculate_sat_value(raw_bin_63, raw_bin_64)
        dist = self._distance(hist)
        mean = torch.mean(hist, dim=1, keepdim=True)
        std = torch.std(hist, dim=1, keepdim=True, unbiased=False)
        vmax = torch.max(hist, dim=1, keepdim=True).values
        signal = vmax - mean
        
        snr = signal / std
        reflectance = dist * dist * signal / REFLECT_K * PULSES / sat_value
        conf = ((snr > SNR_THRESH) & (reflectance > REFLECT_THRESH))
        return dist, snr, reflectance, conf


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, snr, reflectance, conf = net(inp)
    print(dist.shape, snr.shape, reflectance.shape, conf.shape)


