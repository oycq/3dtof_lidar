#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

规则网络, 直接由 64-bin 直方图输出:
- dist: 距离(米)
- snr: 信噪比
- reflectance: 反射率
- conf: 置信度(0/1)

流程:
1) 先拆分前 62 个有效 bin 与最后两个饱和 bin:
   sat_value = bin64 * 1024 + bin63
   若 sat_value <= 0, 则赋值为 50000
2) 归一化: hist = hist * (50000 / sat_value), 统一到每脉冲基准
3) hist_bias: 减去 bin0 的固定偏置 80, relu
4) 窜光抑制(crosstalk suppression):
   - 对每个 bin(共 62 个)在整帧(H,W)上取均值
   - 低于对应均值 * 系数的位置置 0, 抑制窜管干扰
5) argmax clip 到 [1, 60]
6) 距离 = argmax[-1, 0, 1] 三个 bin 的重心 * 0.6m
   (dist/peak 使用抑制后 hist, mean 使用原始 hist 避免被拉低)
7) snr = (max - mean) / sqrt(mean)
8) 反射率 = dist^2 * (max - mean) / 156250
9) 若 snr > 4 且 reflectance > 2.5%, 则 conf = 1, 否则为 0
"""

from __future__ import annotations

import torch
import torch.nn as nn

IS_6321 = True
DIST_SCALE_M = 0.6
TAIL_BASE = 1024.0
PULSES = 50000.0

if IS_6321:
    REFLECT_K = 156250.0 / 3
    DIST_BIAS = 0.25
else:
    REFLECT_K = 156250.0
    DIST_BIAS = 0.6
REFLECT_THRESH = 0.025
SNR_THRESH = 4.0
ARGMAX_CLIP_MIN = 1
ARGMAX_CLIP_MAX = 60
NOISE_BIAS = 4.0
CROSSTALK_MEAN_COEF = 1
EPS = 1e-6

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        hist_bias = torch.tensor([80.0] + [0.0] * 61, dtype=torch.float32).view(1, 62, 1, 1)
        self.register_buffer("hist_bias", hist_bias)

    def _apply_hist_bias(self, hist):
        return torch.relu(hist - self.hist_bias)

    def _split_hist_and_tail(self, x):
        hist, raw_bin_63, raw_bin_64 = torch.split(x, [62, 1, 1], dim=1)
        return hist, raw_bin_63, raw_bin_64

    def _caculate_sat_value(self, raw_bin_63, raw_bin_64):
        if IS_6321:
            sat_value = raw_bin_63 * TAIL_BASE + raw_bin_64  #6321
        else:
            sat_value = raw_bin_64 * TAIL_BASE + raw_bin_63  #1860
            sat_value = torch.where(sat_value > 0, sat_value, torch.full_like(sat_value, PULSES)) #1860
        return sat_value

    def _distance(self, x):
        peak_idx = torch.argmax(x, dim=1, keepdim=True)
        peak_idx = torch.clamp(peak_idx, min=ARGMAX_CLIP_MIN, max=ARGMAX_CLIP_MAX)

        a = torch.gather(x, dim=1, index=peak_idx - 1)
        b = torch.gather(x, dim=1, index=peak_idx)
        c = torch.gather(x, dim=1, index=peak_idx + 1)

        centroid = (c - a) / (a + b + c + 1) + peak_idx # +1这里是为了防止除0
        dist = centroid * DIST_SCALE_M
        dist = dist + DIST_BIAS
        return dist

    def _crosstalk_suppression(self, hist):
        bin_thresh = torch.mean(hist, dim=(2, 3), keepdim=True) * CROSSTALK_MEAN_COEF
        keep = hist >= bin_thresh
        return torch.where(keep, hist, torch.zeros_like(hist))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist, raw_bin_63, raw_bin_64 = self._split_hist_and_tail(x)

        # 计算 sat_value, 升级动态范围
        sat_value = self._caculate_sat_value(raw_bin_63, raw_bin_64)
        k = PULSES / sat_value
        hist = hist * k

        hist = self._apply_hist_bias(hist)

        # 窜光抑制: dist/peak 用抑制后 hist, mean 用原始 hist
        hist_eroded = self._crosstalk_suppression(hist)

        dist = self._distance(hist_eroded)

        # 计算信号强度和噪声
        mean = torch.mean(hist, dim=1, keepdim=True)
        peak = torch.max(hist_eroded, dim=1, keepdim=True).values
        signal = peak - mean
        noise = torch.sqrt(mean) + NOISE_BIAS
        snr = signal / noise

        # 计算反射率和置信度
        reflectance = dist * dist * signal / REFLECT_K
        conf = ((snr > SNR_THRESH) & (reflectance > REFLECT_THRESH)).to(torch.float32)

        return dist, conf, peak, reflectance, snr


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf, peak, reflectance, snr = net(inp)
    print(dist.shape, conf.shape, peak.shape, reflectance.shape, snr.shape)


