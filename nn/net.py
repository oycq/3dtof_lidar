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
import torch.nn.functional as F

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
NOISE_BIAS = 3.0
CROSSTALK_MEAN_COEF = 1

# peak mask 的偏置: 用于 sign(x - peak_val + PEAK_EPS) 在 peak 位置输出 1
PEAK_EPS = 0.1

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        hist_bias = torch.tensor([80.0] + [0.0] * 61, dtype=torch.float32).view(1, 62, 1, 1)
        self.register_buffer("hist_bias", hist_bias)

        # ---- 距离重心计算: 用 1x1 conv 替代 argmax + gather (BPU 友好) ----
        # 输出通道 i 表示"假设 peak 在 bin i"时的三邻域重心距离:
        #   tilt[i]   =  x[i+1] - x[i-1]           (右邻 - 左邻, 反映峰往哪边倾)
        #   total[i]  =  x[i-1] + x[i] + x[i+1]    (三邻域总和)
        #   centroid  =  tilt / (total + 1) + i    (+1 防除零)
        # 等价于原先的 clip(argmax, 1, 60): i=0 复用 i=1 的 anchor, i=61 复用 i=60
        tilt_kernel  = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        total_kernel = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        for i in range(62):
            anchor = min(max(i, 1), 60)  # 实际参与重心计算的中心 bin, clip 到 [1, 60]
            tilt_kernel[i,  anchor - 1, 0, 0] = -1.0
            tilt_kernel[i,  anchor + 1, 0, 0] = 1.0
            total_kernel[i, anchor - 1, 0, 0] = 1.0
            total_kernel[i, anchor,     0, 0] = 1.0
            total_kernel[i, anchor + 1, 0, 0] = 1.0
        self.register_buffer("tilt_kernel",  tilt_kernel)
        self.register_buffer("total_kernel", total_kernel)

        # 每个通道加上自身 bin 索引, 同样 clip 到 [1, 60]
        bin_index = torch.arange(62, dtype=torch.float32).clamp(1, 60).view(1, 62, 1, 1)
        self.register_buffer("bin_index", bin_index)

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
        # 1) 卷积并行算出 62 个候选距离 (B, 62, H, W)
        #    channel i 表示"假设 peak 在 bin i"时的三邻域重心距离
        tilt  = F.conv2d(x, self.tilt_kernel)
        total = F.conv2d(x, self.total_kernel) + 1.0  # +1 防止除 0
        centroid = tilt / total + self.bin_index
        dist_candidates = centroid * DIST_SCALE_M + DIST_BIAS

        # 2) max + sign + relu 构造 one-hot peak mask (替代 argmax)
        #    sign(0) = 0, 所以 peak 位置自己需要靠 +PEAK_EPS 唤醒成 1
        #    PEAK_EPS = 0.1 < 1 (hist 最小计量单位), 不会把次大位置误判成 peak
        peak_val  = torch.amax(x, dim=1, keepdim=True)
        peak_mask = torch.relu(torch.sign(x - peak_val + PEAK_EPS))

        # 3) 只在 peak 通道保留候选距离, sum 合并到单通道
        dist = torch.sum(peak_mask * dist_candidates, dim=1, keepdim=True)
        return dist

    def _crosstalk_suppression(self, hist):
        bin_thresh = torch.mean(hist, dim=(2, 3), keepdim=True) * CROSSTALK_MEAN_COEF
        # 用 sign+relu 构造 0/1 mask，避开 BPU 不支持的 where：
        # hist > bin_thresh -> sign=1, relu=1；hist <= bin_thresh -> sign<=0, relu=0
        mask = torch.relu(torch.sign(hist - bin_thresh))
        return hist * mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hist, raw_bin_63, raw_bin_64 = self._split_hist_and_tail(x)

        # 计算 sat_value, 升级动态范围
        sat_value = self._caculate_sat_value(raw_bin_63, raw_bin_64)
        k = PULSES / sat_value
        hist = hist * k

        hist = self._apply_hist_bias(hist)

        # 窜光抑制: dist/peak 用抑制后 hist, mean 用原始 hist
        hist_eroded = self._crosstalk_suppression(hist)

        # 计算距离
        dist = self._distance(hist_eroded)

        # 计算信号强度和噪声
        mean = torch.mean(hist, dim=1, keepdim=True)
        peak = torch.amax(hist_eroded, dim=1, keepdim=True)
        signal = peak - mean
        noise = torch.sqrt(mean) + NOISE_BIAS
        snr = signal / noise

        # 计算反射率
        reflectance = dist * dist * signal / REFLECT_K

        # 计算置信度
        snr_pass     = torch.relu(torch.sign(snr - SNR_THRESH))
        reflect_pass = torch.relu(torch.sign(reflectance - REFLECT_THRESH))
        conf = snr_pass * reflect_pass

        return dist, conf, peak, reflectance, snr


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf, peak, reflectance, snr = net(inp)
    print(dist.shape, conf.shape, peak.shape, reflectance.shape, snr.shape)


