#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

规则网络, 直接由 64-bin 直方图输出:
- dist: 距离(米)
- snr: 信噪比
- reflectance: 反射率
- conf: 置信度(0/1)
- peak: 峰值信号 (原始 hist at 峰值 bin)

流程:
1) 拆分前 62 个有效 bin 与最后两个饱和 bin:
      sat_value = bin64 * 1024 + bin63
      若 sat_value <= 0, 则赋值为 50000
   得到归一化系数 k = PULSES / sat_value, 并构造归一化后的 hist_k = hist * k.

   两份直方图各司其职:
   - 原始 hist  : 用于距离重心 (dist_per_bin) 和 argmax 选峰 bin, 保持原始形状信息
   - 归一化 hist_k: 用于 snr / crosstalk / reflect 三路, 使阈值在不同脉冲数下一致

2) 用 1x1 conv 在原始 hist 上并行算出 62 路候选距离 (每 bin 都做三邻域重心细化):
      tilt[i]  =  x[i+1] - x[i-1]                 (右邻 - 左邻)
      total[i] =  x[i-1] + x[i] + x[i+1]          (三邻域和)
      centroid =  tilt / (total + 1) + i          (clip 到 [1, 60])
      dist_per_bin = centroid * DIST_SCALE_M + DIST_BIAS       (B, 62, H, W)

3) 在 hist_k 上算每 bin 的信号量 (形状都是 (B, 62, H, W)):
      signal_per_bin      = hist_k - mean(hist_k, dim=1)
      snr_per_bin         = signal_per_bin / (sqrt(mean) + NOISE_BIAS)
      reflectance_per_bin = dist_per_bin^2 * signal_per_bin / REFLECT_K

4) 三路 per-bin mask (都是 (B, 62, H, W), 0/1), 全部基于 hist_k:
      crosstalk_mask : hist_k > mean(hist_k) * CROSSTALK_MEAN_COEF
      snr_mask       : snr_per_bin         > SNR_THRESH
      reflect_mask   : reflectance_per_bin > REFLECT_THRESH
   三者相乘得到 per-bin 的 valid_mask.

5) 测距: gated_hist   = valid_mask * hist           (注意: 用原始 hist)
         one_hot_mask = one-hot(argmax(gated_hist)), 并列时按最小 bin 做 tiebreak

6) 用 one_hot_mask 从 62 路候选里各挑出命中那一路 (sum):
      dist        = Σ one_hot_mask * dist_per_bin
      reflectance = Σ one_hot_mask * reflectance_per_bin
      snr         = Σ one_hot_mask * snr_per_bin
      peak        = Σ one_hot_mask * gated_hist    (= amax(gated_hist), 无效像素为 0)

7) conf = amax(valid_mask, dim=1), 即任一 bin 通过 3 路 mask 即为有效像素
"""

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
# 反射率上限: 避免反射率太高,导致int16量化精度不够
MAX_REFL = 30.0
SNR_THRESH = 4.0
NOISE_BIAS = 3.0
CROSSTALK_MEAN_COEF = 0.66

# peak mask 的偏置: 用于 sign(x - peak_val + PEAK_EPS) 在 peak 位置输出 1
PEAK_EPS = 0.4


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- 距离重心计算: 用 1x1 conv 替代 argmax + gather (BPU 友好) ----
        # 输出通道 i 表示"假设 peak 在 bin i"时的三邻域重心距离:
        #   tilt[i]   =  x[i+1] - x[i-1]           (右邻 - 左邻, 反映峰往哪边倾)
        #   total[i]  =  x[i-1] + x[i] + x[i+1]    (三邻域总和)
        #   centroid  =  tilt / (total + 1) + i    (+1 防除零)
        # 等价于原先的 clip(argmax, 1, 60): i=0 复用 i=1 的 anchor, i=61 复用 i=60
        tilt_kernel = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        total_kernel = torch.zeros(62, 62, 1, 1, dtype=torch.float32)
        for i in range(62):
            anchor = min(max(i, 1), 60)  # 实际参与重心计算的中心 bin, clip 到 [1, 60]
            tilt_kernel[i, anchor - 1, 0, 0] = -1.0
            tilt_kernel[i, anchor + 1, 0, 0] = 1.0
            total_kernel[i, anchor - 1, 0, 0] = 1.0
            total_kernel[i, anchor,     0, 0] = 1.0
            total_kernel[i, anchor + 1, 0, 0] = 1.0
        self.register_buffer("tilt_kernel", tilt_kernel)
        self.register_buffer("total_kernel", total_kernel)

        # 每通道加上自身 bin 索引, 同样 clip 到 [1, 60]
        bin_index = torch.arange(62, dtype=torch.float32).clamp(1, 60).view(1, 62, 1, 1)
        self.register_buffer("bin_index", bin_index)

    def _split_hist_and_tail(self, x):
        hist, raw_bin_63, raw_bin_64 = torch.split(x, [62, 1, 1], dim=1)
        return hist, raw_bin_63, raw_bin_64

    def _caculate_sat_value(self, raw_bin_63, raw_bin_64):
        if IS_6321:
            sat_value = raw_bin_63 * TAIL_BASE + raw_bin_64  # 6321
        else:
            sat_value = raw_bin_64 * TAIL_BASE + raw_bin_63  # 1860
            sat_value = torch.where(sat_value > 0, sat_value, torch.full_like(sat_value, PULSES))  # 1860
        return sat_value

    def _crosstalk_mask(self, hist):
        """窜光抑制: 只输出 (B,62,H,W) 的 0/1 mask, 不改变 hist 本身."""
        # 全图均值 mean(H,W): 拆成两次单轴 ReduceMean, 避免被 horizon 优化器折叠成
        # GlobalAveragePool(只支持 int8 输入), 保证整条路径留在 int16 精度上.
        mean_h = torch.mean(hist,   dim=2, keepdim=True)  # axes=[2]  -> ReduceMean int16
        mean_hw = torch.mean(mean_h, dim=3, keepdim=True)  # axes=[3]  -> ReduceMean int16
        crosstalk_threshold = mean_hw * CROSSTALK_MEAN_COEF
        return torch.relu(torch.sign(hist - crosstalk_threshold))

    def _dist_per_bin(self, hist):
        """用三邻域重心算 62 路候选距离, 输出 (B, 62, H, W)."""
        tilt = F.conv2d(hist, self.tilt_kernel)
        total = F.conv2d(hist, self.total_kernel) + 1.0  # +1 防止除 0
        centroid = tilt / total + self.bin_index
        return centroid * DIST_SCALE_M + DIST_BIAS

    def _snr_and_mask(self, signal_per_bin, noise):
        """每 bin 的 SNR = signal / noise, 以及 SNR mask = (snr > SNR_THRESH)."""
        snr  = signal_per_bin / noise
        mask = torch.relu(torch.sign(snr - SNR_THRESH))
        return snr, mask

    def _reflectance_and_mask(self, signal_per_bin, dist_per_bin):
        """每 bin 的反射率 = dist^2 * signal / K, 以及 reflect mask = (reflect > REFLECT_THRESH).

        量化友好: signal 先 clamp 到 MAX_REFL*K/dist^2, 保证 reflect 最大 = MAX_REFL (物理上限 3000%)
        """
        dist_sq             = dist_per_bin * dist_per_bin
        signal_clip         = torch.minimum(signal_per_bin, (MAX_REFL * REFLECT_K) / dist_sq)
        reflectance_per_bin = dist_sq * signal_clip / REFLECT_K
        mask                = torch.relu(torch.sign(reflectance_per_bin - REFLECT_THRESH))
        return reflectance_per_bin, mask

    def _argmax_onehot(self, x):
        """
        在 (B, 62, H, W) 上按 channel 维找 argmax, 返回 one-hot mask.
        多个 bin 同值并列最大时, 只保留 bin 索引最小的那个.
        """
        # 1) 标出所有等于 max 的位置; 若有并列, max_mask 不止一个 1
        #    sign(0)=0, 所以 max 位置靠 +PEAK_EPS 被唤醒成 1
        max_val  = torch.amax(x, dim=1, keepdim=True)
        max_mask = torch.relu(torch.sign(x - max_val + PEAK_EPS))

        # 2) 对并列位置用 "bin 越小优先级越高" 的权重再取一次 argmax, 强制唯一
        #    非 max 位=0; max 位=(99 - bin_index)∈[39, 98] 一定为正, bin 越小权重越大
        priority     = max_mask * (99.0 - self.bin_index)
        priority_max = torch.amax(priority, dim=1, keepdim=True)
        one_hot      = torch.relu(torch.sign(priority - priority_max + PEAK_EPS))
        return one_hot

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- 0) 拆分: 62 个有效 bin + 2 个饱和 bin, 计算脉冲归一化系数 k ----
        hist, raw_bin_63, raw_bin_64 = self._split_hist_and_tail(x)
        sat_value = self._caculate_sat_value(raw_bin_63, raw_bin_64)
        k = PULSES / sat_value
        hist_k = hist * k  # 用于 snr / crosstalk / reflect (阈值在不同脉冲数下一致)

        # ---- 1) 距离重心 & argmax 门控, 都用原始 hist ----
        dist_per_bin = self._dist_per_bin(hist)  # (B, 62, H, W)

        # ---- 2) 每 bin 的信号量 / 噪声 (基于归一化 hist_k) ----
        mean_k         = torch.mean(hist_k, dim=1, keepdim=True)    # (B, 1, H, W)
        signal_per_bin = hist_k - mean_k                            # (B, 62, H, W)
        noise          = torch.sqrt(mean_k) + NOISE_BIAS            # (B, 1, H, W)

        snr_per_bin,         snr_mask     = self._snr_and_mask(signal_per_bin, noise)
        reflectance_per_bin, reflect_mask = self._reflectance_and_mask(signal_per_bin, dist_per_bin)

        # ---- 3) 三路 per-bin mask ----
        crosstalk_mask = self._crosstalk_mask(hist_k)                                   # (B, 62, H, W)
        valid_mask     = crosstalk_mask * snr_mask * reflect_mask                       # (B, 62, H, W)

        # ---- 4) 在原始 hist 上做 argmax 选峰 bin ----
        gated_hist   = valid_mask * hist                  # (B, 62, H, W)
        one_hot_mask = self._argmax_onehot(gated_hist)    # (B, 62, H, W)

        # ---- 5) 用 one_hot_mask 从 62 路候选里各挑出命中那一路 ----
        dist        = torch.sum(one_hot_mask * dist_per_bin,        dim=1, keepdim=True)
        reflectance = torch.sum(one_hot_mask * reflectance_per_bin, dim=1, keepdim=True)
        snr         = torch.sum(one_hot_mask * snr_per_bin,         dim=1, keepdim=True)
        peak        = torch.sum(one_hot_mask * hist_k,          dim=1, keepdim=True)  # = amax(gated_hist)

        # ---- 6) conf: 该像素任一 bin 通过 3 路 mask 即为有效 ----
        conf = torch.amax(valid_mask, dim=1, keepdim=True)

        return dist, conf, peak, reflectance, snr


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf, peak, reflectance, snr = net(inp)
    print(dist.shape, conf.shape, peak.shape, reflectance.shape, snr.shape)
