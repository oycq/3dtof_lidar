#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

规则网络，直接由 64-bin 直方图输出：
- dist: 距离（米）
- conf: 置信度（0/1）

流程：
1) 先计算最后两个 bin 的饱和系数：
   sat_coef = 50000 / (bin64 * 1024 + bin63)
2) 将最后两个 bin 置 0 后，基于 64 个 bin 计算：
   mean / std / max / argmax
3) argmax clip 到 [1, 60]
4) 距离 = argmax[-1, 0, 1] 三个 bin 的重心 * 0.6m
5) snr = (max - mean) / std
6) 反射率 = dist^2 * max / 156250
   若 max == 1023，则反射率乘以 sat_coef
7) 若 snr > 4 且 reflectance > 2.5%，则 conf = 1，否则为 0
"""

from __future__ import annotations

import torch
import torch.nn as nn

DIST_SCALE_M = 0.6
DIST_EPS = 1e-6
TAIL_SCALE = 50000.0
TAIL_BASE = 1024.0
REFLECT_DENOM = 156250.0
REFLECT_SAT_VALUE = 1023.0
REFLECT_THRESH = 0.025
SNR_THRESH = 4.0
ARGMAX_CLIP_MIN = 1
ARGMAX_CLIP_MAX = 60

class Network(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)

    def _prepare_bins(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_bin_63 = x[:, -2:-1, :, :]
        raw_bin_64 = x[:, -1:, :, :]
        sat_denom = raw_bin_64 * float(TAIL_BASE) + raw_bin_63
        sat_coef = torch.where(
            sat_denom > float(DIST_EPS),
            float(TAIL_SCALE) / sat_denom,
            torch.ones_like(sat_denom),
        )

        zero_tail = torch.zeros_like(x[:, -2:, :, :])
        work = torch.cat([x[:, :-2, :, :], zero_tail], dim=1)
        return work.contiguous(), sat_coef.contiguous()

    def _distance(self, work: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        peak_idx = torch.argmax(work, dim=1, keepdim=True)
        peak_idx = torch.clamp(peak_idx, min=ARGMAX_CLIP_MIN, max=ARGMAX_CLIP_MAX)

        offsets = torch.tensor([-1, 0, 1], device=work.device, dtype=peak_idx.dtype).view(1, 3, 1, 1)
        idxs = peak_idx + offsets
        vals = torch.gather(work, dim=1, index=idxs)

        w_sum = torch.sum(vals, dim=1, keepdim=True)
        num = torch.sum(idxs.to(dtype=work.dtype) * vals, dim=1, keepdim=True)
        centroid = torch.where(
            w_sum > float(DIST_EPS),
            num / torch.clamp(w_sum, min=float(DIST_EPS)),
            peak_idx.to(dtype=work.dtype),
        )
        dist = centroid * float(DIST_SCALE_M)
        return dist.contiguous(), peak_idx.contiguous(), vals.contiguous()

    def _stats(self, work: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.mean(work, dim=1, keepdim=True)
        std = torch.std(work, dim=1, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=float(DIST_EPS))
        vmax = torch.max(work, dim=1, keepdim=True).values
        return mean.contiguous(), std.contiguous(), vmax.contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """正式推理接口（用于 ONNX 导出）：输出距离和置信度。"""
        work, sat_coef = self._prepare_bins(x)
        dist, _, _ = self._distance(work)
        mean, std, vmax = self._stats(work)

        snr = (vmax - mean) / std
        reflectance = dist * dist * vmax / float(REFLECT_DENOM)
        reflectance = torch.where(
            torch.eq(vmax, float(REFLECT_SAT_VALUE)),
            reflectance * sat_coef,
            reflectance,
        )
        conf = ((snr > float(SNR_THRESH)) & (reflectance > float(REFLECT_THRESH))).to(dtype=x.dtype)
        dist = dist.contiguous()
        conf = conf.contiguous()
        return dist, conf


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf = net(inp)
    print(dist.shape, conf.shape)


