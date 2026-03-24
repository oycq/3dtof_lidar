#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/net.py

网络结构：
- 距离基线：直接由输入原始直方图计算
    1) n = argmax(x)
    2) 取 [n-2, n-1, n, n+1, n+2] 五个 bin 计算重心 bin_idx
    3) dist_raw = bin_idx * 0.6 (m)
- pileup bias 补偿（每像素）：
    输入 [v(n-2), v(n-1), v(n), v(n+1), v(n+2)]，经 MLP 5-8-8-1 输出 bias
    bias clip 到 [-1.2, 1.2]m
    dist = dist_raw - bias
- 置信度：固定 SNR（非学习）
    peak = max(x[:62])
    mean = mean(x[:62])
    std = std(x[:62])
    snr = (peak - mean) / std
"""

from __future__ import annotations

import torch
import torch.nn as nn

CONF_BINS = 62
DIST_SCALE_M = 0.6
DIST_EPS = 1e-6
BIAS_CLIP_M = 1.2

class Network(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.in_channels = int(in_channels)
        self.bias_head = nn.Sequential(
            nn.Linear(5, 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4, 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1, bias=True),
        )

    def _distance_raw_and_window5(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """由输入原始直方图计算 raw 距离（5-bin 重心），并返回 5-bin 值。"""
        if x.ndim != 4:
            raise ValueError(f"expect x shape (B,C,H,W), got {tuple(x.shape)}")
        channels = int(x.shape[1])
        if channels <= 0:
            raise ValueError("expect positive channel count")

        peak_idx = torch.argmax(x, dim=1)  # (B,H,W), int64
        offsets = torch.tensor([-2, -1, 0, 1, 2], device=x.device, dtype=peak_idx.dtype).view(1, 5, 1, 1)
        idxs = torch.clamp(peak_idx.unsqueeze(1) + offsets, min=0, max=channels - 1)  # (B,5,H,W)
        vals = torch.gather(x, dim=1, index=idxs)  # (B,5,H,W)

        w_sum = torch.sum(vals, dim=1)  # (B,H,W)
        num = torch.sum(idxs.to(dtype=x.dtype) * vals, dim=1)  # (B,H,W)
        centroid = torch.where(
            w_sum > DIST_EPS,
            num / torch.clamp(w_sum, min=DIST_EPS),
            peak_idx.to(dtype=x.dtype),
        )
        dist_raw = (centroid * float(DIST_SCALE_M)).unsqueeze(1).contiguous()
        window5 = vals.permute(0, 2, 3, 1).contiguous()  # (B,H,W,5)
        return dist_raw, window5

    def _predict_bias(self, window5: torch.Tensor) -> torch.Tensor:
        """每像素 5-8-8-1 输出 bias，并 clip 到 [-1.2, 1.2]m。"""
        if window5.ndim != 4 or window5.shape[-1] != 5:
            raise ValueError(f"expect window5 shape (B,H,W,5), got {tuple(window5.shape)}")
        b, h, w, _ = window5.shape
        z = window5.reshape(b * h * w, 5)
        bias = self.bias_head(z).reshape(b, 1, h, w)
        bias = torch.clamp(bias, min=-float(BIAS_CLIP_M), max=float(BIAS_CLIP_M))
        return bias.contiguous()

    def _fixed_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """按前 62 个 bin 计算固定 SNR，输出形状 (B,1,H,W)。"""
        src = x[:, :CONF_BINS, :, :]
        vmax = torch.max(src, dim=1, keepdim=True).values
        mean = torch.mean(src, dim=1, keepdim=True)
        std = torch.std(src, dim=1, keepdim=True, unbiased=False)
        snr = (vmax - mean) / std
        return snr.contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """正式推理接口（用于 ONNX 导出）：只输出距离和概率。"""
        dist_raw, window5 = self._distance_raw_and_window5(x)
        bias = self._predict_bias(window5)
        dist = (dist_raw - bias).contiguous()
        conf = self._fixed_confidence(x)  # (B,1,H,W)
        return dist, conf

    @torch.jit.ignore
    def forward_train(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """训练接口：输出 raw 距离、bias、补偿后距离与置信度。"""
        dist_raw, window5 = self._distance_raw_and_window5(x)
        bias = self._predict_bias(window5)
        dist = (dist_raw - bias).contiguous()
        conf = self._fixed_confidence(x)  # (B,1,H,W)
        return {
            "bin_logits": x,
            "window5": window5,
            "dist_raw": dist_raw,
            "bias": bias,
            "dist": dist,
            "conf": conf,
        }


if __name__ == "__main__":
    net = Network()
    inp = torch.randn(1, 64, 30, 40)
    dist, conf = net(inp)
    out_train = net.forward_train(inp)
    print(dist.shape, conf.shape, out_train["dist"].shape)


