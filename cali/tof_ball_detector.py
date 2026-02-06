#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof_ball_detector.py

ToF 小球检测模块, 只做检测, 不做可视化.
输入: tof.raw
输出: ToF 图像 2D 像素坐标, 即 centroid (x, y).

方法:
- 反射率强度 = histogram sum, 先用 peak 阈值做置信度过滤.
- 在最亮点附近 window_size x window_size 区域内, 用强度作为权重计算 2D 重心.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 允许在 cali/ 目录直接运行: 把项目根目录加入 sys.path, 以便 import tof3d.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tof3d import ToF3DParams, tof_histograms, tof_reflectance_mean3_max


@dataclass(frozen=True)
class ToFBallDetection2D:
    centroid_xy: Optional[Tuple[float, float]]  # (x, y), ToF 像素坐标系


def detect_ball_tof_2d(
    tof_raw: str | Path,
    *,
    window_size: int = 5,
    min_peak: float = 100.0,
    valid_bins: int = int(ToF3DParams().valid_bin_num),
) -> ToFBallDetection2D:
    """
    返回 ToF 2D 像素坐标 (x, y).
    centroid_xy 为 None 表示未检出.
    """
    p = Path(tof_raw)
    if not p.exists():
        return ToFBallDetection2D(centroid_xy=None)

    params = ToF3DParams(min_peak_count=float(min_peak))
    hists = tof_histograms(p, params=params).astype(np.float32, copy=False)  # (H,W,64)
    if hists.size == 0:
        return ToFBallDetection2D(centroid_xy=None)

    h, w, _ = hists.shape
    # 反射率/强度相关：只使用有效 bin 范围（与 tof3d.py 对齐）
    vb = int(np.clip(int(valid_bins), 1, min(int(ToF3DParams().valid_bin_num), hists.shape[2])))
    h_use = hists[:, :, :vb]
    peak = h_use.max(axis=2).astype(np.float32, copy=False)
    # 强度：交给 tof3d.py 的统一策略（这里已裁剪到 vb）
    inten = tof_reflectance_mean3_max(h_use)
    inten = np.where(peak >= float(min_peak), inten, 0.0)
    if float(np.max(inten)) <= 0.0:
        return ToFBallDetection2D(centroid_xy=None)

    flat_idx = int(np.argmax(inten))
    y0, x0 = int(flat_idx // w), int(flat_idx % w)

    # window_size x window_size 强度加权重心
    r = max(0, int(window_size) // 2)
    xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1), dtype=np.float32)
    ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    ww = inten[ys.astype(np.int32)[:, None], xs.astype(np.int32)[None, :]].astype(np.float32, copy=False)
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        cx, cy = float(x0), float(y0)
    else:
        cx = float(np.sum(xx * ww) / wsum)
        cy = float(np.sum(yy * ww) / wsum)

    return ToFBallDetection2D(centroid_xy=(cx, cy))


