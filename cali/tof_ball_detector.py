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

from tof3d import ToF3DParams, tof_distance_and_histograms


@dataclass(frozen=True)
class ToFBallDetection2D:
    centroid_xy: Optional[Tuple[float, float]]  # (x, y), ToF 像素坐标系


def detect_ball_tof_2d(
    tof_raw: str | Path,
    *,
    window_size: int = 3,
    min_peak: float = 512.0,
    valid_bins: int = int(ToF3DParams().valid_bin_num),
) -> ToFBallDetection2D:
    """
    返回 ToF 2D 像素坐标 (x, y).
    centroid_xy 为 None 表示未检出.
    """
    p = Path(tof_raw)
    if not p.exists():
        return ToFBallDetection2D(centroid_xy=None)

    # 这里需要同时用到：
    # - bin 峰值（作为“bin 值反射率”）做阈值过滤与重心权重
    # - 距离矩阵做“最近点”选择
    #
    # 注意：距离矩阵内部会做一系列滤波/最小深度过滤，导致某些像素 depth=0；
    # 我们在“最近点”选择时只考虑 depth>0 的像素。
    params = ToF3DParams(min_peak_count=0.0)
    depth_m, hists_u16 = tof_distance_and_histograms(p, params=params)
    if hists_u16.size == 0 or depth_m.size == 0:
        return ToFBallDetection2D(centroid_xy=None)

    h, w, _ = hists_u16.shape
    # “bin 值反射率”：取有效 bin 范围内的峰值（max bin count）
    vb = int(np.clip(int(valid_bins), 1, min(int(ToF3DParams().valid_bin_num), hists_u16.shape[2])))
    peak = hists_u16[:, :, :vb].max(axis=2).astype(np.float32, copy=False)  # (H,W)

    # 1) 在所有点里先筛：peak > min_peak（默认 512）
    # 2) 在这些点里找距离最近：depth>0 且 depth 最小
    m = (peak > float(min_peak)) & (depth_m > 0.0)
    if not bool(np.any(m)):
        return ToFBallDetection2D(centroid_xy=None)

    depth_sel = np.where(m, depth_m.astype(np.float32, copy=False), np.inf)
    flat_idx = int(np.argmin(depth_sel))
    if not np.isfinite(float(depth_sel.flat[flat_idx])):
        return ToFBallDetection2D(centroid_xy=None)
    y0, x0 = int(flat_idx // w), int(flat_idx % w)

    # 在该点周围 3x3（可通过 window_size 调整）用“反射率(peak)”做加权重心
    ws = int(max(1, window_size))
    if ws % 2 == 0:
        ws += 1
    r = max(0, ws // 2)
    xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1), dtype=np.float32)
    ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    ww = peak[ys.astype(np.int32)[:, None], xs.astype(np.int32)[None, :]].astype(np.float32, copy=False)
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        cx, cy = float(x0), float(y0)
    else:
        cx = float(np.sum(xx * ww) / wsum)
        cy = float(np.sum(yy * ww) / wsum)

    return ToFBallDetection2D(centroid_xy=(cx, cy))


