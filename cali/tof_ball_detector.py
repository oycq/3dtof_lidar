#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof_ball_detector.py

ToF 小球检测模块, 只做检测, 不做可视化.
输入: tof.raw
输出: ToF 图像 2D 像素坐标, 即 centroid (x, y).

方法:
- 反射率强度 = tof3d.tof_reflectance_mean3_max().
- 在有效像素（depth>0）中找到“反射率最高点”作为初始中心.
- 在该点周围 3x3 区域内, 用反射率作为权重计算 2D 重心（亚像素）.
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

from tof3d import ToF3DParams, tof_distance_and_histograms, tof_reflectance_mean3_max


# ===================== 可调参数（统一入口） =====================
# 注意：这些阈值不要在 check.py / my_calibrate.py 等外部脚本里重复定义，
# 统一在本文件顶部维护，避免“一个脚本改了阈值另一个没改”造成结果不一致。
#
# 使用的有效 bin 数（跟随 tof3d 参数上限）
TOF_BALL_VALID_BINS: int = int(ToF3DParams().valid_bin_num)
# 重心窗口默认值
TOF_BALL_WINDOW_SIZE_DEFAULT: int = 3


@dataclass(frozen=True)
class ToFBallDetection2D:
    centroid_xy: Optional[Tuple[float, float]]  # (x, y), ToF 像素坐标系


def detect_ball_tof_2d(
    tof_raw: str | Path,
    *,
    window_size: int = int(TOF_BALL_WINDOW_SIZE_DEFAULT),
) -> ToFBallDetection2D:
    """
    返回 ToF 2D 像素坐标 (x, y).
    centroid_xy 为 None 表示未检出.
    """
    p = Path(tof_raw)
    if not p.exists():
        return ToFBallDetection2D(centroid_xy=None)

    # 这里需要同时用到：
    # - 反射率强度图（由直方图统一策略计算）做阈值过滤、中心选择与重心权重
    # - 距离矩阵只用于过滤无效像素（depth=0）
    params = ToF3DParams(min_peak_count=0.0)
    depth_m, hists_u16 = tof_distance_and_histograms(p, params=params)
    if hists_u16.size == 0 or depth_m.size == 0:
        return ToFBallDetection2D(centroid_xy=None)

    h, w, _ = hists_u16.shape
    # 反射率强度：与 check.py / visualize_data.py 保持一致
    vb = int(np.clip(int(TOF_BALL_VALID_BINS), 1, min(int(ToF3DParams().valid_bin_num), hists_u16.shape[2])))
    refl = tof_reflectance_mean3_max(hists_u16, use_bins=vb)  # (H,W) float32

    # 中心选择：在有效像素中找反射率最高点（不再选最近 depth，也不做固定阈值过滤）
    m = depth_m > 0.0
    if not bool(np.any(m)):
        return ToFBallDetection2D(centroid_xy=None)

    refl_sel = np.where(m, refl.astype(np.float32, copy=False), -np.inf)
    flat_idx = int(np.argmax(refl_sel))
    if not np.isfinite(float(refl_sel.flat[flat_idx])):
        return ToFBallDetection2D(centroid_xy=None)
    y0, x0 = int(flat_idx // w), int(flat_idx % w)

    # 在该点周围 3x3 用“反射率(refl)”做加权重心（亚像素）
    # 兼容保留 window_size 参数，但当前策略固定为 3x3（更稳定、逻辑更简单）。
    _ = int(window_size)  # keep signature compatibility; intentionally unused
    r = 1
    xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1), dtype=np.float32)
    ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    xi = xs.astype(np.int32, copy=False)
    yi = ys.astype(np.int32, copy=False)
    ww = refl[yi[:, None], xi[None, :]].astype(np.float32, copy=False)
    # 邻域内若 depth=0（无效），则不参与重心（权重置 0）
    dm = depth_m[yi[:, None], xi[None, :]].astype(np.float32, copy=False)
    ww = np.where(dm > 0.0, ww, 0.0).astype(np.float32, copy=False)
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        cx, cy = float(x0), float(y0)
    else:
        cx = float(np.sum(xx * ww) / wsum)
        cy = float(np.sum(yy * ww) / wsum)

    return ToFBallDetection2D(centroid_xy=(cx, cy))


