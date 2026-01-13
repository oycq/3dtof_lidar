#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof3d.py

专门负责 TOF 3D（深度/距离矩阵）解析与计算：
- 输入：设备导出的 tof.raw（uint16，小端）
- 输出：30x40 的距离矩阵（单位：米，float32；无效为 0）

该实现保持与仓库内 get_tof.py / visualize_data.py 的核心计算逻辑一致：
去 5120 字节头 -> (H*W,64) -> 取前 62 bins -> 去底噪/弱信号/噪声 -> 峰值附近质心
-> depth = c * time_resolution / 2 * centroid * cos(theta_x)*cos(theta_y)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ToF3DParams:
    # 图像尺寸（像素）
    width: int = 40
    height: int = 30

    # 文件格式
    header_bytes: int = 5120  # 头部元数据长度（字节）

    # TDC 参数
    bin_num: int = 64
    valid_bin_num: int = 62
    offset_bin: int = 0

    # 过滤/质心参数（与 get_tof.py 对齐）
    base_th_ratio: float = 10.0
    clop_bin_num: int = 4
    pde_min_ratio: float = 80.0
    std_ratio: float = 2.2
    min_depth_m: float = 0.4
    # 置信度：峰值太小认为无效（0 表示不启用）
    min_peak_count: float = 0.0

    # 物理参数
    c: float = 3e8
    time_resolution_s: float = 1e-9

    # FOV（度）
    fov_x_deg: float = 60.0
    fov_y_deg: float = 45.0


DEFAULT_PARAMS = ToF3DParams()


def tof_read_u16(tof_raw_path: str | Path) -> np.ndarray:
    """读取 tof.raw 为 uint16 一维数组（包含头部）。"""
    p = Path(tof_raw_path)
    return np.fromfile(str(p), dtype=np.uint16)


def tof_distance_matrix(tof_raw_path: str | Path, params: ToF3DParams = DEFAULT_PARAMS) -> np.ndarray:
    """
    从 tof.raw 计算距离（深度）矩阵。

    返回:
        depth: (H, W) float32，单位米；无效为 0
    """
    raw_u16 = tof_read_u16(tof_raw_path)
    return tof_distance_matrix_from_u16(raw_u16, params=params)


def tof_histograms(tof_raw_path: str | Path, params: ToF3DParams = DEFAULT_PARAMS) -> np.ndarray:
    """
    从 tof.raw 解析每个像素的原始直方图（未做滤波）。

    返回:
        hist: (H, W, bin_num) uint16
    """
    raw_u16 = tof_read_u16(tof_raw_path)
    return tof_histograms_from_u16(raw_u16, params=params)


def tof_histograms_from_u16(raw_u16: np.ndarray, params: ToF3DParams = DEFAULT_PARAMS) -> np.ndarray:
    """
    直接从 uint16 数组（包含头部）解析 (H, W, bin_num) 的直方图。
    """
    h, w = int(params.height), int(params.width)
    b = int(params.bin_num)

    header_words = int(params.header_bytes) // 2
    if raw_u16.size <= header_words:
        return np.zeros((h, w, b), dtype=np.uint16)

    data = raw_u16[header_words:]
    expected = h * w * b
    if data.size < expected:
        return np.zeros((h, w, b), dtype=np.uint16)

    data = data[:expected]
    return data.reshape((h, w, b)).astype(np.uint16, copy=False)


def tof_distance_matrix_from_u16(raw_u16: np.ndarray, params: ToF3DParams = DEFAULT_PARAMS) -> np.ndarray:
    """
    直接从 uint16 数组（包含头部）计算距离矩阵。
    """
    h, w = int(params.height), int(params.width)

    header_words = int(params.header_bytes) // 2
    if raw_u16.size <= header_words:
        return np.zeros((h, w), dtype=np.float32)

    data = raw_u16[header_words:]

    expected = h * w * int(params.bin_num)
    if data.size < expected:
        return np.zeros((h, w), dtype=np.float32)
    data = data[:expected]

    hist_u16 = data.reshape((h * w, int(params.bin_num)))[:, : int(params.valid_bin_num)]
    hist = hist_u16.astype(np.float32, copy=True)

    # 峰值置信度过滤（基于原始直方图的峰值）
    if float(params.min_peak_count) > 0.0:
        peak_raw = hist_u16.max(axis=1).astype(np.float32, copy=False)
        low_conf = peak_raw < float(params.min_peak_count)
        hist[low_conf] = 0.0

    # total photons
    shots = np.sum(hist, axis=1)  # (H*W,)

    # 去底噪
    hist[hist <= float(params.base_th_ratio)] = 0.0
    # 去弱信号
    hist[shots < float(params.pde_min_ratio)] = 0.0

    # 噪声筛选：max < mean + k*std
    max_vals = hist.max(axis=1)
    mean_vals = hist.mean(axis=1)
    std_vals = hist.std(axis=1)
    thresholds = mean_vals + float(params.std_ratio) * std_vals
    noise_mask = max_vals < thresholds
    hist[noise_mask] = 0.0
    max_vals[noise_mask] = 0.0

    # 峰值位置（1-based，和 get_tof.py 习惯一致）
    max_pos = hist.argmax(axis=1) + 1
    max_pos[max_vals == 0] = 0

    centroid = _centroid_bins(hist, max_pos, params=params)  # (H*W,)
    centroid_map = centroid.reshape((h, w))

    # FOV 修正
    theta_x = np.linspace(-params.fov_x_deg / 2.0, params.fov_x_deg / 2.0, w)
    theta_y = np.linspace(-params.fov_y_deg / 2.0, params.fov_y_deg / 2.0, h)
    theta_x_grid, theta_y_grid = np.deg2rad(np.meshgrid(theta_x, theta_y))

    depth = (params.c * params.time_resolution_s / 2.0) * centroid_map * np.cos(theta_x_grid) * np.cos(theta_y_grid)
    depth = depth.astype(np.float32, copy=False)
    depth[depth < float(params.min_depth_m)] = 0.0
    return depth


def tof_distance_and_histograms(
    tof_raw_path: str | Path, params: ToF3DParams = DEFAULT_PARAMS
) -> tuple[np.ndarray, np.ndarray]:
    """
    一次读取 tof.raw，同时返回:
    - depth: (H,W) float32 米
    - hist:  (H,W,bin_num) uint16 原始直方图
    """
    raw_u16 = tof_read_u16(tof_raw_path)
    depth = tof_distance_matrix_from_u16(raw_u16, params=params)
    hist = tof_histograms_from_u16(raw_u16, params=params)
    return depth, hist


def _centroid_bins(hist: np.ndarray, max_pos_1based: np.ndarray, params: ToF3DParams) -> np.ndarray:
    """
    在峰值附近窗口计算质心位置（bin）。

    hist: (N, valid_bin_num) float32
    max_pos_1based: (N,) int，0 表示无效
    """
    n = int(hist.shape[0])
    valid_bins = int(params.valid_bin_num)
    r = int(params.clop_bin_num)

    centroid = np.zeros((n,), dtype=np.float32)
    offset = float(params.offset_bin)

    for i in range(n):
        mp = int(max_pos_1based[i])
        if mp == 0:
            continue

        mp_clamped = max(r, min(mp, valid_bins - r))
        s = mp_clamped - r
        e = mp_clamped + r
        counts = hist[i, s:e]

        denom = float(np.sum(counts))
        if denom <= 0.0:
            continue

        bins = np.arange(s, e, dtype=np.float32)
        centroid[i] = float(np.dot(bins, counts) / denom - offset)

    return centroid


