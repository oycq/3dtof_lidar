#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lidar_ball_detector.py

LiDAR 小球检测模块, 只做检测, 不做可视化.

输入:
- points_xyz: (N, 3) 点云, 单位 m

输出:
- 球心 3D 坐标, center_xyz_m
- 拟合半径, radius_m
- 球心距离, center_range_m
- 最终拟合内点坐标, fit_points_xyz_m

为了便于 check.py 做可视化, 这里额外返回:
- render_points_xyz_m: 基础预过滤后的点, 前方 + 4m 截断
- filtered_points_xyz_m: 第一轮过滤后保留的点, 只累计到命中窗口时刻
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 允许在 cali/ 目录直接运行: 把项目根目录加入 sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ===================== 算法参数, 只允许在本文件出现 =====================
# 球半径范围, 单位 m, 直径约 10cm 到 20cm.
SPHERE_R_MIN_M = 0.05
SPHERE_R_MAX_M = 0.10

# RANSAC
SPHERE_RANSAC_ITERS = 50
SPHERE_INLIER_THRESH_M = 0.02  # | ||p-c|| - r | < thresh
SPHERE_MIN_INLIERS_GLOBAL = 250
SPHERE_MIN_INLIERS_CELL = 80
SPHERE_MAX_FIT_POINTS = 60_000

# 基础预过滤.
MAX_RANGE_M = 4.0  # 先去掉距离 > 4m 的点

# 竖直重叠窗口, win=14deg, step=7deg.
VERT_WIN_DEG = 14.0
VERT_STEP_DEG = 7.0
VERT_NEAREST_RANK = 100
VERT_NEAREST_KEEP_DELTA_M = 0.2

# LiDAR 视场角, 单位 deg, 用于筛选点并限定窗口扫描范围.
LIDAR_FOV_DEG = 70.0


@dataclass(frozen=True)
class LidarBallDetection:
    center_xyz_m: Optional[Tuple[float, float, float]]
    radius_m: Optional[float]
    center_range_m: Optional[float]
    fit_points_xyz_m: np.ndarray  # (K, 3) float32, 最终拟合内点

    # 给 check.py 可视化用.
    render_points_xyz_m: np.ndarray  # (M,3) float32
    filtered_points_xyz_m: np.ndarray  # (M2,3) float32


def _prefilter_lidar_points(pts: np.ndarray) -> np.ndarray:
    # 只保留前方点 (x>0), 并做距离截断 (||p||<=MAX_RANGE_M).
    if pts.shape[0] == 0:
        return pts
    p = np.asarray(pts)
    if p.ndim != 2 or p.shape[1] != 3:
        return p[:0]
    x = p[:, 0].astype(np.float64, copy=False)
    y = p[:, 1].astype(np.float64, copy=False)
    z = p[:, 2].astype(np.float64, copy=False)
    rng = np.sqrt(x * x + y * y + z * z)
    m = (x > 0.0) & np.isfinite(rng) & (rng <= float(MAX_RANGE_M))
    return p[m] if np.any(m) else p[:0]


def _window_near_filter(pts: np.ndarray) -> np.ndarray:
    """
    单个竖直窗口内的第一轮过滤:
    - 取距离排序第 VERT_NEAREST_RANK 个作为 ref, 用于抗极近噪声点.
    - 只保留 r <= ref + VERT_NEAREST_KEEP_DELTA_M.
    """
    if pts.shape[0] == 0:
        return pts
    p = pts.astype(np.float64, copy=False)
    r = np.linalg.norm(p, axis=1)
    if r.size == 0:
        return pts[:0]
    rank = int(VERT_NEAREST_RANK)
    if rank <= 0:
        rank = 1
    k = min(rank, int(r.size)) - 1
    ref = float(np.partition(r, k)[k])
    keep_delta = float(VERT_NEAREST_KEEP_DELTA_M)
    if (not np.isfinite(keep_delta)) or keep_delta < 0.0:
        keep_delta = 0.2
    m = r <= (ref + keep_delta)
    return pts[m]


def _maybe_subsample(pts: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] <= int(n_max):
        return pts
    idx = rng.choice(pts.shape[0], size=int(n_max), replace=False)
    return pts[idx]


def _sphere_from_4pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    # 用 4 个点解球 (一般位置). 返回 (center, radius) 或 None.
    p1 = p1.astype(np.float64, copy=False)
    p2 = p2.astype(np.float64, copy=False)
    p3 = p3.astype(np.float64, copy=False)
    p4 = p4.astype(np.float64, copy=False)

    A = np.stack([p2 - p1, p3 - p1, p4 - p1], axis=0) * 2.0
    b = np.array(
        [
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1),
            np.dot(p4, p4) - np.dot(p1, p1),
        ],
        dtype=np.float64,
    )
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    try:
        c = np.linalg.solve(A, b)
    except Exception:
        return None
    r = float(np.linalg.norm(c - p1))
    if (not np.isfinite(r)) or (not np.all(np.isfinite(c))):
        return None
    return c.astype(np.float64, copy=False), r


def _sphere_refine_least_squares(pts: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    # 在内点上做一次线性最小二乘 refine.
    # 目标: 给定一组点 p=(x,y,z), 拟合球 (c,r), 满足 ||p-c||^2 = r^2.
    # 展开可得线性形式:
    #   x^2 + y^2 + z^2 = 2*cx*x + 2*cy*y + 2*cz*z + k
    # 其中 k = r^2 - cx^2 - cy^2 - cz^2.
    # 于是可以用最小二乘解 [2x,2y,2z,1] * [cx,cy,cz,k]^T = x^2+y^2+z^2.
    if pts.shape[0] < 10:
        # 点太少时不做 refine, 避免不稳定.
        return None
    p = pts.astype(np.float64, copy=False)
    A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], 2.0 * p[:, 2], np.ones((p.shape[0],), dtype=np.float64)])
    b = (p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2).astype(np.float64, copy=False)
    try:
        # rcond=None 使用 numpy 默认策略.
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, cz, k = sol.tolist()
    c = np.array([cx, cy, cz], dtype=np.float64)
    r2 = float(k + cx * cx + cy * cy + cz * cz)
    if r2 <= 0.0 or (not np.isfinite(r2)):
        # r^2 非正或数值异常, 认为 refine 失败.
        return None
    return c, float(np.sqrt(r2))


def _ransac_sphere(
    pts: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    iters: int,
    inlier_thresh: float,
    min_inliers: int,
    seed: int = 0,
) -> Optional[tuple[np.ndarray, float]]:
    if pts.shape[0] < 50:
        # 点太少时 RANSAC 意义不大, 直接失败.
        return None
    seed_u32 = int(seed) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_u32)
    p = pts.astype(np.float64, copy=False)
    n = p.shape[0]

    best_cnt = 0
    best_center = None
    best_radius = 0.0

    for _ in range(int(iters)):
        # 每次随机选 4 个点, 解一个球. 4 点解球若退化(共面或数值不稳)则返回 None.
        idx = rng.choice(n, size=4, replace=False)
        m = _sphere_from_4pts(p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]])
        if m is None:
            continue
        c, r = m
        # 半径约束, 过滤掉不在目标尺寸范围内的候选.
        if (r < float(r_min)) or (r > float(r_max)):
            continue
        # 内点判定: 对每个点计算到球面的残差 | ||p-c|| - r |, 小于阈值认为是内点.
        d = np.linalg.norm(p - c.reshape(1, 3), axis=1)
        inliers = np.abs(d - float(r)) < float(inlier_thresh)
        cnt = int(np.count_nonzero(inliers))
        if cnt > best_cnt and cnt >= int(min_inliers):
            # 只保留内点更多的候选.
            best_cnt = cnt
            best_center = c
            best_radius = float(r)

    if best_center is None or best_cnt < int(min_inliers):
        # 没有任何满足最小内点数的候选.
        return None

    # 用当前内点做一次 refine.
    refined = _sphere_refine_least_squares(
        p[np.abs(np.linalg.norm(p - best_center.reshape(1, 3), axis=1) - best_radius) < float(inlier_thresh)]
    )
    if refined is not None:
        c2, r2 = refined
        if float(r_min) <= float(r2) <= float(r_max):
            best_center, best_radius = c2, float(r2)

    return best_center.astype(np.float64, copy=False), float(best_radius)


def detect_ball_lidar(points_xyz: np.ndarray, *, seed: int = 0) -> LidarBallDetection:
    # 主流程:
    # 1) 基础预过滤: 前方 + 距离截断
    # 2) 计算竖直角 elevation, 从上往下扫描竖直窗口
    # 3) 窗口内做第一轮过滤 (ref + delta), 再做 RANSAC 拟合球
    # 4) 在全局点云上验证内点数, 命中即退出
    pts = np.asarray(points_xyz)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be a (N, 3) numpy array")

    if pts.shape[0] == 0:
        z = pts.astype(np.float32, copy=False)
        return LidarBallDetection(None, None, None, z[:0], z, z)

    p_render = _prefilter_lidar_points(pts.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    if p_render.shape[0] == 0:
        return LidarBallDetection(None, None, None, p_render[:0], p_render, p_render)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    # FOV 筛选 + 竖直角 elevation 计算.
    fov = float(LIDAR_FOV_DEG)
    if (not np.isfinite(fov)) or fov <= 0.0:
        fov = 70.0
    half = float(np.deg2rad(fov / 2.0))
    half_deg = float(np.rad2deg(half))

    x = p_render[:, 0].astype(np.float64, copy=False)
    y = p_render[:, 1].astype(np.float64, copy=False)
    z = p_render[:, 2].astype(np.float64, copy=False)
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))  # elevation
    m_fov = (x > 0.0) & (np.abs(yaw) <= half) & (np.abs(pitch) <= half)
    idx_fov = np.nonzero(m_fov)[0].astype(np.int32, copy=False)
    if idx_fov.size == 0:
        return LidarBallDetection(None, None, None, p_render[:0], p_render, p_render)

    # 给 check.py: 第一轮过滤保留点的并集, 只累计到命中窗口时刻.
    stage1_keep = np.zeros((p_render.shape[0],), dtype=bool)
    p64 = p_render.astype(np.float64, copy=False)
    pitch_deg = np.rad2deg(pitch[idx_fov]).astype(np.float64, copy=False)

    win = float(VERT_WIN_DEG)
    step = float(VERT_STEP_DEG)
    if (not np.isfinite(win)) or win <= 0.0:
        win = 14.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = 7.0

    # 固定扫描顺序: 从上到下 (+half_deg -> -half_deg), 命中即退出.
    start_deg = half_deg - win
    end_deg = -half_deg
    if start_deg < end_deg:
        start_deg = end_deg
    starts = np.arange(start_deg, end_deg - 1e-6, -step, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([start_deg], dtype=np.float64)

    for s0 in starts.tolist():
        s1 = float(s0 + win)
        sel = (pitch_deg >= float(s0)) & (pitch_deg < float(s1))
        if not np.any(sel):
            continue
        idx_win = idx_fov[sel]
        if int(idx_win.size) < 50:
            continue

        p_win0 = p_render[idx_win]
        p_win = _window_near_filter(p_win0)

        # 记录第一轮过滤保留点 (并集): 对窗口内点应用 ref+delta mask.
        if p_win0.shape[0] > 0:
            p64w = p_win0.astype(np.float64, copy=False)
            rw = np.linalg.norm(p64w, axis=1)
            rank = int(VERT_NEAREST_RANK)
            if rank <= 0:
                rank = 1
            kw = min(rank, int(rw.size)) - 1
            refw = float(np.partition(rw, kw)[kw])
            keep_delta = float(VERT_NEAREST_KEEP_DELTA_M)
            if (not np.isfinite(keep_delta)) or keep_delta < 0.0:
                keep_delta = 0.2
            mw = rw <= (refw + keep_delta)
            stage1_keep[idx_win[mw]] = True

        if p_win.shape[0] < 50:
            continue

        p_fit = _maybe_subsample(p_win, int(SPHERE_MAX_FIT_POINTS), rng)
        model = _ransac_sphere(
            p_fit,
            r_min=float(SPHERE_R_MIN_M),
            r_max=float(SPHERE_R_MAX_M),
            iters=int(SPHERE_RANSAC_ITERS),
            inlier_thresh=float(SPHERE_INLIER_THRESH_M),
            min_inliers=int(SPHERE_MIN_INLIERS_CELL),
            seed=int(seed) + int(round(s0 * 10.0)),
        )
        if model is None:
            continue
        c, r = model

        # 全局验证: 在 p_render 上算 inliers, 要求内点数达到阈值.
        d0 = np.linalg.norm(p64 - c.reshape(1, 3), axis=1)
        in0 = np.abs(d0 - float(r)) < float(SPHERE_INLIER_THRESH_M)
        if int(np.count_nonzero(in0)) < int(SPHERE_MIN_INLIERS_GLOBAL):
            continue

        refined = _sphere_refine_least_squares(p64[in0])
        if refined is not None:
            c2, r2 = refined
            if float(SPHERE_R_MIN_M) <= float(r2) <= float(SPHERE_R_MAX_M):
                d2 = np.linalg.norm(p64 - c2.reshape(1, 3), axis=1)
                in2 = np.abs(d2 - float(r2)) < float(SPHERE_INLIER_THRESH_M)
                if int(np.count_nonzero(in2)) >= int(np.count_nonzero(in0)):
                    c, r, in0 = c2, float(r2), in2

        center = (float(c[0]), float(c[1]), float(c[2]))
        center_range = float(np.linalg.norm(np.array(center, dtype=np.float64)))
        filt_pts = p_render[stage1_keep].astype(np.float32, copy=False) if np.any(stage1_keep) else p_render.astype(np.float32, copy=False)
        inlier_pts = p_render[in0].astype(np.float32, copy=False)
        return LidarBallDetection(
            center_xyz_m=center,
            radius_m=float(r),
            center_range_m=center_range,
            fit_points_xyz_m=inlier_pts,
            render_points_xyz_m=p_render,
            filtered_points_xyz_m=filt_pts,
        )

    # 未命中.
    return LidarBallDetection(None, None, None, p_render[:0], p_render, p_render)


