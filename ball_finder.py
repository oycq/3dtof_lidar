#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ball_finder.py

只基于激光雷达点云找球（球心/半径），不依赖 3DToF，也不接受 3DToF 结果作为引导。

注意：
- 所有“点云找球”的算法参数只放在本文件内（find_ball.py 不应包含这些参数）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ===================== 点云找球算法参数（只允许在本文件出现） =====================
# 目标球半径范围（m）：直径约 10cm~20cm
SPHERE_R_MIN_M = 0.05
SPHERE_R_MAX_M = 0.10

# RANSAC
SPHERE_RANSAC_ITERS = 50  # 每个格子各自做 1000 次
SPHERE_INLIER_THRESH_M = 0.02 # | ||p-c|| - r | < thresh
SPHERE_MIN_INLIERS_GLOBAL = 250  # 全局（在整帧点云上重新算 inliers 后）最低内点数
SPHERE_MIN_INLIERS_CELL = 80     # 单个格子内做 RANSAC 时的最低内点数（允许只看到部分球面）
SPHERE_MAX_FIT_POINTS = 60_000  # RANSAC 前最多参与拟合的点数（加速）

# 基础预过滤（仅用于降低离群点/加速，不依赖外部引导）
# 1) 距离裁剪：先去掉距离 > 4m 的点（用户需求）
MAX_RANGE_M = 4.0
# 2) 竖直重叠窗口：0-14°，7-21°...（窗口 14°，步长 7°）
VERT_WIN_DEG = 14.0
VERT_STEP_DEG = 7.0
# 2.1) 窗口内计算“参考最近距离”时，为了抗噪，使用距离排序后的第 N 个（1-based）
VERT_NEAREST_RANK = 100
# 2.2) 窗口内只保留距离 <= (参考最近距离 + 0.2m) 的点
VERT_NEAREST_KEEP_DELTA_M = 0.2

# 视场角（用于筛选参与RANSAC的点；竖直窗口也只在这个 FOV 内滑动）
LIDAR_FOV_DEG = 70.0

# “球的圆周可见/不遮挡” 质量判定（过滤一小片弧面被误拟合成球的情况）
SPHERE_EDGE_ANGLE_DEG = 18.0          # 轮廓带宽：极角 > 90°-edge_angle
SPHERE_EDGE_BINS = 36                # 方位角分桶数（越大越严格）
SPHERE_EDGE_MIN_POINTS = 80          # 轮廓点最少数量
SPHERE_EDGE_MIN_BIN_FRACTION = 0.65  # 被占用的方位桶比例（0~1）
SPHERE_EDGE_MAX_GAP_BINS = 2         # 最大连续空桶数（环形）


@dataclass(frozen=True)
class SphereFit:
    center: np.ndarray  # (3,) float64
    radius: float
    inliers: np.ndarray  # (N,) bool（对应“拟合阶段”使用的点集）


@dataclass(frozen=True)
class BallFindResult:
    center_xyz: Optional[Tuple[float, float, float]]
    sphere: Optional[SphereFit]
    render_points: np.ndarray  # (M,3) float32，用于显示（基础预过滤后的点云）
    filtered_points: np.ndarray  # (M2,3) float32，用于显示（最终用于窗口过滤后的点云）
    inlier_points: np.ndarray  # (K,3) float32，球面内点，用于高亮


def _prefilter_lidar_points(pts: np.ndarray) -> np.ndarray:
    """
    预过滤（降低离群点/加速）：
    - 只保留前方（x > 0）
    - 距离裁剪：||p|| <= MAX_RANGE_M
    """
    if pts.shape[0] == 0:
        return pts

    p = np.asarray(pts)
    if p.ndim != 2 or p.shape[1] != 3:
        return p[:0]

    # 只保留前方 + 距离裁剪（用户：先将距离大于 4m 的点去掉）
    x = p[:, 0].astype(np.float64, copy=False)
    y = p[:, 1].astype(np.float64, copy=False)
    z = p[:, 2].astype(np.float64, copy=False)
    rng = np.sqrt(x * x + y * y + z * z)
    m0 = (x > 0.0) & np.isfinite(rng) & (rng <= float(MAX_RANGE_M))
    if not np.any(m0):
        return p[:0]
    return p[m0]


def _window_near_filter(pts: np.ndarray) -> np.ndarray:
    """
    在“一个竖直窗口(14°范围)”内做距离窗口过滤：
    - 先取该窗口内距离排序后的第 VERT_NEAREST_RANK 个作为参考最近距离（抗噪）
    - 保留距离 <= 参考距离 + VERT_NEAREST_KEEP_DELTA_M 的点
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


def _sphere_from_4pts(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Optional[tuple[np.ndarray, float]]:
    """由 4 个点确定一个球（一般位置）。返回 (center(3,), radius) 或 None（退化/不可解）。"""
    p1 = p1.astype(np.float64, copy=False)
    p2 = p2.astype(np.float64, copy=False)
    p3 = p3.astype(np.float64, copy=False)
    p4 = p4.astype(np.float64, copy=False)

    A = np.stack([p2 - p1, p3 - p1, p4 - p1], axis=0) * 2.0  # (3,3)
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
    """
    用线性最小二乘在 inliers 上拟合球：
      x^2+y^2+z^2 = 2cx x + 2cy y + 2cz z + k
    """
    if pts.shape[0] < 10:
        return None
    p = pts.astype(np.float64, copy=False)
    A = np.column_stack(
        [2.0 * p[:, 0], 2.0 * p[:, 1], 2.0 * p[:, 2], np.ones((p.shape[0],), dtype=np.float64)]
    )
    b = (p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2).astype(np.float64, copy=False)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, cz, k = sol.tolist()
    c = np.array([cx, cy, cz], dtype=np.float64)
    r2 = float(k + cx * cx + cy * cy + cz * cz)
    if r2 <= 0.0 or (not np.isfinite(r2)):
        return None
    r = float(np.sqrt(r2))
    return c, r


def _sphere_edge_coverage_ok(*, pts: np.ndarray, center: np.ndarray, inliers: np.ndarray) -> bool:
    """轮廓覆盖质量检查（过滤残缺弧面误拟合）。"""
    if pts.shape[0] == 0:
        return False
    if inliers.shape[0] != pts.shape[0]:
        return False

    c = center.astype(np.float64, copy=False).reshape(3)
    cn = float(np.linalg.norm(c))
    if cn <= 1e-6 or (not np.isfinite(cn)):
        return False

    # v：从球心指向传感器（原点）
    v = (-c / cn).astype(np.float64, copy=False)

    # 构造正交基 u,w,v
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(tmp, v))) > 0.90:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(tmp, v)
    un = float(np.linalg.norm(u))
    if un <= 1e-9:
        return False
    u /= un
    w = np.cross(v, u)

    pin = pts[inliers].astype(np.float64, copy=False)
    if pin.shape[0] < int(SPHERE_MIN_INLIERS_GLOBAL):
        return False

    s = pin - c.reshape(1, 3)
    sn = np.linalg.norm(s, axis=1)
    m = sn > 1e-6
    if not np.any(m):
        return False
    s = s[m] / sn[m].reshape(-1, 1)

    cos_a = np.clip((s @ v.reshape(3, 1)).reshape(-1), -1.0, 1.0)
    a = np.arccos(cos_a)
    edge_thr = (np.pi / 2.0) - np.deg2rad(float(SPHERE_EDGE_ANGLE_DEG))
    edge = a > float(edge_thr)
    if int(np.count_nonzero(edge)) < int(SPHERE_EDGE_MIN_POINTS):
        return False

    se = s[edge]
    az = np.arctan2((se @ w.reshape(3, 1)).reshape(-1), (se @ u.reshape(3, 1)).reshape(-1))

    bins = int(SPHERE_EDGE_BINS)
    bidx = np.floor((az + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    bidx = np.clip(bidx, 0, bins - 1)
    occ = np.zeros((bins,), dtype=np.uint8)
    occ[bidx] = 1

    occ_cnt = int(np.count_nonzero(occ))
    if occ_cnt < int(np.ceil(bins * float(SPHERE_EDGE_MIN_BIN_FRACTION))):
        return False

    z = np.concatenate([occ, occ], axis=0)
    best_gap = 0
    cur = 0
    for val in z.tolist():
        if val == 0:
            cur += 1
            best_gap = max(best_gap, cur)
        else:
            cur = 0
    best_gap = min(best_gap, bins)
    if best_gap > int(SPHERE_EDGE_MAX_GAP_BINS):
        return False

    return True


def _ransac_sphere(
    pts: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    iters: int,
    inlier_thresh: float,
    min_inliers: int,
    seed: int = 0,
) -> Optional[SphereFit]:
    if pts.shape[0] < 50:
        return None

    rng = np.random.default_rng(int(seed))
    p = pts.astype(np.float64, copy=False)
    n = p.shape[0]

    best_cnt = 0
    best_center = None
    best_radius = 0.0
    best_inliers = None

    for _ in range(int(iters)):
        idx = rng.choice(n, size=4, replace=False)
        m = _sphere_from_4pts(p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]])
        if m is None:
            continue
        c, r = m
        if (r < float(r_min)) or (r > float(r_max)):
            continue

        d = np.linalg.norm(p - c.reshape(1, 3), axis=1)
        inliers = np.abs(d - float(r)) < float(inlier_thresh)
        cnt = int(np.count_nonzero(inliers))
        if cnt > best_cnt and cnt >= int(min_inliers):
            if not _sphere_edge_coverage_ok(pts=p, center=c, inliers=inliers):
                continue
            best_cnt = cnt
            best_center = c
            best_radius = float(r)
            best_inliers = inliers

    if best_center is None or best_inliers is None or best_cnt < int(min_inliers):
        return None

    refined = _sphere_refine_least_squares(p[best_inliers])
    if refined is not None:
        c2, r2 = refined
        if float(r_min) <= float(r2) <= float(r_max):
            d2 = np.linalg.norm(p - c2.reshape(1, 3), axis=1)
            in2 = np.abs(d2 - float(r2)) < float(inlier_thresh)
            if int(np.count_nonzero(in2)) >= best_cnt:
                if _sphere_edge_coverage_ok(pts=p, center=c2, inliers=in2):
                    best_center, best_radius, best_inliers = c2, float(r2), in2

    return SphereFit(center=best_center, radius=float(best_radius), inliers=best_inliers)


def find_ball_from_lidar(points_xyz: np.ndarray, *, seed: int = 0) -> BallFindResult:
    """
    仅用 LiDAR 点云找球心。
    - points_xyz: (N,3) numpy 数组（单位 m）
    """
    pts = np.asarray(points_xyz)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz 必须是 (N,3) 的 numpy 数组")

    if pts.shape[0] == 0:
        z = pts.astype(np.float32, copy=False)
        return BallFindResult(center_xyz=None, sphere=None, render_points=z, filtered_points=z, inlier_points=z[:0])

    p_render = _prefilter_lidar_points(pts.astype(np.float32, copy=False))
    if p_render.shape[0] == 0:
        return BallFindResult(center_xyz=None, sphere=None, render_points=p_render, filtered_points=p_render, inlier_points=p_render[:0])

    # -------- 竖直重叠窗口：每个窗口(14°)先过滤，再做 RANSAC，最后选全局 inliers 最多的一个 --------
    rng = np.random.default_rng(int(seed))

    # 视场角筛选（竖直窗口也只在该 FOV 内滑动）
    fov = float(LIDAR_FOV_DEG)
    if (not np.isfinite(fov)) or fov <= 0.0:
        fov = 70.0
    half = float(np.deg2rad(fov / 2.0))
    half_deg = float(np.rad2deg(half))

    x = p_render[:, 0].astype(np.float64, copy=False)
    y = p_render[:, 1].astype(np.float64, copy=False)
    z = p_render[:, 2].astype(np.float64, copy=False)
    yaw = np.arctan2(y, x)
    # 竖直角使用 elevation：atan2(z, hypot(x,y))，避免 y≠0 时 atan2(z,x) 的偏差
    pitch = np.arctan2(z, np.hypot(x, y))
    m_fov = (x > 0.0) & (np.abs(yaw) <= half) & (np.abs(pitch) <= half)
    idx_fov = np.nonzero(m_fov)[0].astype(np.int32, copy=False)
    if idx_fov.size == 0:
        return BallFindResult(center_xyz=None, sphere=None, render_points=p_render, filtered_points=p_render, inlier_points=p_render[:0])

    best_c = None
    best_r = 0.0
    best_in = None  # 对齐 p_render 的 bool mask
    best_cnt = 0

    # 用于显示：第一轮过滤（4m裁剪 + 竖直窗口内“最近距离+0.2m”）后的点的并集
    stage1_keep = np.zeros((p_render.shape[0],), dtype=bool)

    p64 = p_render.astype(np.float64, copy=False)

    pitch_deg = np.rad2deg(pitch[idx_fov]).astype(np.float64, copy=False)

    win = float(VERT_WIN_DEG)
    step = float(VERT_STEP_DEG)
    if (not np.isfinite(win)) or win <= 0.0:
        win = 14.0
    if (not np.isfinite(step)) or step <= 0.0:
        step = 7.0

    # 按你的例子从 0° 开始：0-14, 7-21...
    start_deg = 0.0
    max_start = half_deg - win
    if max_start < start_deg:
        max_start = start_deg

    starts = np.arange(start_deg, max_start + 1e-6, step, dtype=np.float64)
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

        # 14° 窗口内：先找“参考最近距离”并过滤，再 RANSAC
        p_win0 = p_render[idx_win]
        # 同时记录“第一轮过滤”保留下来的点（并集）
        p_win = _window_near_filter(p_win0)
        if p_win.shape[0] > 0:
            # 通过距离一致性把 p_win 映射回 idx_win（避免重复计算 ref）
            # 这里用 mask 的方式更稳定：直接在窗口内重新算一次 keep mask
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

        fit = _ransac_sphere(
            p_fit,
            r_min=float(SPHERE_R_MIN_M),
            r_max=float(SPHERE_R_MAX_M),
            iters=int(SPHERE_RANSAC_ITERS),
            inlier_thresh=float(SPHERE_INLIER_THRESH_M),
            min_inliers=int(SPHERE_MIN_INLIERS_CELL),
            seed=int(seed) + int(round(s0 * 10.0)),
        )
        if fit is None:
            continue

        # 在整帧 p_render 上重新算 inliers（不让“窗口切分”影响最终内点统计）
        c = fit.center.astype(np.float64, copy=False).reshape(3)
        r = float(fit.radius)
        d0 = np.linalg.norm(p64 - c.reshape(1, 3), axis=1)
        in0 = np.abs(d0 - r) < float(SPHERE_INLIER_THRESH_M)
        cnt0 = int(np.count_nonzero(in0))
        if cnt0 < int(SPHERE_MIN_INLIERS_GLOBAL):
            continue

        refined = _sphere_refine_least_squares(p64[in0])
        if refined is not None:
            c2, r2 = refined
            if float(SPHERE_R_MIN_M) <= float(r2) <= float(SPHERE_R_MAX_M):
                d2 = np.linalg.norm(p64 - c2.reshape(1, 3), axis=1)
                in2 = np.abs(d2 - float(r2)) < float(SPHERE_INLIER_THRESH_M)
                if int(np.count_nonzero(in2)) >= cnt0:
                    c, r, in0 = c2, float(r2), in2
                    cnt0 = int(np.count_nonzero(in0))

        if not _sphere_edge_coverage_ok(pts=p64, center=c, inliers=in0):
            continue

        if cnt0 > best_cnt:
            best_cnt = cnt0
            best_c = c
            best_r = float(r)
            best_in = in0

    if best_c is None or best_in is None:
        return BallFindResult(center_xyz=None, sphere=None, render_points=p_render, filtered_points=p_render, inlier_points=p_render[:0])

    cx0, cy0, cz0 = best_c.tolist()
    inlier_pts = p_render[best_in].astype(np.float32, copy=False)
    sphere = SphereFit(center=best_c, radius=float(best_r), inliers=best_in)
    filt_pts = p_render[stage1_keep].astype(np.float32, copy=False) if np.any(stage1_keep) else p_render.astype(np.float32, copy=False)
    return BallFindResult(
        center_xyz=(float(cx0), float(cy0), float(cz0)),
        sphere=sphere,
        render_points=p_render,
        filtered_points=filt_pts,
        inlier_points=inlier_pts,
    )


