#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_ball.py

遍历 data/<场景>/，自动估计球的中心坐标，并以 OpenCV imshow 的方式交互显示（不落盘）。

球心算法（按你的描述实现）：
- 3D TOF：先找“最亮”像素（反射率强度=∑hist），再在其周围 5×5 的像素范围内，
  用反射率作为权重求重心（输出像素重心 + 3D 重心）
- LiDAR：先减去底面/大平面（把“面积>1㎡”的平面都干掉），剩下点里“最近点”作为球心
  （这里“最近”默认按 x 最小；若你要按欧氏距离可改）

使用方式（按你的要求：不传参数）：
- 直接运行：py .\\find_ball.py
- 关键配置都在本文件顶部“宏定义”区域：DATA_DIR / OVERWRITE / RANSAC 参数等

交互：
- 4：上一个场景
- 6：下一个场景
- ESC：退出
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tof3d import ToF3DParams, tof_distance_and_histograms


# ========= 与 visualize_data.py/client.py 对齐的一些常量 =========
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# LiDAR 2D（view.png 是 client.py 保存的 700x700）
LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = float(np.deg2rad(FOV_DEG / 2.0))

# TOF 2D（visualize_data.py 的显示尺寸）
TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 400
TOF_SHOW_H = 300

TOF_MIN_PEAK = 100
TOF_VALID_BINS = 62
TOF_COMP_ENABLE = True
TOF_COMP_A = 1.0
TOF_COMP_B_MM = -1447.0

TOF_INTEN_GAMMA = 2.2
TOF_INTEN_TARGET_MEAN = 0.18

# ======== LiDAR 小球检测参数（直径 10cm~20cm）========
SPHERE_R_MIN_M = 0.05
SPHERE_R_MAX_M = 0.10
SPHERE_RANSAC_ITERS = 2200
SPHERE_INLIER_THRESH_M = 0.012  # | ||p-c|| - r | < thresh
SPHERE_MIN_INLIERS = 250
SPHERE_MAX_FIT_POINTS = 60_000   # 为了速度：RANSAC 前最多用多少点
SPHERE_PREFILTER_RADIUS_M = 0.6  # 如果有 TOF 粗定位：只取其附近多大半径的点

# “球的圆周可见/不遮挡” 质量判定：
# 思路：在球心-传感器方向的“可见半球”里，靠近轮廓（极角接近 90°）的点，
# 其方位角应接近 360° 连续覆盖；如果只是一小片弧面拟合出来的假球，这里会有大缺口。
SPHERE_EDGE_ANGLE_DEG = 18.0          # 轮廓带宽：极角 > 90°-edge_angle
SPHERE_EDGE_BINS = 36                # 方位角分桶数（越大越严格）
SPHERE_EDGE_MIN_POINTS = 80          # 轮廓点最少数量
SPHERE_EDGE_MIN_BIN_FRACTION = 0.65  # 被占用的方位桶比例（0~1）
SPHERE_EDGE_MAX_GAP_BINS = 8         # 最大连续空桶数（环形）

# LiDAR 投影渲染参数（与 visualize_data.py 一致）
MAX_RANGE_M = 20.0
NEAR_SAT_M = 1.0


def _list_env_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    ds = [p for p in root.iterdir() if p.is_dir()]
    ds.sort(key=lambda p: p.name)
    return ds


def _find_points_npz(env_dir: Path) -> Optional[Path]:
    cands = sorted(env_dir.glob("points_last*.npz"))
    return cands[0] if cands else None


def _load_points(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack([x, y, z]).astype(np.float32, copy=False)


def _tof_pixel_to_disp_xy(px: float, py: float) -> tuple[int, int]:
    """
    复用 visualize_data.py 的映射逻辑：
    - 显示做了 flipV（上下翻转）
    - 返回显示窗口中的像素坐标（用于画点）
    """
    px_i = float(np.clip(px, 0.0, TOF_W - 1.0))
    py_i = float(np.clip(py, 0.0, TOF_H - 1.0))
    py_disp = (TOF_H - 1.0) - py_i
    dx = int((px_i + 0.5) * TOF_SHOW_W / TOF_W)
    dy = int((py_disp + 0.5) * TOF_SHOW_H / TOF_H)
    dx = int(np.clip(dx, 0, TOF_SHOW_W - 1))
    dy = int(np.clip(dy, 0, TOF_SHOW_H - 1))
    return dx, dy


def _depth_to_u8(depth_m: np.ndarray, *, near_sat_m: float = 1.0, max_range_m: float = 20.0) -> np.ndarray:
    """与 visualize_data.py 一致：I≈255/x(m)，0 表示无效。"""
    if depth_m.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    dm = depth_m.astype(np.float32, copy=False)
    out = np.zeros(dm.shape, dtype=np.uint8)
    m = dm > 0
    if not np.any(m):
        return out
    dm2 = np.clip(dm[m], float(near_sat_m), float(max_range_m))
    out[m] = np.clip(np.rint(255.0 / dm2), 0.0, 255.0).astype(np.uint8)
    return out


def _tof_intensity_to_u8(intensity_sum: np.ndarray) -> np.ndarray:
    """
    与 visualize_data.py 一致的强度显示策略：
    - 先按整图平均值做归一化到 target_mean
    - 再做 gamma 显示（1/gamma）
    """
    if intensity_sum.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    v = np.asarray(intensity_sum, dtype=np.float32)
    mean = float(np.mean(v)) if v.size else 0.0
    if mean <= 0.0:
        return np.zeros(v.shape, dtype=np.uint8)

    k = mean / float(TOF_INTEN_TARGET_MEAN)
    k = max(k, 1e-6)
    n = v / k
    n = np.clip(n, 0.0, 1.0)
    if float(TOF_INTEN_GAMMA) > 0:
        n = np.power(n, 1.0 / float(TOF_INTEN_GAMMA))
    return np.clip(np.rint(n * 255.0), 0.0, 255.0).astype(np.uint8)


def _lidar_xyz_to_uv(x: float, y: float, z: float) -> tuple[int, int, bool]:
    """
    把 LiDAR 3D 坐标投影到 700x700 的 2D 图（与 client.py/visualize_data.py 一致）。
    返回 (u,v,in_fov)。
    """
    if x <= 0:
        return 0, 0, False
    yaw = float(np.arctan2(y, x))
    pitch = float(np.arctan2(z, x))
    in_fov = (abs(yaw) <= HALF_FOV) and (abs(pitch) <= HALF_FOV)
    u = int(((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)))
    v = int(((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)))
    u = int(np.clip(u, 0, LIDAR_IMG_W - 1))
    v = int(np.clip(v, 0, LIDAR_IMG_H - 1))
    return u, v, in_fov


def _prefilter_lidar_points(pts: np.ndarray) -> np.ndarray:
    """基础预过滤：只保留前方 + 量程裁剪。"""
    if pts.shape[0] == 0:
        return pts
    p = pts
    m = (p[:, 0] > 0.0) & (p[:, 0] <= float(MAX_RANGE_M))
    return p[m]


def _maybe_subsample(pts: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] <= int(n_max):
        return pts
    idx = rng.choice(pts.shape[0], size=int(n_max), replace=False)
    return pts[idx]


def _sphere_from_4pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[tuple[np.ndarray, float]]:
    """
    由 4 个点确定一个球（一般位置）。
    返回 (center(3,), radius) 或 None（退化/不可解）。
    """
    p1 = p1.astype(np.float64, copy=False)
    p2 = p2.astype(np.float64, copy=False)
    p3 = p3.astype(np.float64, copy=False)
    p4 = p4.astype(np.float64, copy=False)

    A = np.stack([p2 - p1, p3 - p1, p4 - p1], axis=0) * 2.0  # (3,3)
    b = np.array([np.dot(p2, p2) - np.dot(p1, p1), np.dot(p3, p3) - np.dot(p1, p1), np.dot(p4, p4) - np.dot(p1, p1)], dtype=np.float64)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    try:
        c = np.linalg.solve(A, b)
    except Exception:
        return None
    r = float(np.linalg.norm(c - p1))
    if not np.isfinite(r) or not np.all(np.isfinite(c)):
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
    A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], 2.0 * p[:, 2], np.ones((p.shape[0],), dtype=np.float64)])
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


@dataclass(frozen=True)
class SphereFit:
    center: np.ndarray  # (3,) float64
    radius: float
    inliers: np.ndarray  # (N,) bool


def _sphere_edge_coverage_ok(
    pts: np.ndarray,
    *,
    center: np.ndarray,
    radius: float,
    inliers: np.ndarray,
) -> bool:
    """
    质量判定：检查“轮廓点”的方位角覆盖是否足够完整（用来过滤假球/残缺球）。
    - 传感器在原点，球心为 center
    - v = center->sensor 的单位向量（可见半球的极轴）
    - 轮廓点：极角 a = arccos(s·v) 满足 a > 90° - edge_band
    - 对轮廓点计算方位角 b，要求覆盖大多数 bins 且没有很长的空缺口
    """
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
    if pin.shape[0] < int(SPHERE_MIN_INLIERS):
        return False

    s = pin - c.reshape(1, 3)
    sn = np.linalg.norm(s, axis=1)
    m = sn > 1e-6
    if not np.any(m):
        return False
    s = s[m] / sn[m].reshape(-1, 1)

    # 极角 a：0 在正对传感器的“中心”，90° 在轮廓
    cos_a = np.clip(s @ v.reshape(3, 1), -1.0, 1.0).reshape(-1)
    a = np.arccos(cos_a)
    edge_thr = (np.pi / 2.0) - np.deg2rad(float(SPHERE_EDGE_ANGLE_DEG))
    edge = a > float(edge_thr)
    if int(np.count_nonzero(edge)) < int(SPHERE_EDGE_MIN_POINTS):
        return False

    se = s[edge]
    az = np.arctan2(se @ w.reshape(3, 1), se @ u.reshape(3, 1)).reshape(-1)  # [-pi,pi]

    bins = int(SPHERE_EDGE_BINS)
    # 映射到 [0,bins)
    bidx = np.floor((az + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    bidx = np.clip(bidx, 0, bins - 1)
    occ = np.zeros((bins,), dtype=np.uint8)
    occ[bidx] = 1

    occ_cnt = int(np.count_nonzero(occ))
    if occ_cnt < int(np.ceil(bins * float(SPHERE_EDGE_MIN_BIN_FRACTION))):
        return False

    # 环形最大空缺口长度
    # 通过把 occ 拼接两次，找最长连续 0，但最多取 bins 长度
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

    # 为了速度：预先算范数不会特别省；直接循环即可（点数已被上游限制）
    for _ in range(int(iters)):
        idx = rng.choice(n, size=4, replace=False)
        m = _sphere_from_4pts(p[idx[0]], p[idx[1]], p[idx[2]], p[idx[3]])
        if m is None:
            continue
        c, r = m
        if (r < float(r_min)) or (r > float(r_max)):
            continue

        d = np.linalg.norm(p - c.reshape(1, 3), axis=1)
        res = np.abs(d - float(r))
        inliers = res < float(inlier_thresh)
        cnt = int(np.count_nonzero(inliers))
        if cnt > best_cnt and cnt >= int(min_inliers):
            # 质量门槛：必须像“完整球”而不是一小片弧面
            if not _sphere_edge_coverage_ok(p, center=c, radius=float(r), inliers=inliers):
                continue
            best_cnt = cnt
            best_center = c
            best_radius = float(r)
            best_inliers = inliers

    if best_center is None or best_inliers is None or best_cnt < int(min_inliers):
        return None

    # refine：用 inliers 再拟合一次
    refined = _sphere_refine_least_squares(p[best_inliers])
    if refined is not None:
        c2, r2 = refined
        if float(r_min) <= float(r2) <= float(r_max):
            d2 = np.linalg.norm(p - c2.reshape(1, 3), axis=1)
            in2 = np.abs(d2 - float(r2)) < float(inlier_thresh)
            if int(np.count_nonzero(in2)) >= best_cnt:
                if _sphere_edge_coverage_ok(p, center=c2, radius=float(r2), inliers=in2):
                    best_center, best_radius, best_inliers = c2, float(r2), in2
                    best_cnt = int(np.count_nonzero(best_inliers))

    return SphereFit(center=best_center, radius=float(best_radius), inliers=best_inliers)


def _lidar_ball_sphere(
    pts: np.ndarray,
    *,
    tof_hint_xyz: Optional[Tuple[float, float, float]] = None,
    seed: int = 0,
) -> tuple[Optional[Tuple[float, float, float]], Optional[SphereFit], np.ndarray, np.ndarray]:
    """
    返回：
    - ball_center_xyz（若找到）
    - sphere_fit（若找到，包含 inliers）
    - render_points（完整点云：用于显示）
    - inlier_points（球面内点：用于叠加显示）
    """
    if pts.shape[0] == 0:
        return None, None, pts, pts[:0]

    # 完整点云（用于显示）：只做基础预过滤（前方 + 量程裁剪）
    p_render = _prefilter_lidar_points(pts)
    if p_render.shape[0] == 0:
        return None, None, p_render, p_render[:0]

    rng = np.random.default_rng(int(seed))

    # 如果有 TOF 粗定位：只取附近点（大幅提升鲁棒性，且不需要找平面）
    p_roi = p_render
    if tof_hint_xyz is not None:
        hx, hy, hz = tof_hint_xyz
        hh = np.array([hx, hy, hz], dtype=np.float64).reshape(1, 3)
        dist = np.linalg.norm(p_roi.astype(np.float64, copy=False) - hh, axis=1)
        p1 = p_roi[dist < float(SPHERE_PREFILTER_RADIUS_M)]
        if p1.shape[0] >= 200:
            p_roi = p1

    p_fit = _maybe_subsample(p_roi, int(SPHERE_MAX_FIT_POINTS), rng)
    fit = _ransac_sphere(
        p_fit,
        r_min=float(SPHERE_R_MIN_M),
        r_max=float(SPHERE_R_MAX_M),
        iters=int(SPHERE_RANSAC_ITERS),
        inlier_thresh=float(SPHERE_INLIER_THRESH_M),
        min_inliers=int(SPHERE_MIN_INLIERS),
        seed=int(seed),
    )
    if fit is None:
        return None, None, p_render, p_render[:0]

    # 用 p_roi（而不是抽样 p_fit）重新计算 inliers，并再做一次 refine + 质量检查
    c = fit.center.astype(np.float64, copy=False).reshape(3)
    r = float(fit.radius)
    d0 = np.linalg.norm(p_roi.astype(np.float64, copy=False) - c.reshape(1, 3), axis=1)
    in0 = np.abs(d0 - r) < float(SPHERE_INLIER_THRESH_M)
    if int(np.count_nonzero(in0)) < int(SPHERE_MIN_INLIERS):
        return None, None, p_render, p_render[:0]

    refined = _sphere_refine_least_squares(p_roi[in0])
    if refined is not None:
        c2, r2 = refined
        if float(SPHERE_R_MIN_M) <= float(r2) <= float(SPHERE_R_MAX_M):
            d2 = np.linalg.norm(p_roi.astype(np.float64, copy=False) - c2.reshape(1, 3), axis=1)
            in2 = np.abs(d2 - float(r2)) < float(SPHERE_INLIER_THRESH_M)
            if int(np.count_nonzero(in2)) >= int(np.count_nonzero(in0)):
                c, r, in0 = c2, float(r2), in2

    if not _sphere_edge_coverage_ok(p_roi.astype(np.float64, copy=False), center=c, radius=float(r), inliers=in0):
        return None, None, p_render, p_render[:0]

    cx, cy, cz = c.tolist()
    inlier_pts = p_roi[in0].astype(np.float32, copy=False)
    return (float(cx), float(cy), float(cz)), SphereFit(center=c, radius=float(r), inliers=in0), p_render, inlier_pts


def _render_lidar_gray(points_xyz: np.ndarray) -> np.ndarray:
    """
    把 3D 点投影成 2D 灰度（与 visualize_data.py/client.py 一致的 FOV/映射；强度 I≈255/x）。
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    x = points_xyz[:, 0].astype(np.float32, copy=False)
    y = points_xyz[:, 1].astype(np.float32, copy=False)
    z = points_xyz[:, 2].astype(np.float32, copy=False)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    depth_m = np.clip(x.astype(np.float32, copy=False), float(NEAR_SAT_M), float(MAX_RANGE_M))
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0.0, 255.0).astype(np.uint8)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_IMG_W - 1)
    row = np.clip(row, 0, LIDAR_IMG_H - 1)

    img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
    np.maximum.at(img, (row, col), depth_u8)
    return img


def _tof_ball_center(
    tof_raw: Path,
    *,
    window_size: int = 5,
) -> Optional[Dict[str, Any]]:
    if not tof_raw.exists():
        return None

    params = ToF3DParams(
        min_peak_count=float(TOF_MIN_PEAK),
        enable_distance_compensation=bool(TOF_COMP_ENABLE),
        distance_comp_a=float(TOF_COMP_A),
        distance_comp_b_mm=float(TOF_COMP_B_MM),
    )
    depth, hists = tof_distance_and_histograms(tof_raw, params=params)  # depth:(30,40), hists:(30,40,64)
    if depth.size == 0 or hists.size == 0:
        return None

    # 反射率强度：直方图求和
    inten = hists.sum(axis=2).astype(np.float32, copy=False)  # (30,40)
    # 置信度：峰值太小的像素认为无效
    peak = hists[:, :, : int(TOF_VALID_BINS)].max(axis=2).astype(np.float32, copy=False)
    inten = np.where(peak >= float(TOF_MIN_PEAK), inten, 0.0)
    if float(np.max(inten)) <= 0.0:
        return None

    # 最亮点（整数像素）
    flat_idx = int(np.argmax(inten))
    y0, x0 = int(flat_idx // TOF_W), int(flat_idx % TOF_W)

    # 5x5 窗口反射率重心
    r = int(window_size) // 2
    xs = np.arange(max(0, x0 - r), min(TOF_W, x0 + r + 1), dtype=np.float32)
    ys = np.arange(max(0, y0 - r), min(TOF_H, y0 + r + 1), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    w = inten[ys.astype(np.int32)[:, None], xs.astype(np.int32)[None, :]].astype(np.float32, copy=False)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        cx, cy = float(x0), float(y0)
    else:
        cx = float(np.sum(xx * w) / wsum)
        cy = float(np.sum(yy * w) / wsum)

    # 3D（x,y,z，单位 m）：用深度 + 像素角度还原
    # 约定：x 正前方；y 向左；z 向上
    # 为了与 LiDAR 的 yaw 正方向一致（y>0 => yaw>0 => 左侧），这里让 x=0 对应 +FOV/2（左侧）
    yaw_x = np.deg2rad(np.linspace(params.fov_x_deg / 2.0, -params.fov_x_deg / 2.0, TOF_W)).astype(np.float32)
    pitch_y = np.deg2rad(np.linspace(-params.fov_y_deg / 2.0, params.fov_y_deg / 2.0, TOF_H)).astype(np.float32)

    # 在窗口内算 3D 重心（权重=反射率）
    xs_i = xs.astype(np.int32)
    ys_i = ys.astype(np.int32)
    yaws = yaw_x[xs_i]          # (wx,)
    pitchs = pitch_y[ys_i]      # (wy,)
    yaw_grid, pitch_grid = np.meshgrid(yaws, pitchs)  # (wy,wx)

    depth_win = depth[ys_i[:, None], xs_i[None, :]].astype(np.float32, copy=False)  # (wy,wx)
    valid = depth_win > 0
    # depth_win = R*cos(yaw)*cos(pitch) => R = depth/(cos*cos)
    denom = (np.cos(yaw_grid) * np.cos(pitch_grid)).astype(np.float32)
    denom = np.where(np.abs(denom) > 1e-6, denom, 1e-6)
    R = np.where(valid, depth_win / denom, 0.0)
    X = np.where(valid, R * np.cos(yaw_grid) * np.cos(pitch_grid), 0.0)  # == depth_win
    Y = np.where(valid, R * np.sin(yaw_grid) * np.cos(pitch_grid), 0.0)
    Z = np.where(valid, R * np.sin(pitch_grid), 0.0)

    w3 = np.where(valid, w, 0.0)
    w3sum = float(np.sum(w3))
    if w3sum > 0.0:
        x3 = float(np.sum(X * w3) / w3sum)
        y3 = float(np.sum(Y * w3) / w3sum)
        z3 = float(np.sum(Z * w3) / w3sum)
    else:
        # 退化：用最亮点的 3D
        x3 = float(depth[y0, x0])
        y3 = 0.0
        z3 = 0.0

    return {
        "brightest_pixel_xy": [int(x0), int(y0)],
        "centroid_pixel_xy": [float(cx), float(cy)],
        "centroid_xyz_m": [float(x3), float(y3), float(z3)],
    }


def _draw_cross(img_bgr: np.ndarray, u: int, v: int, *, color: tuple[int, int, int], r: int = 8, t: int = 2) -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        return
    cv2.line(img_bgr, (u - r, v), (u + r, v), color, t, cv2.LINE_AA)
    cv2.line(img_bgr, (u, v - r), (u, v + r), color, t, cv2.LINE_AA)
    cv2.circle(img_bgr, (u, v), max(2, t), (0, 0, 0), -1, cv2.LINE_AA)


def _build_views(
    env_dir: Path,
    *,
    lidar_center: Optional[Tuple[float, float, float]],
    lidar_sphere: Optional[SphereFit],
    lidar_render_points: np.ndarray,
    lidar_inlier_points: np.ndarray,
    tof_info: Optional[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e

    meta: Dict[str, Any] = {"env": env_dir.name}

    # LiDAR：两张图
    lidar_raw_bgr = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)
    lidar_ball_bgr = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)
    tof_depth_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_inten_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)

    # --- LiDAR：不再依赖 view.png，直接从点云渲染 2D ---
    raw_u8 = _render_lidar_gray(lidar_render_points if lidar_render_points.shape[0] else np.zeros((0, 3), dtype=np.float32))
    lidar_raw_bgr = cv2.applyColorMap(raw_u8, cv2.COLORMAP_TURBO)
    lidar_ball_bgr = lidar_raw_bgr.copy()

    if lidar_sphere is not None:
        meta["lidar_sphere_center_xyz_m"] = [float(x) for x in lidar_sphere.center.tolist()]
        meta["lidar_sphere_radius_m"] = float(lidar_sphere.radius)
        meta["lidar_sphere_inliers"] = int(lidar_inlier_points.shape[0])

        # 把球面 inliers 投影出来高亮
        in_pts = lidar_inlier_points.astype(np.float32, copy=False)
        if in_pts.shape[0] > 0:
            xs = in_pts[:, 0]
            ys = in_pts[:, 1]
            zs = in_pts[:, 2]
            yaw = np.arctan2(ys, xs)
            pitch = np.arctan2(zs, xs)
            m = (xs > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
            xs, ys, zs, yaw, pitch = xs[m], ys[m], zs[m], yaw[m], pitch[m]
            col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
            row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
            col = np.clip(col, 0, LIDAR_IMG_W - 1)
            row = np.clip(row, 0, LIDAR_IMG_H - 1)
            # 红色点覆盖
            lidar_ball_bgr[row, col] = (0, 0, 255)

    if lidar_center is not None:
        lx, ly, lz = lidar_center
        meta["lidar_ball_center_xyz_m"] = [lx, ly, lz]
        u, v, in_fov = _lidar_xyz_to_uv(lx, ly, lz)
        meta["lidar_ball_center_uv"] = [int(u), int(v)]
        meta["lidar_ball_center_in_fov"] = bool(in_fov)
        _draw_cross(lidar_ball_bgr, u, v, color=(0, 255, 255), r=10, t=2)
        cv2.putText(
            lidar_ball_bgr,
            f"ball(lidar): center=({lx:.3f},{ly:.3f},{lz:.3f})m  r={meta.get('lidar_sphere_radius_m', 0.0):.3f}m  in={meta.get('lidar_sphere_inliers', 0)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            lidar_ball_bgr,
            f"{env_dir.name}: sphere not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # --- TOF 标注：生成两张 2D 图（depth + intensity）并画球心 ---
    tof_path = env_dir / "tof.raw"
    if tof_info is not None and tof_path.exists():
        meta["tof"] = tof_info
        params = ToF3DParams(
            min_peak_count=float(TOF_MIN_PEAK),
            enable_distance_compensation=bool(TOF_COMP_ENABLE),
            distance_comp_a=float(TOF_COMP_A),
            distance_comp_b_mm=float(TOF_COMP_B_MM),
        )
        depth, hists = tof_distance_and_histograms(tof_path, params=params)
        inten = hists.sum(axis=2).astype(np.float32, copy=False)

        # depth 图（colormap turbo）
        u8 = _depth_to_u8(depth)
        u8_big = cv2.resize(u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
        u8_big = cv2.flip(u8_big, 0)
        tof_depth_bgr = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)

        # intensity 图（灰度）
        inten_u8 = _tof_intensity_to_u8(inten)
        inten_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
        inten_big = cv2.flip(inten_big, 0)
        tof_inten_bgr = cv2.cvtColor(inten_big, cv2.COLOR_GRAY2BGR)

        cx, cy = tof_info["centroid_pixel_xy"]
        dx, dy = _tof_pixel_to_disp_xy(float(cx), float(cy))
        _draw_cross(tof_depth_bgr, dx, dy, color=(255, 255, 255), r=10, t=2)
        _draw_cross(tof_inten_bgr, dx, dy, color=(255, 255, 255), r=10, t=2)

        x3, y3, z3 = tof_info.get("centroid_xyz_m", [0.0, 0.0, 0.0])
        cv2.putText(
            tof_depth_bgr,
            f"ball(tof): px=({cx:.2f},{cy:.2f})  xyz=({x3:.3f},{y3:.3f},{z3:.3f}) m",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tof_inten_bgr,
            f"ball(tof): px=({cx:.2f},{cy:.2f})  xyz=({x3:.3f},{y3:.3f},{z3:.3f}) m",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            tof_depth_bgr,
            f"{env_dir.name}: missing tof.raw / tof center not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tof_inten_bgr,
            f"{env_dir.name}: missing tof.raw / tof center not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 注意：这里返回 LiDAR_RAW/LiDAR_BALL/TOF_DEPTH/TOF_INTEN
    return lidar_raw_bgr, lidar_ball_bgr, tof_depth_bgr, tof_inten_bgr, meta


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e

    data_root = Path(DATA_DIR)
    envs = _list_env_dirs(data_root)
    if not envs:
        raise FileNotFoundError(f"data 目录下没有场景：{data_root}")

    cv2.namedWindow("LiDAR_RAW", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("LiDAR_BALL", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_DEPTH", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_INTEN", cv2.WINDOW_AUTOSIZE)

    idx = 0
    while True:
        env = envs[idx]
        npz = _find_points_npz(env)
        pts = _load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)
        tof_info = _tof_ball_center(env / "tof.raw", window_size=5)

        tof_hint = None
        if tof_info is not None and "centroid_xyz_m" in tof_info:
            try:
                tox, toy, toz = tof_info["centroid_xyz_m"]
                tof_hint = (float(tox), float(toy), float(toz))
            except Exception:
                tof_hint = None

        lidar_center, lidar_sphere, lidar_render_pts, lidar_inlier_pts = _lidar_ball_sphere(pts, tof_hint_xyz=tof_hint, seed=0)

        lidar_raw, lidar_ball, tof_depth, tof_inten, meta = _build_views(
            env,
            lidar_center=lidar_center,
            lidar_sphere=lidar_sphere,
            lidar_render_points=lidar_render_pts,
            lidar_inlier_points=lidar_inlier_pts,
            tof_info=tof_info,
        )

        # 左上角统一显示场景索引
        cv2.putText(
            lidar_raw,
            f"{env.name}  ({idx+1}/{len(envs)})",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            lidar_ball,
            f"{env.name}  ({idx+1}/{len(envs)})",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tof_depth,
            f"{env.name}  ({idx+1}/{len(envs)})",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tof_inten,
            f"{env.name}  ({idx+1}/{len(envs)})",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LiDAR_RAW", lidar_raw)
        cv2.imshow("LiDAR_BALL", lidar_ball)
        cv2.imshow("TOF_DEPTH", tof_depth)
        cv2.imshow("TOF_INTEN", tof_inten)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:  # ESC
            break
        if k == ord("4"):
            idx = (idx - 1) % len(envs)
        if k == ord("6"):
            idx = (idx + 1) % len(envs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


