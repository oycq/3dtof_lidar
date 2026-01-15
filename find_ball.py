#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
find_ball.py

遍历 data/<场景>/，自动估计球的中心坐标，并以 OpenCV imshow 的方式交互显示（不落盘）。

球心算法（按你的要求）：
- 只使用 LiDAR 点云找球心（完全不使用 3DToF，也不会用 3DToF 的结果去引导 LiDAR）。
  具体的点云找球算法与参数均封装在同级目录的 ball_finder.py 内。

使用方式（按你的要求：不传参数）：
- 直接运行：py .\\find_ball.py
- 关键配置都在本文件顶部“宏定义”区域：DATA_DIR / OVERWRITE / RANSAC 参数等

交互：
- 4：上一个场景
- 6：下一个场景
- ESC：退出
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ball_finder import BallFindResult, SphereFit, find_ball_from_lidar


# ========= 与 visualize_data.py/client.py 对齐的一些常量 =========
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# LiDAR 2D（view.png 是 client.py 保存的 700x700）
LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = float(np.deg2rad(FOV_DEG / 2.0))

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
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e

    meta: dict = {"env": env_dir.name}

    # LiDAR：两张图
    lidar_raw_bgr = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)
    lidar_ball_bgr = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)

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

    # 注意：只返回 LiDAR_RAW/LiDAR_BALL
    return lidar_raw_bgr, lidar_ball_bgr, meta


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

    idx = 0
    while True:
        env = envs[idx]
        npz = _find_points_npz(env)
        pts = _load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)
        res: BallFindResult = find_ball_from_lidar(pts, seed=0)
        lidar_raw, lidar_ball, meta = _build_views(
            env,
            lidar_center=res.center_xyz,
            lidar_sphere=res.sphere,
            lidar_render_points=res.render_points,
            lidar_inlier_points=res.inlier_points,
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
        cv2.imshow("LiDAR_RAW", lidar_raw)
        cv2.imshow("LiDAR_BALL", lidar_ball)

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


