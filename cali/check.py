#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cali/check.py

遍历 cali/data/<场景>/，同时显示：
- 3D ToF：Depth / Intensity，并标注“最亮点周围 5x5 的反射率加权重心”
- LiDAR：点云投影图（RAW + BALL），标注球心与球面内点

交互：
- 4：上一个场景
- 6：下一个场景
- ESC：退出
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 允许在 cali/ 目录直接运行：把项目根目录加进 sys.path 以便 import tof3d
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ball_finder import BallFindResult, find_ball_from_lidar  # noqa: E402
from tof3d import ToF3DParams, tof_distance_matrix, tof_histograms  # noqa: E402


# ========= 数据目录 =========
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# ========= LiDAR 2D（与 visualize_data.py/client.py 对齐）=========
LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = float(np.deg2rad(FOV_DEG / 2.0))

# LiDAR 投影渲染（仅用于显示）
LIDAR_VIS_MAX_RANGE_M = 20.0
LIDAR_NEAR_SAT_M = 1.0

# 竖直分割线（7°一条线，和 ball_finder.py 的窗口步长一致）
DRAW_ELEV_LINES = True
ELEV_LINE_DEG = 7.0
ELEV_LINE_COLOR = (255, 255, 255)  # BGR
ELEV_LINE_THICKNESS = 1

# 鼠标悬停显示距离
MOUSE_HOVER_ENABLED = True

# ========= TOF 2D（显示尺寸）=========
TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 400
TOF_SHOW_H = 300

TOF_MIN_PEAK = 100
TOF_VALID_BINS = 62
TOF_INTEN_GAMMA = 2.2
TOF_INTEN_TARGET_MEAN = 0.18


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
    与 visualize_data.py 一致的显示映射：
    - 显示做了 flipV（上下翻转）
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
    与 visualize_data.py 类似的强度显示策略：
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


def _tof_ball_center(tof_raw: Path, *, window_size: int = 5) -> Optional[Dict[str, Any]]:
    if not tof_raw.exists():
        return None

    params = ToF3DParams(min_peak_count=float(TOF_MIN_PEAK))
    depth = tof_distance_matrix(tof_raw, params=params)  # (30,40) m
    hists = tof_histograms(tof_raw, params=params).astype(np.float32, copy=False)  # (30,40,64)
    if depth.size == 0 or hists.size == 0:
        return None

    inten = hists.sum(axis=2).astype(np.float32, copy=False)  # (30,40)
    peak = hists[:, :, : int(TOF_VALID_BINS)].max(axis=2).astype(np.float32, copy=False)
    inten = np.where(peak >= float(TOF_MIN_PEAK), inten, 0.0)
    if float(np.max(inten)) <= 0.0:
        return None

    flat_idx = int(np.argmax(inten))
    y0, x0 = int(flat_idx // TOF_W), int(flat_idx % TOF_W)

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

    yaw_x = np.deg2rad(np.linspace(params.fov_x_deg / 2.0, -params.fov_x_deg / 2.0, TOF_W)).astype(np.float32)
    pitch_y = np.deg2rad(np.linspace(-params.fov_y_deg / 2.0, params.fov_y_deg / 2.0, TOF_H)).astype(np.float32)

    xs_i = xs.astype(np.int32)
    ys_i = ys.astype(np.int32)
    yaws = yaw_x[xs_i]
    pitchs = pitch_y[ys_i]
    yaw_grid, pitch_grid = np.meshgrid(yaws, pitchs)

    depth_win = depth[ys_i[:, None], xs_i[None, :]].astype(np.float32, copy=False)
    valid = depth_win > 0
    denom = (np.cos(yaw_grid) * np.cos(pitch_grid)).astype(np.float32)
    denom = np.where(np.abs(denom) > 1e-6, denom, 1e-6)
    R = np.where(valid, depth_win / denom, 0.0)
    X = np.where(valid, R * np.cos(yaw_grid) * np.cos(pitch_grid), 0.0)
    Y = np.where(valid, R * np.sin(yaw_grid) * np.cos(pitch_grid), 0.0)
    Z = np.where(valid, R * np.sin(pitch_grid), 0.0)

    w3 = np.where(valid, w, 0.0)
    w3sum = float(np.sum(w3))
    if w3sum > 0.0:
        x3 = float(np.sum(X * w3) / w3sum)
        y3 = float(np.sum(Y * w3) / w3sum)
        z3 = float(np.sum(Z * w3) / w3sum)
    else:
        x3, y3, z3 = float(depth[y0, x0]), 0.0, 0.0

    return {
        "brightest_pixel_xy": [int(x0), int(y0)],
        "centroid_pixel_xy": [float(cx), float(cy)],
        "centroid_xyz_m": [float(x3), float(y3), float(z3)],
    }


def _render_lidar_gray_and_range(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_xyz.shape[0] == 0:
        img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
        rmap = np.full((LIDAR_IMG_H, LIDAR_IMG_W), np.inf, dtype=np.float32)
        return img, rmap

    x = points_xyz[:, 0].astype(np.float32, copy=False)
    y = points_xyz[:, 1].astype(np.float32, copy=False)
    z = points_xyz[:, 2].astype(np.float32, copy=False)
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, y, z, yaw, pitch = x[m], y[m], z[m], yaw[m], pitch[m]
    if x.size == 0:
        img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
        rmap = np.full((LIDAR_IMG_H, LIDAR_IMG_W), np.inf, dtype=np.float32)
        return img, rmap

    depth_m = np.clip(x.astype(np.float32, copy=False), float(LIDAR_NEAR_SAT_M), float(LIDAR_VIS_MAX_RANGE_M))
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0.0, 255.0).astype(np.uint8)

    col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_IMG_W - 1)
    row = np.clip(row, 0, LIDAR_IMG_H - 1)

    img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
    np.maximum.at(img, (row, col), depth_u8)

    r = np.sqrt(x * x + y * y + z * z).astype(np.float32, copy=False)
    rmap = np.full((LIDAR_IMG_H, LIDAR_IMG_W), np.inf, dtype=np.float32)
    np.minimum.at(rmap, (row, col), r)
    return img, rmap


def _draw_cross(img_bgr: np.ndarray, u: int, v: int, *, color: tuple[int, int, int], r: int = 8, t: int = 2) -> None:
    import cv2  # type: ignore

    cv2.line(img_bgr, (u - r, v), (u + r, v), color, t, cv2.LINE_AA)
    cv2.line(img_bgr, (u, v - r), (u, v + r), color, t, cv2.LINE_AA)
    cv2.circle(img_bgr, (u, v), max(2, t), (0, 0, 0), -1, cv2.LINE_AA)


def _draw_elev_lines(img_bgr: np.ndarray, *, step_deg: float) -> None:
    import cv2  # type: ignore

    if not bool(DRAW_ELEV_LINES):
        return
    if (not np.isfinite(step_deg)) or step_deg <= 0.0:
        return
    half_deg = float(np.rad2deg(HALF_FOV))
    k_min = int(np.floor(-half_deg / step_deg))
    k_max = int(np.ceil(half_deg / step_deg))
    for k in range(k_min, k_max + 1):
        deg = float(k) * float(step_deg)
        if deg <= -half_deg + 1e-6 or deg >= half_deg - 1e-6:
            continue
        pitch = float(np.deg2rad(deg))
        v = int(((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)))
        v = int(np.clip(v, 0, LIDAR_IMG_H - 1))
        cv2.line(img_bgr, (0, v), (LIDAR_IMG_W - 1, v), ELEV_LINE_COLOR, int(ELEV_LINE_THICKNESS), cv2.LINE_AA)


def _build_lidar_views(res: BallFindResult) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    import cv2  # type: ignore

    meta: Dict[str, Any] = {}
    raw_u8, raw_range = _render_lidar_gray_and_range(res.render_points.astype(np.float32, copy=False))
    ball_u8, ball_range = _render_lidar_gray_and_range(res.filtered_points.astype(np.float32, copy=False))

    lidar_raw_bgr = cv2.applyColorMap(raw_u8, cv2.COLORMAP_TURBO)
    lidar_ball_bgr = cv2.applyColorMap(ball_u8, cv2.COLORMAP_TURBO)

    _draw_elev_lines(lidar_raw_bgr, step_deg=float(ELEV_LINE_DEG))
    _draw_elev_lines(lidar_ball_bgr, step_deg=float(ELEV_LINE_DEG))

    meta["lidar_raw_range_map"] = raw_range
    meta["lidar_ball_range_map"] = ball_range

    if res.inlier_points.shape[0] > 0:
        in_pts = res.inlier_points.astype(np.float32, copy=False)
        xs, ys, zs = in_pts[:, 0], in_pts[:, 1], in_pts[:, 2]
        yaw = np.arctan2(ys, xs)
        pitch = np.arctan2(zs, np.hypot(xs, ys))
        m = (xs > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
        xs, ys, zs, yaw, pitch = xs[m], ys[m], zs[m], yaw[m], pitch[m]
        col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
        row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
        col = np.clip(col, 0, LIDAR_IMG_W - 1)
        row = np.clip(row, 0, LIDAR_IMG_H - 1)
        lidar_ball_bgr[row, col] = (0, 0, 255)

    if res.center_xyz is not None:
        lx, ly, lz = res.center_xyz
        if lx > 0:
            yaw = float(np.arctan2(ly, lx))
            pitch = float(np.arctan2(lz, float(np.hypot(lx, ly))))
            u = int(((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)))
            v = int(((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)))
            u = int(np.clip(u, 0, LIDAR_IMG_W - 1))
            v = int(np.clip(v, 0, LIDAR_IMG_H - 1))
            _draw_cross(lidar_ball_bgr, u, v, color=(0, 255, 255), r=10, t=2)
        r = float(res.sphere.radius) if res.sphere is not None else 0.0
        cv2.putText(
            lidar_ball_bgr,
            f"ball(lidar): center=({lx:.3f},{ly:.3f},{lz:.3f})m  r={r:.3f}m  in={int(res.inlier_points.shape[0])}",
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
            "ball(lidar): not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return lidar_raw_bgr, lidar_ball_bgr, meta


def _build_tof_views(env_dir: Path, tof_info: Optional[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    import cv2  # type: ignore

    tof_depth_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_inten_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_path = env_dir / "tof.raw"

    if tof_info is None or (not tof_path.exists()):
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
        return tof_depth_bgr, tof_inten_bgr

    params = ToF3DParams(min_peak_count=float(TOF_MIN_PEAK))
    depth = tof_distance_matrix(tof_path, params=params)
    hists = tof_histograms(tof_path, params=params).astype(np.float32, copy=False)
    inten = hists.sum(axis=2).astype(np.float32, copy=False)

    u8 = _depth_to_u8(depth)
    u8_big = cv2.resize(u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
    u8_big = cv2.flip(u8_big, 0)
    tof_depth_bgr = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)

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
    return tof_depth_bgr, tof_inten_bgr


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e

    envs = _list_env_dirs(Path(DATA_DIR))
    if not envs:
        raise FileNotFoundError(f"data 目录下没有场景：{DATA_DIR}")

    cv2.namedWindow("LiDAR_RAW", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("LiDAR_BALL", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_DEPTH", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_INTEN", cv2.WINDOW_AUTOSIZE)

    mouse_state: dict = {"x": 0, "y": 0}

    def _on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if not bool(MOUSE_HOVER_ENABLED):
            return
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse_state["x"] = int(x)
        mouse_state["y"] = int(y)

    cv2.setMouseCallback("LiDAR_RAW", _on_mouse)
    cv2.setMouseCallback("LiDAR_BALL", _on_mouse)

    idx = 0
    while True:
        env = envs[idx]
        npz = _find_points_npz(env)
        pts = _load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)

        tof_info = _tof_ball_center(env / "tof.raw", window_size=5)
        tof_depth, tof_inten = _build_tof_views(env, tof_info)

        res: BallFindResult = find_ball_from_lidar(pts, seed=0)
        lidar_raw, lidar_ball, meta = _build_lidar_views(res)

        # 场景索引
        for img in (lidar_raw, lidar_ball):
            cv2.putText(img, f"{env.name}  ({idx+1}/{len(envs)})", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        for img in (tof_depth, tof_inten):
            cv2.putText(img, f"{env.name}  ({idx+1}/{len(envs)})", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 鼠标悬停距离
        if bool(MOUSE_HOVER_ENABLED):
            mx = int(np.clip(mouse_state.get("x", 0), 0, LIDAR_IMG_W - 1))
            my = int(np.clip(mouse_state.get("y", 0), 0, LIDAR_IMG_H - 1))
            rr = float(meta["lidar_raw_range_map"][my, mx])
            br = float(meta["lidar_ball_range_map"][my, mx])
            txt_raw = f"r={rr:.3f}m" if np.isfinite(rr) else "r=--"
            txt_ball = f"r={br:.3f}m" if np.isfinite(br) else "r=--"
            cv2.putText(lidar_raw, txt_raw, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(lidar_ball, txt_ball, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("LiDAR_RAW", lidar_raw)
        cv2.imshow("LiDAR_BALL", lidar_ball)
        cv2.imshow("TOF_DEPTH", tof_depth)
        cv2.imshow("TOF_INTEN", tof_inten)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:
            break
        if k == ord("4"):
            idx = (idx - 1) % len(envs)
        if k == ord("6"):
            idx = (idx + 1) % len(envs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


