#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cali/check.py

遍历 cali/data/<scene>/, 做可视化:
- TOF_REFLECT: ToF 反射率(强度)灰度图, 叠加球心十字
- LIDAR: 雷达点云投影图, 叠加拟合内点(红色), 球心(十字), 以及文字信息

按键:
- 4: 上一个场景
- 6: 下一个场景
- 0: 删除当前场景(弹窗确认)
- ESC: 退出
"""

from __future__ import annotations

import ctypes
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 允许在 cali/ 目录直接运行: 把项目根目录加入 sys.path, 以便 import tof3d.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lidar_ball_detector import LidarBallDetection, detect_ball_lidar  # noqa: E402
from tof3d import ToF3DParams, tof_histograms, tof_reflectance_mean3_max  # noqa: E402
from tof_ball_detector import detect_ball_tof_2d  # noqa: E402


# ========= 数据目录 =========
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# ========= LiDAR 2D 显示参数, 与 visualize_data.py/client.py 对齐 =========
LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = float(np.deg2rad(FOV_DEG / 2.0))

# LiDAR 投影渲染参数, 仅用于显示.
LIDAR_VIS_MAX_RANGE_M = 20.0
LIDAR_NEAR_SAT_M = 1.0

# 可选的竖直辅助线, 按需求默认关闭.
DRAW_ELEV_LINES = False
ELEV_LINE_DEG = 7.0
ELEV_LINE_COLOR = (255, 255, 255)  # BGR
ELEV_LINE_THICKNESS = 1

# 鼠标悬停显示 LiDAR 距离, 只作用于 LIDAR 窗口.
MOUSE_HOVER_ENABLED = True

# ========= ToF 2D 显示尺寸 =========
TOF_W = 40
TOF_H = 30
# 与 run.py / visualize_data.py 对齐：显示做“向右旋转90° + 水平翻转”，宽高对调
TOF_SHOW_W = 300
TOF_SHOW_H = 400

TOF_INTEN_GAMMA = 1
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
    与 run.py / visualize_data.py 对齐的显示映射:
    - 显示做“向右旋转90° + 水平翻转”(rot90CW + flipH).
    """
    px_i = float(np.clip(px, 0.0, TOF_W - 1.0))
    py_i = float(np.clip(py, 0.0, TOF_H - 1.0))
    # rot90CW + flipH 后，变换图像坐标 (row, col) = (px, py)
    dx = int((py_i + 0.5) * TOF_SHOW_W / TOF_H)
    dy = int((px_i + 0.5) * TOF_SHOW_H / TOF_W)
    dx = int(np.clip(dx, 0, TOF_SHOW_W - 1))
    dy = int(np.clip(dy, 0, TOF_SHOW_H - 1))
    return dx, dy


def _tof_intensity_to_u8(intensity_sum: np.ndarray) -> np.ndarray:
    """
    ToF 强度显示映射:
    - 先按整图 mean 做归一化到 target_mean
    - 再做 gamma 显示(1/gamma)
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


def _render_lidar_gray_and_range(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 把 3D 点投影成 2D 灰度图, 并输出每像素的最近距离 rmap.
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
    # 绘制竖直辅助线, 当前默认关闭.
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


def _confirm_delete_scene(scene_dir: Path) -> bool:
    """
    弹出阻塞确认框.
    用户点击 Yes 返回 True, 否则返回 False.
    """
    title = "Confirm delete"
    msg = f"Delete scene folder?\n\n{scene_dir.name}\n\nThis cannot be undone."
    try:
        # MessageBoxW: returns IDYES (6) or IDNO (7)
        MB_YESNO = 0x00000004
        MB_ICONWARNING = 0x00000030
        IDYES = 6
        ret = ctypes.windll.user32.MessageBoxW(0, msg, title, MB_YESNO | MB_ICONWARNING)
        return int(ret) == IDYES
    except Exception:
        # Fallback: do not delete if dialog is not available.
        return False


def _build_lidar_view(det: LidarBallDetection) -> tuple[np.ndarray, Dict[str, Any]]:
    import cv2  # type: ignore

    meta: Dict[str, Any] = {}

    # 先用 render_points 渲染底图, 再叠加拟合内点(红色)和球心(十字).
    raw_u8, raw_range = _render_lidar_gray_and_range(det.render_points_xyz_m.astype(np.float32, copy=False))
    lidar_bgr = cv2.applyColorMap(raw_u8, cv2.COLORMAP_TURBO)
    _draw_elev_lines(lidar_bgr, step_deg=float(ELEV_LINE_DEG))
    meta["lidar_range_map"] = raw_range

    if det.fit_points_xyz_m.shape[0] > 0:
        in_pts = det.fit_points_xyz_m.astype(np.float32, copy=False)
        xs, ys, zs = in_pts[:, 0], in_pts[:, 1], in_pts[:, 2]
        yaw = np.arctan2(ys, xs)
        pitch = np.arctan2(zs, np.hypot(xs, ys))
        m = (xs > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
        xs, ys, zs, yaw, pitch = xs[m], ys[m], zs[m], yaw[m], pitch[m]
        col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
        row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
        col = np.clip(col, 0, LIDAR_IMG_W - 1)
        row = np.clip(row, 0, LIDAR_IMG_H - 1)
        lidar_bgr[row, col] = (0, 0, 255)

    if det.center_xyz_m is not None:
        lx, ly, lz = det.center_xyz_m
        if lx > 0:
            yaw = float(np.arctan2(ly, lx))
            pitch = float(np.arctan2(lz, float(np.hypot(lx, ly))))
            u = int(((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)))
            v = int(((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)))
            u = int(np.clip(u, 0, LIDAR_IMG_W - 1))
            v = int(np.clip(v, 0, LIDAR_IMG_H - 1))
            _draw_cross(lidar_bgr, u, v, color=(0, 255, 255), r=10, t=2)
        r_m = float(det.radius_m) if det.radius_m is not None else 0.0
        d_m = float(det.center_range_m) if det.center_range_m is not None else 0.0
        cv2.putText(
            lidar_bgr,
            f"lidar: c=({lx:.3f},{ly:.3f},{lz:.3f})m  r={r_m:.3f}m  d={d_m:.3f}m  in={int(det.fit_points_xyz_m.shape[0])}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            lidar_bgr,
            "lidar: not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return lidar_bgr, meta


def _build_tof_reflect_view(env_dir: Path, tof_center_xy: Optional[tuple[float, float]]) -> np.ndarray:
    import cv2  # type: ignore

    tof_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_path = env_dir / "tof.raw"

    if not tof_path.exists():
        cv2.putText(
            f"{env_dir.name}: missing tof.raw / tof center not found",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return tof_bgr

    # 可视化用的最小峰值过滤（只影响显示，不影响检测）。
    # 这里不再依赖 tof_ball_detector.py 的固定阈值常量，避免检测与显示耦合。
    params = ToF3DParams(min_peak_count=0.0)
    hists = tof_histograms(tof_path, params=params).astype(np.float32, copy=False)
    if hists.size == 0:
        return tof_bgr

    # 反射率强度：交给 tof3d.py 里的统一策略（见 tof_reflectance_mean3_max 默认配置）
    # 与 visualize_data.py 保持一致: 这里不做 peak mask, 否则容易看起来像二值黑白图.
    inten = tof_reflectance_mean3_max(hists)

    inten_u8 = _tof_intensity_to_u8(inten)
    # 显示方向与 run.py / visualize_data.py 对齐：向右旋转90° + 水平翻转
    inten_u8 = cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE)
    inten_u8 = cv2.flip(inten_u8, 1)
    inten_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
    tof_bgr = cv2.cvtColor(inten_big, cv2.COLOR_GRAY2BGR)

    if tof_center_xy is not None:
        cx, cy = tof_center_xy
        dx, dy = _tof_pixel_to_disp_xy(float(cx), float(cy))
        _draw_cross(tof_bgr, dx, dy, color=(255, 255, 255), r=10, t=2)

    return tof_bgr


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    envs = _list_env_dirs(Path(DATA_DIR))
    if not envs:
        raise FileNotFoundError(f"no scenes found under: {DATA_DIR}")

    cv2.namedWindow("LIDAR", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_REFLECT", cv2.WINDOW_AUTOSIZE)

    mouse_state: dict = {"x": 0, "y": 0}

    def _on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if not bool(MOUSE_HOVER_ENABLED):
            return
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse_state["x"] = int(x)
        mouse_state["y"] = int(y)

    cv2.setMouseCallback("LIDAR", _on_mouse)

    idx = 0
    while True:
        env = envs[idx]
        npz = _find_points_npz(env)
        pts = _load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)

        tof_det = detect_ball_tof_2d(env / "tof.raw", window_size=5)
        tof_view = _build_tof_reflect_view(env, tof_det.centroid_xy)

        lidar_det: LidarBallDetection = detect_ball_lidar(pts, seed=0)
        lidar_view, meta = _build_lidar_view(lidar_det)

        # 场景索引
        cv2.putText(lidar_view, f"{env.name}  ({idx+1}/{len(envs)})", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        # Do not overlay any text on TOF_REFLECT to avoid blocking the image.

        # 鼠标悬停距离, 只针对 LiDAR.
        if bool(MOUSE_HOVER_ENABLED):
            mx = int(np.clip(mouse_state.get("x", 0), 0, LIDAR_IMG_W - 1))
            my = int(np.clip(mouse_state.get("y", 0), 0, LIDAR_IMG_H - 1))
            rr = float(meta["lidar_range_map"][my, mx])
            txt = f"r={rr:.3f}m" if np.isfinite(rr) else "r=--"
            cv2.putText(lidar_view, txt, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("LIDAR", lidar_view)
        cv2.imshow("TOF_REFLECT", tof_view)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:
            break
        if k == ord("4"):
            idx = (idx - 1) % len(envs)
        if k == ord("6"):
            idx = (idx + 1) % len(envs)
        if k == ord("0"):
            # 删除当前场景目录, 弹窗确认.
            if _confirm_delete_scene(env):
                try:
                    shutil.rmtree(env, ignore_errors=False)
                except Exception:
                    # If deletion fails, keep current list.
                    pass
                # Refresh scene list after deletion.
                envs = _list_env_dirs(Path(DATA_DIR))
                if not envs:
                    break
                idx = min(idx, len(envs) - 1)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


