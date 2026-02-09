#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cali/calibrate_tof_lidar.py

用 LiDAR 3D 球心点 和 ToF 2D 像素点做标定.
目标:
- 标定 ToF 内参: fx, fy, cx, cy
- 标定 LiDAR->ToF 外参: R, t

注意:
- 会估计基础畸变 distCoeffs: [k1,k2,p1,p2,k3]
- 3D 点坐标系取 LiDAR 坐标系, 视为 objectPoints
- 2D 点坐标系取 ToF 原始像素坐标系, 视为 imagePoints

输出:
- rms (OpenCV calibrateCamera 返回的总 RMS)
- cali/calib_result.json
- cali/reproj_error.png (重投影误差矢量放大图)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

CAMERA_MATRIX_INIT = np.array(
    [
        [36.0, 0.0, 20.0],
        [0.0, 36.0, 15.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# Allow running from cali/ directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from lidar_ball_detector import detect_ball_lidar  # noqa: E402
from tof_ball_detector import detect_ball_tof_2d  # noqa: E402


@dataclass(frozen=True)
class CalibResult:
    rms: float
    camera_matrix: list[list[float]]
    # OpenCV distCoeffs: [k1, k2, p1, p2, k3]
    dist_coeffs: list[float]
    rvec: list[float]
    tvec: list[float]
    n_points: int


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


def _tof_xy_to_disp_xy(px: float, py: float, *, tof_w: int, tof_h: int, show_w: int, show_h: int) -> tuple[int, int]:
    # 与 cali/check.py 对齐: 显示做 flipV(上下翻转), 再 resize.
    px_i = float(np.clip(px, 0.0, float(tof_w - 1)))
    py_i = float(np.clip(py, 0.0, float(tof_h - 1)))
    py_disp = float((tof_h - 1) - py_i)
    dx = int((px_i + 0.5) * float(show_w) / float(tof_w))
    dy = int((py_disp + 0.5) * float(show_h) / float(tof_h))
    dx = int(np.clip(dx, 0, show_w - 1))
    dy = int(np.clip(dy, 0, show_h - 1))
    return dx, dy


def _draw_cross(img_bgr: np.ndarray, u: int, v: int, *, color: tuple[int, int, int], r: int = 6, t: int = 2) -> None:
    import cv2  # type: ignore

    cv2.line(img_bgr, (u - r, v), (u + r, v), color, t, cv2.LINE_AA)
    cv2.line(img_bgr, (u, v - r), (u, v + r), color, t, cv2.LINE_AA)


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    data_dir = HERE / "data"
    envs = _list_env_dirs(data_dir)
    if not envs:
        raise FileNotFoundError(f"no scenes found under: {data_dir}")

    # ToF image size is fixed by tof3d format.
    tof_w, tof_h = 40, 30

    obj_pts: List[tuple[float, float, float]] = []
    img_pts: List[tuple[float, float]] = []
    # used scenes list is not stored in json by request, but we keep it for logging if needed
    used: List[str] = []

    for env in envs:
        npz = _find_points_npz(env)
        if npz is None or (not npz.exists()):
            continue
        pts = _load_points(npz)

        lidar_det = detect_ball_lidar(pts, seed=0)
        if lidar_det.center_xyz_m is None:
            continue

        tof_det = detect_ball_tof_2d(env / "tof.raw", window_size=5)
        if tof_det.centroid_xy is None:
            continue

        X, Y, Z = lidar_det.center_xyz_m
        u, v = tof_det.centroid_xy
        obj_pts.append((float(X), float(Y), float(Z)))
        img_pts.append((float(u), float(v)))
        used.append(env.name)

    if len(obj_pts) < 10:
        raise RuntimeError(f"not enough correspondences, got {len(obj_pts)}")

    # OpenCV expects list of views. Here we treat all points as one view.
    obj = np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3)
    img = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    # Init camera matrix guess (hard-coded).
    camera_matrix = CAMERA_MATRIX_INIT.copy()
    # Distortion coeffs: [k1, k2, p1, p2, k3]
    dist = np.zeros((5, 1), dtype=np.float64)

    flags = 0
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # 开启基础畸变参数估计: k1,k2,p1,p2,k3

    rms, cam_mtx, dist_out, rvecs, tvecs = cv2.calibrateCamera(
        [obj],
        [img],
        (int(tof_w), int(tof_h)),
        camera_matrix,
        dist,
        flags=flags,
    )
    rvec = np.array(rvecs[0], dtype=np.float64).reshape(3)
    tvec = np.array(tvecs[0], dtype=np.float64).reshape(3)

    # Reprojection and error stats.
    proj, _ = cv2.projectPoints(obj.reshape(-1, 3), rvec, tvec, cam_mtx, dist_out)
    proj = proj.reshape(-1, 2).astype(np.float64, copy=False)
    img2 = img.reshape(-1, 2).astype(np.float64, copy=False)
    err = proj - img2
    err_norm = np.sqrt(np.sum(err * err, axis=1))
    mean_err = float(np.mean(err_norm))
    max_err = float(np.max(err_norm))

    # Output directory.
    out_dir = HERE / "cali_result"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save json (do not store used_scenes by request).
    res = CalibResult(
        rms=float(rms),
        camera_matrix=cam_mtx.astype(np.float64).tolist(),
        dist_coeffs=dist_out.reshape(-1).astype(np.float64).tolist(),
        rvec=rvec.tolist(),
        tvec=tvec.tolist(),
        n_points=int(len(obj_pts)),
    )
    out_json = out_dir / "calib_result.json"
    out_json.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")

    # Error visualization image.
    show_w, show_h = 700, 700
    bg = np.zeros((show_h, show_w, 3), dtype=np.uint8)
    bg[:] = (30, 30, 30)

    for (u0, v0), (up, vp) in zip(img_pts, proj.tolist()):
        mx, my = _tof_xy_to_disp_xy(float(u0), float(v0), tof_w=tof_w, tof_h=tof_h, show_w=show_w, show_h=show_h)
        px, py = _tof_xy_to_disp_xy(float(up), float(vp), tof_w=tof_w, tof_h=tof_h, show_w=show_w, show_h=show_h)

        _draw_cross(bg, mx, my, color=(255, 255, 255), r=5, t=1)  # measured
        _draw_cross(bg, px, py, color=(0, 0, 255), r=5, t=1)  # projected

        # Draw a direct line between measured and projected points (no scaling, no arrows).
        cv2.line(bg, (mx, my), (px, py), (0, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(bg, f"rms={float(rms):.4f}px  mean={mean_err:.3f}px  max={max_err:.3f}px  n={len(obj_pts)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bg, "white: measured, red: projected, yellow: reproj error", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    out_img = out_dir / "reproj_error.png"
    ok = cv2.imwrite(str(out_img), bg)
    if not ok:
        raise RuntimeError(f"failed to write: {out_img}")

    # Print extrinsics in mm for convenience.
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    t_mm = (tvec.reshape(3) * 1000.0).tolist()
    print(f"rms={float(rms):.6f} px")
    print(f"mean_err={mean_err:.6f} px, max_err={max_err:.6f} px, n={len(obj_pts)}")
    print("extrinsics (LiDAR->ToF):")
    print("R =")
    for row in R.tolist():
        print("  " + "  ".join([f"{v:+.9f}" for v in row]))
    print(f"t_mm = [{t_mm[0]:+.3f}, {t_mm[1]:+.3f}, {t_mm[2]:+.3f}]")
    print(f"saved: {out_json}")
    print(f"saved: {out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


