#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_data.py

可视化 client.py 保存到 data/ 目录下的“环境数据”：
- 激光雷达：读取 points_last*.npz 中的 x/y/z，使用固定 FOV 的 2D 投影渲染
- TOF：读取 tof.raw，参考 get_tof.py 的处理逻辑计算 30*40 深度图，并拉伸到 300*400 显示

按键：
- 4：上一个环境
- 6：下一个环境
- ESC：退出
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from tof3d import tof_distance_matrix
import cv2

# ========= 配置 =========
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = np.deg2rad(FOV_DEG / 2.0)
MAX_RANGE_M = 20.0
NEAR_SAT_M = 1.0

AUTO_EXPOSURE = True
AE_LOW_PCT = 2.0
AE_HIGH_PCT = 98.0
AE_GAMMA = 0.8
AE_MEAN_SECONDS = 2.0

TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 400
TOF_SHOW_H = 300


@dataclass
class _AEState:
    ts: list[float]
    lo: list[float]
    hi: list[float]


def _list_env_dirs() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    ds = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    ds.sort(key=lambda p: p.name)
    return ds


def _find_points_npz(env_dir: Path) -> Optional[Path]:
    # 兼容：points_last{N}s.npz 或其它以 points_last 开头的 npz
    cands = sorted(env_dir.glob("points_last*.npz"))
    return cands[0] if cands else None


def _load_points(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    return x, y, z


def _render_lidar_gray(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    # 反比例亮度（和 client.py 一致）
    depth_m = np.clip(x, NEAR_SAT_M, MAX_RANGE_M)
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


def _auto_expose_u8(img: np.ndarray, st: _AEState, now_ts: float) -> np.ndarray:
    if (not AUTO_EXPOSURE) or img.size == 0:
        return img
    nz = img[img > 0]
    if nz.size < 500:
        return img

    lo_now = float(np.percentile(nz, AE_LOW_PCT))
    hi_now = float(np.percentile(nz, AE_HIGH_PCT))
    if hi_now <= lo_now + 1.0:
        return img

    st.ts.append(now_ts)
    st.lo.append(lo_now)
    st.hi.append(hi_now)

    cutoff = now_ts - float(max(AE_MEAN_SECONDS, 0.05))
    while st.ts and st.ts[0] < cutoff:
        st.ts.pop(0)
        st.lo.pop(0)
        st.hi.pop(0)

    if not st.ts:
        return img
    lo = float(np.mean(st.lo))
    hi = float(np.mean(st.hi))
    if hi <= lo + 1.0:
        return img

    out = (img.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if AE_GAMMA != 1.0:
        out = np.power(out, AE_GAMMA)
    out = (out * 255.0).astype(np.uint8)
    out[img == 0] = 0
    return out


def _depth_to_u8(depth_m: np.ndarray) -> np.ndarray:
    """
    把 TOF 深度图转为 u8：
    - 量程/映射规则与激光雷达一致：NEAR_SAT_M~MAX_RANGE_M，I≈255/x
    - 0 表示无效（不参与显示）
    """
    if depth_m.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    dm = depth_m.astype(np.float32, copy=False)
    out = np.zeros(dm.shape, dtype=np.uint8)
    m = dm > 0
    if not np.any(m):
        return out
    dm2 = np.clip(dm[m], NEAR_SAT_M, MAX_RANGE_M)
    out[m] = np.clip(np.rint(255.0 / dm2), 0.0, 255.0).astype(np.uint8)
    return out


def main() -> int:

    envs = _list_env_dirs()
    if not envs:
        raise FileNotFoundError(f"data 目录下没有环境数据：{DATA_DIR}")

    idx = 0
    ae_state = _AEState(ts=[], lo=[], hi=[])

    cv2.namedWindow("LiDAR", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF", cv2.WINDOW_AUTOSIZE)

    while True:
        env = envs[idx]

        # ---- LiDAR ----
        npz_path = _find_points_npz(env)
        lidar_view = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)
        lidar_pts = 0
        if npz_path is not None and npz_path.exists():
            x, y, z = _load_points(npz_path)
            lidar_pts = int(x.size)
            g = _auto_expose_u8(_render_lidar_gray(x, y, z), ae_state, now_ts=float(__import__("time").time()))
            lidar_view = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)

        # ---- TOF ----
        tof_path = env / "tof.raw"
        tof_view = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
        if tof_path.exists():
            depth = tof_distance_matrix(tof_path)
            u8 = _depth_to_u8(depth)
            u8_big = cv2.resize(u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
            # “旋转 180° + 左右镜像”等价于“只做上下镜像”
            u8_big = cv2.flip(u8_big, 0)
            tof_view = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)

        # overlay
        cv2.putText(
            lidar_view,
            f"{env.name}  ({idx+1}/{len(envs)})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LiDAR", lidar_view)
        cv2.imshow("TOF", tof_view)

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


