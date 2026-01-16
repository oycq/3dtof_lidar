#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_combine.py

把 cali/data 下的所有场景输出一个拼接效果图, 保存为 combine.png.

布局:
- 固定 8 列.
- 每行 4 个场景, 左 4 列是 LiDAR 图, 右 4 列是 ToF 反射率图.
- 例如 20 个场景时, 共 5 行.

说明:
- LiDAR/ToF 的渲染逻辑直接复用 cali/check.py, 保证和 check.py 最终显示一致.
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import numpy as np

# Allow running from cali/ directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    # Import after cv2 check, and reuse helpers from check.py.
    import check as chk  # type: ignore

    data_dir = Path(chk.DATA_DIR)
    envs = chk._list_env_dirs(data_dir)
    if not envs:
        raise FileNotFoundError(f"no scenes found under: {data_dir}")

    # Grid config.
    scenes_per_row = 4
    cols = 8  # 4 lidar + 4 tof
    rows = int(ceil(len(envs) / scenes_per_row))

    w = int(chk.LIDAR_IMG_W)
    h = int(chk.LIDAR_IMG_H)
    out = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, env in enumerate(envs):
        row = int(i // scenes_per_row)
        col = int(i % scenes_per_row)

        # Load points.
        npz = chk._find_points_npz(env)
        pts = chk._load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)

        # Detect.
        tof_det = chk.detect_ball_tof_2d(env / "tof.raw", window_size=5, min_peak=float(chk.TOF_MIN_PEAK), valid_bins=int(chk.TOF_VALID_BINS))
        lidar_det = chk.detect_ball_lidar(pts, seed=0)

        # Render.
        lidar_img, _meta = chk._build_lidar_view(lidar_det)
        tof_img = chk._build_tof_reflect_view(env, tof_det.centroid_xy)

        # Align ToF size to LiDAR for grid layout.
        if tof_img.shape[0] != h or tof_img.shape[1] != w:
            tof_img = cv2.resize(tof_img, (w, h), interpolation=cv2.INTER_NEAREST)

        # Place in grid.
        y0 = row * h
        x_lidar = col * w
        x_tof = (col + scenes_per_row) * w
        out[y0 : y0 + h, x_lidar : x_lidar + w] = lidar_img
        out[y0 : y0 + h, x_tof : x_tof + w] = tof_img

    out_path = HERE / "combine.png"
    ok = cv2.imwrite(str(out_path), out)
    if not ok:
        raise RuntimeError(f"failed to write: {out_path}")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


