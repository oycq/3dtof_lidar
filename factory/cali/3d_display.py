#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    data_path = Path("data.npy")
    if not data_path.exists():
        raise FileNotFoundError(f"未找到文件: {data_path.resolve()}")

    d = np.load(str(data_path))
    d = np.asarray(d, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"data.npy 期望是 2D 矩阵，实际 shape={d.shape}")

    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]

    z = d.reshape(-1)
    x = xx.reshape(-1).astype(np.float32)
    y = yy.reshape(-1).astype(np.float32)

    # 仅显示有效距离点（>0）
    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]

    fig = plt.figure("3D Display")
    ax = fig.add_subplot(111, projection="3d")

    if z.size > 0:
        sc = ax.scatter(x, y, z, c=z, s=18, cmap="turbo")
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label="Distance (m)")
    else:
        ax.text2D(0.05, 0.95, "No valid points (z<=0).", transform=ax.transAxes)

    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_zlabel("Distance (m)")
    ax.set_title(f"data.npy  shape={h}x{w}")

    # Matplotlib 默认可鼠标交互：旋转/平移/缩放
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


