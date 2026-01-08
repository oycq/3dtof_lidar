from __future__ import annotations

from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
LAS_FILE = _HERE / "output.las"
IMG_W = 800
IMG_H = 800

# 为了避免超大点云导致显示卡顿，可以做一个上限抽样（None 表示不抽样）
MAX_POINTS = 3_000_000

# x-z 平面视场角（单位：度）。总角 70°，半角 35°。
FOV_XZ_DEG = 70.0
HALF_FOV_XZ_DEG = FOV_XZ_DEG / 2.0


def main() -> None:
    if not LAS_FILE.exists():
        raise FileNotFoundError(f"找不到点云文件：{LAS_FILE}\n请先运行：py .\\run.py")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 matplotlib，请先执行：py -m pip install matplotlib") from e

    try:
        import laspy  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 laspy，请先执行：py -m pip install laspy") from e

    las = laspy.read(str(LAS_FILE))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)

    if MAX_POINTS is not None and x.size > MAX_POINTS:
        idx = np.random.choice(x.size, size=MAX_POINTS, replace=False)
        x, y, z = x[idx], y[idx], z[idx]

    # 视场角过滤：x 作为深度/前方距离，x-z 平面俯仰角限制在 ±35°
    # 角度定义：theta = atan2(z, x)，x>0 才有意义（前方）
    theta = np.arctan2(z, x)
    mask = (x > 0) & (np.abs(theta) <= np.deg2rad(HALF_FOV_XZ_DEG))
    x, y, z = x[mask], y[mask], z[mask]
    if x.size == 0:
        raise ValueError(f"视场角过滤后无点：|atan2(z,x)| <= {HALF_FOV_XZ_DEG}° 且 x>0")

    xmin, xmax = float(np.min(x)), float(np.max(x))  # depth (distance)
    ymin, ymax = float(np.min(y)), float(np.max(y))  # left (+)
    zmin, zmax = float(np.min(z)), float(np.max(z))  # up (+)

    # 防止除 0
    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    if yr == 0 or zr == 0:
        raise ValueError(f"点云 YZ 范围异常：y[{ymin},{ymax}] z[{zmin},{zmax}]")

    # YZ -> 像素坐标
    # - y 轴朝左：y 越大，越靠左 -> col 需要反向映射
    # - z 轴朝上：z 越大，越靠上 -> row 需要反向映射（图像 row 向下为正）
    col = ((ymax - y) / yr * (IMG_W - 1)).astype(np.int32)
    row = ((zmax - z) / zr * (IMG_H - 1)).astype(np.int32)

    col = np.clip(col, 0, IMG_W - 1)
    row = np.clip(row, 0, IMG_H - 1)

    # X(depth) -> 灰度亮度（0~255），x 正方向越远越亮
    if xr == 0:
        bright = np.full(x.shape, 255, dtype=np.uint8)
    else:
        bright = ((x - xmin) / xr * 255.0).astype(np.uint8)

    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    # 同一像素多个点：取更亮（更大 Z）
    np.maximum.at(img, (row, col), bright)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(
        f"{LAS_FILE.name}  |  X(depth)->brightness  |  "
        f"FOV(x-z)=±{HALF_FOV_XZ_DEG:.1f}°  |  "
        f"x:[{xmin:.3f},{xmax:.3f}] y:[{ymin:.3f},{ymax:.3f}] z:[{zmin:.3f},{zmax:.3f}]  points:{x.size}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


