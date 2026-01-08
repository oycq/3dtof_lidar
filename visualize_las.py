from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def visualize_las(
    las_path: str,
    *,
    voxel_size: Optional[float] = None,
    max_points: Optional[int] = 2_000_000,
    color_by_intensity: bool = True,
) -> None:
    """
    使用 open3d 可视化 LAS 点云（独立脚本，不依赖 openpylivox 包内代码）。

    依赖:
      - laspy (读取 .las)
      - open3d (可视化) -> `py -m pip install open3d`
    """
    p = Path(las_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"LAS 文件不存在: {p}")

    try:
        import laspy  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 laspy，请先执行：py -m pip install laspy") from e

    try:
        import open3d as o3d  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 open3d，请先执行：py -m pip install open3d") from e

    las = laspy.read(str(p))
    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
    if pts.size == 0:
        raise ValueError(f"LAS 文件没有点: {p}")

    idx = None
    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)

    if color_by_intensity and hasattr(las, "intensity"):
        inten = np.asarray(las.intensity)
        if idx is not None:
            inten = inten[idx]

        inten = inten.astype(np.float64, copy=False)
        mn, mx = float(np.min(inten)), float(np.max(inten))
        if mx > mn:
            gray = ((inten - mn) / (mx - mn)).reshape(-1, 1)
        else:
            gray = np.zeros((inten.shape[0], 1), dtype=np.float64)
        pc.colors = o3d.utility.Vector3dVector(np.repeat(gray, 3, axis=1))

    if voxel_size is not None and voxel_size > 0:
        pc = pc.voxel_down_sample(voxel_size=float(voxel_size))

    o3d.visualization.draw_geometries([pc], window_name=f"OpenPyLivox - {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="可视化 LAS 点云（需要 open3d）")
    ap.add_argument("las", help="LAS 文件路径，例如 test.bin.las")
    ap.add_argument("--voxel", type=float, default=None, help="体素降采样尺寸（米），例如 0.02")
    ap.add_argument("--max-points", type=int, default=2_000_000, help="超过该点数则随机抽样")
    ap.add_argument("--no-intensity", action="store_true", help="不按 intensity 着色")
    args = ap.parse_args()

    visualize_las(
        args.las,
        voxel_size=args.voxel,
        max_points=args.max_points,
        color_by_intensity=not args.no_intensity,
    )


if __name__ == "__main__":
    main()



