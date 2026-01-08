from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


_HERE = Path(__file__).resolve().parent
LAS_FILE = _HERE / "output.las"

# 为了避免超大点云导致显示卡顿，可以做一个上限抽样（None 表示不抽样）
MAX_POINTS = 2_000_000

# 初始视角（相机 FOV，单位：度）
INIT_FOV_DEG = 70.0


def visualize_las_fixed(
    *,
    voxel_size: Optional[float] = None,
    color_by_intensity: bool = True,
) -> None:
    """
    3d 可视化（固定读取仓库根目录 output.las）

    坐标系/视角约定：
      - x 为深度（纵深，朝里）
      - y 轴朝左
      - z 轴朝上
    """
    if not LAS_FILE.exists():
        raise FileNotFoundError(f"找不到点云文件：{LAS_FILE}\n请先运行：py .\\run.py")

    try:
        import laspy  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 laspy，请先执行：py -m pip install laspy") from e

    try:
        import open3d as o3d  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 open3d，请先执行：py -m pip install open3d") from e

    las = laspy.read(str(LAS_FILE))
    pts = np.column_stack([las.x, las.y, las.z]).astype(np.float64, copy=False)
    if pts.size == 0:
        raise ValueError(f"LAS 文件没有点：{LAS_FILE}")

    idx = None
    if MAX_POINTS is not None and pts.shape[0] > MAX_POINTS:
        idx = np.random.choice(pts.shape[0], size=MAX_POINTS, replace=False)
        pts = pts[idx]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)

    if voxel_size is not None and voxel_size > 0:
        pc = pc.voxel_down_sample(voxel_size=float(voxel_size))

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

    # 使用 Visualizer 以便设置相机视角
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"OpenPyLivox - {LAS_FILE.name}", width=1280, height=800)
    vis.add_geometry(pc)
    vis.update_geometry(pc)
    vis.poll_events()
    vis.update_renderer()

    # 默认视角：从 +x 方向往回看（front=-x），z 朝上
    ctr = vis.get_view_control()
    bbox = pc.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([-1.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.7)
    # 设置初始 FOV=70°（不同 open3d 版本 API 略有差异，这里做兼容处理）
    try:
        if hasattr(ctr, "set_field_of_view"):
            ctr.set_field_of_view(float(INIT_FOV_DEG))  # type: ignore[attr-defined]
        elif hasattr(ctr, "get_field_of_view") and hasattr(ctr, "change_field_of_view"):
            cur = float(ctr.get_field_of_view())  # type: ignore[attr-defined]
            ctr.change_field_of_view(float(INIT_FOV_DEG - cur))  # type: ignore[attr-defined]
    except Exception:
        # 不影响运行：至少保留 front/up/zoom
        pass

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    visualize_las_fixed()

 