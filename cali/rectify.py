#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- 读取 cali/cali_result/calib_result.json 的标定结果(camera_matrix, rvec, tvec)
- 遍历 DATA_DIR 下每个场景文件夹, 读取:
  - points_last*.npz: LiDAR 点云(x,y,z)
  - tof.raw: ToF 直方图数据
- ToF 反射率图: 对每个像素直方图做 sum 得到强度, 再做归一化+gamma, 最后 flipV+resize 到 400x300
- LiDAR 投影图: 用 numpy 矩阵运算把 LiDAR 3D 点投到 ToF 40x30 像素坐标系
  - 过滤: 只保留落在 0<=u<40, 0<=v<30 且位于相机"前方"的点(前方 Z 符号自动选择)
- 聚合: 同一 ToF 像素内多个点, 取"最近"(欧式距离最小)的前 5%-10% 点再求均值
  - 显示: 把平均距离映射成强度 u8=round(255/d), 再用 COLORMAP_TURBO 着色, 无点为黑
"""

import json
import sys
import shutil
import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 允许在 cali 目录直接运行
# ROOT 是项目根目录，用于调整 sys.path 以便导入自定义模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入自定义模块，用于处理 ToF 数据
from tof3d import ToF3DParams, tof_histograms  # noqa: E402

# HERE 是当前脚本所在目录
HERE = Path(__file__).resolve().parent
# DATA_DIR 是数据目录，存放场景文件夹
DATA_DIR = HERE / "../data"
# CALIB_JSON 是标定结果文件路径
CALIB_JSON = HERE / "cali_result" / "calib_result.json"

# 常量集中管理，便于修改
# LiDAR 显示参数
LIDAR_IMG_W = 700  # LiDAR 投影图像宽度
LIDAR_IMG_H = 700  # LiDAR 投影图像高度
FOV_DEG = 70.0  # 视场角（度）
HALF_FOV_RAD = np.deg2rad(FOV_DEG / 2.0)  # 半视场角（弧度）

# ToF 显示参数
TOF_W = 40  # ToF 原始宽度
TOF_H = 30  # ToF 原始高度
TOF_SHOW_W = 400  # ToF 显示宽度
TOF_SHOW_H = 300  # ToF 显示高度
TOF_MIN_PEAK = 100  # ToF 最小峰值计数，用于过滤
TOF_INTEN_GAMMA = 2.2  # ToF 强度 gamma 校正值
TOF_INTEN_TARGET_MEAN = 0.18  # ToF 强度目标均值，用于归一化
TOF_TOP_FRAC = 0.2  # 聚合分位比例(0~1)，例如 0.2 表示取第 20% 分位处的距离值

# 其他常量
TITLE_H = 26  # 标题栏高度
EPSILON = 1e-6  # 小 epsilon 值，用于避免除零
DIST_EPS2 = 1e-12  # 距离平方 epsilon，用于过滤零点


def tof_disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """
    把“显示窗口中的坐标”(已 resize 后的像素)映射回 ToF 40x30 像素坐标。
    注意：rectify.py 的 ToF 显示同样做了 flipV(上下翻转)。

    参数:
      dx, dy: 显示坐标中的 x, y
      show_w, show_h: 显示图像的宽度和高度

    返回:
      (px, py): 原始 ToF 像素坐标（未翻转的 py）
    """
    sw = max(show_w, 1)
    sh = max(show_h, 1)
    px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
    py_disp = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    py = (TOF_H - 1) - py_disp  # 还原 flipV
    return px, py


def list_env_dirs(root: Path) -> List[Path]:
    """列出所有场景目录, 按目录名排序"""
    if not root.exists():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name)
    return dirs


def find_points_npz(env_dir: Path) -> Optional[Path]:
    """
    找到 points_last*.npz 文件（默认取第一个匹配的）
    如果没有找到，返回 None
    """
    cands = sorted(env_dir.glob("points_last*.npz"))
    return cands[0] if cands else None


def load_points(npz_path: Path) -> np.ndarray:
    """
    读取 npz 中的 x/y/z, 返回 Nx3 点云(单位米)。
    删除距离=0的坏点和非有限值（nan/inf）
    """
    d = np.load(npz_path)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.column_stack([x, y, z])

    finite = np.isfinite(pts).all(axis=1)
    dist2 = np.sum(pts * pts, axis=1)
    good = finite & (dist2 > DIST_EPS2)
    return pts[good]


def tof_intensity_to_u8(intensity_sum: np.ndarray) -> np.ndarray:
    """
    ToF 反射率显示映射: 用整图 mean 做归一化到 target_mean,
    然后做 gamma 显示(1/gamma), 最终转换为 uint8
    """
    if intensity_sum.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    v = intensity_sum.astype(np.float32)
    mean = np.mean(v) if v.size else 0.0
    if mean <= 0.0:
        return np.zeros_like(v, dtype=np.uint8)

    k = max(mean / TOF_INTEN_TARGET_MEAN, EPSILON)
    n = np.clip(v / k, 0.0, 1.0)
    if TOF_INTEN_GAMMA > 0:
        n = np.power(n, 1.0 / TOF_INTEN_GAMMA)
    return np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)


def render_lidar_gray(points_xyz: np.ndarray) -> np.ndarray:
    """
    LiDAR 2D 投影渲染(仅用于显示):
    - 只取 x>0 且 yaw/pitch 在 FOV 内的点
    - 用欧式距离作为"距离", 映射成 u8=round(255/d)
    - 使用 maximum.at 来处理重叠点，取最大强度
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    x, y, z = points_xyz.T
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))
    mask = (x > 0) & (np.abs(yaw) <= HALF_FOV_RAD) & (np.abs(pitch) <= HALF_FOV_RAD)
    if not np.any(mask):
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    x_m, y_m, z_m = x[mask], y[mask], z[mask]
    depth_m = np.sqrt(x_m**2 + y_m**2 + z_m**2)
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0, 255).astype(np.uint8)

    col = ((HALF_FOV_RAD - yaw[mask]) / (2 * HALF_FOV_RAD) * (LIDAR_IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV_RAD - pitch[mask]) / (2 * HALF_FOV_RAD) * (LIDAR_IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_IMG_W - 1)
    row = np.clip(row, 0, LIDAR_IMG_H - 1)

    img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
    np.maximum.at(img, (row, col), depth_u8)
    return img


def load_calib(calib_json: Path) -> Dict[str, np.ndarray]:
    """
    读取 calib_result.json 中的标定参数
    预计算 R (旋转矩阵), t (平移向量), K (相机内参) 为 float32 以加速后续计算
    """
    with calib_json.open("r", encoding="utf-8") as f:
        d = json.load(f)
    calib = {
        "camera_matrix": np.asarray(d["camera_matrix"], dtype=np.float64).reshape(3, 3),
        "rvec": np.asarray(d["rvec"], dtype=np.float64).reshape(3),
        "tvec": np.asarray(d["tvec"], dtype=np.float64).reshape(3),
    }
    # 预计算
    calib["R"] = cv2.Rodrigues(calib["rvec"].reshape(3, 1))[0].astype(np.float32)
    calib["t"] = calib["tvec"].reshape(1, 3).astype(np.float32)
    calib["K"] = calib["camera_matrix"].astype(np.float32)
    return calib


def build_tof_reflect_view(env_dir: Path) -> np.ndarray:
    """
    生成 ToF 反射率图(400x300, flipV, BGR)
    - 读取 tof.raw 文件
    - 计算每个像素的强度总和
    - 转换为 uint8 并 resize 和 flip
    """
    tof_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_path = env_dir / "tof.raw"
    if not tof_path.exists():
        return tof_bgr

    params = ToF3DParams(min_peak_count=TOF_MIN_PEAK)
    hists = tof_histograms(tof_path, params=params).astype(np.float32)  # (30,40,64)
    if hists.size == 0:
        return tof_bgr

    inten = hists.sum(axis=2)
    inten_u8 = tof_intensity_to_u8(inten)
    inten_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
    inten_big = cv2.flip(inten_big, 0)
    return cv2.cvtColor(inten_big, cv2.COLOR_GRAY2BGR)


def project_lidar_to_tof(pts_lidar_xyz: np.ndarray, calib: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 LiDAR 3D 点投影到 ToF 像素坐标(u,v), 同时返回相机坐标系 Z 以判断前后
    使用矩阵运算加速投影
    """
    if pts_lidar_xyz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    R, t, K = calib["R"], calib["t"], calib["K"]
    X = pts_lidar_xyz.astype(np.float32)  # (N,3)
    cam = (R @ X.T).T + t  # (N,3)
    zc = cam[:, 2]

    z = zc.copy()
    z[z == 0] = EPSILON

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    uv = np.column_stack([u, v])
    return uv, zc


def aggregate_lidar_to_tof_pixels(uv: np.ndarray, zc: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    聚合: 同一 ToF 像素内多个点, 取按欧式距离排序后第 TOF_TOP_FRAC 分位处的那个距离值
    - 自动选择前方 Z 符号
    - 使用 argsort 和 partition 高效处理分组和选择最近点
    返回 u8map (30,40) 和 dmap (30,40)
    """
    n_all = uv.shape[0]
    if n_all == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8), np.zeros((TOF_H, TOF_W), dtype=np.float32)

    n_z_pos = np.count_nonzero(zc > EPSILON)
    n_z_neg = np.count_nonzero(zc < -EPSILON)
    front = (zc > EPSILON) if n_z_pos >= n_z_neg else (zc < -EPSILON)
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < TOF_W) & (uv[:, 1] >= 0) & (uv[:, 1] < TOF_H)
    mask = front & in_img
    if not np.any(mask):
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8), np.zeros((TOF_H, TOF_W), dtype=np.float32)

    uu, vv = uv[mask, 0], uv[mask, 1]
    pts_m = pts[mask]
    dist_lidar = np.sqrt(np.sum(pts_m**2, axis=1))

    ui = np.clip(np.floor(uu).astype(np.int32), 0, TOF_W - 1)
    vi = np.clip(np.floor(vv).astype(np.int32), 0, TOF_H - 1)
    pix = vi * TOF_W + ui

    order = np.argsort(pix, kind="stable")
    pix_s = pix[order]
    d_s = dist_lidar[order]

    cuts = np.flatnonzero(pix_s[1:] != pix_s[:-1]) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [pix_s.size]])
    grp_ids = pix_s[starts]

    u8map_flat = np.zeros(TOF_H * TOF_W, dtype=np.uint8)
    dmap_flat = np.zeros(TOF_H * TOF_W, dtype=np.float32)

    for gid, s, e in zip(grp_ids, starts, ends):
        vals = d_s[s:e]
        n = vals.size
        if n == 0:
            continue
        # 取排序后第 ceil(n*TOF_TOP_FRAC) 小的那个值（一个值，不是均值）
        k = max(1, int(np.ceil(n * TOF_TOP_FRAC)))
        k_idx = min(n - 1, k - 1)  # 0-based
        d = float(np.partition(vals, k_idx)[k_idx])
        u8map_flat[gid] = np.clip(np.rint(255.0 / d), 0, 255).astype(np.uint8)
        dmap_flat[gid] = d

    u8map = u8map_flat.reshape((TOF_H, TOF_W))
    dmap = dmap_flat.reshape((TOF_H, TOF_W))
    return u8map, dmap


def resize_keep_aspect_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
    """
    按高度缩放, 保持宽高比(最近邻插值, 避免模糊)
    如果目标高度无效，返回原图
    """
    if img.size == 0 or target_h <= 0:
        return img
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def with_header(img: np.ndarray, header_text: str, header_h: int) -> np.ndarray:
    """
    在图像上方增加黑色标题条(额外区域)
    - 添加文字到标题条
    """
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    hh = max(1, header_h)
    out = np.zeros((h + hh, w, 3), dtype=np.uint8)
    out[hh:, :] = img
    cv2.putText(out, header_text, (10, hh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def confirm_delete_scene(scene_dir: Path) -> bool:
    """
    按下 0 删除场景前的确认弹窗（Windows）。
    用户点击“是/Yes”返回 True，否则返回 False。
    """
    title = "确认删除"
    msg = f"确定删除这个场景文件夹吗？\n\n{scene_dir.name}\n\n删除后不可恢复。"
    try:
        # MessageBoxW: returns IDYES (6) or IDNO (7)
        MB_YESNO = 0x00000004
        MB_ICONWARNING = 0x00000030
        IDYES = 6
        ret = ctypes.windll.user32.MessageBoxW(0, msg, title, MB_YESNO | MB_ICONWARNING)
        return int(ret) == IDYES
    except Exception:
        # 无法弹窗时默认不删除
        return False


def compute_scene(env: Path, calib: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    计算单个场景的渲染图像
    - 加载点云并渲染 LiDAR 图
    - 构建 ToF 反射率图
    - 投影 LiDAR 到 ToF 并聚合
    - 生成投影彩色图
    """
    # 点云
    npz = find_points_npz(env)
    pts = load_points(npz) if npz and npz.exists() else np.zeros((0, 3), dtype=np.float32)
    lidar_u8 = render_lidar_gray(pts)
    lidar_bgr = cv2.applyColorMap(lidar_u8, cv2.COLORMAP_TURBO)

    # ToF 反射率
    tof_reflect = build_tof_reflect_view(env)

    # 投影和聚合
    uv, zc = project_lidar_to_tof(pts, calib)
    u8map, dmap = aggregate_lidar_to_tof_pixels(uv, zc, pts)
    u8_big = cv2.resize(u8map, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
    u8_big = cv2.flip(u8_big, 0)
    proj_full_color = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)
    proj_full_color[u8_big == 0] = (0, 0, 0)

    return {
        "lidar_bgr": lidar_bgr,
        "tof_reflect": tof_reflect,
        "proj_full_color": proj_full_color,
        "proj_dmap_m": dmap,  # (30,40) float32, 单位 m（无点为 0）
    }


def _win_message_box(title: str, msg: str) -> None:
    """Windows 下弹一个提示框；失败时静默忽略。"""
    try:
        ctypes.windll.user32.MessageBoxW(0, str(msg), str(title), 0x00000000)
    except Exception:
        pass


def export_all_scenes_to_train_data(envs: List[Path], calib: Dict[str, np.ndarray], out_dir: Path) -> None:
    """
    导出所有场景到 train_data：
    - input_00001.npy: ToF 原始直方图 (H,W,C) float32
    - output_00001.npy: 聚合后的 LiDAR 距离图 (H,W) float32，单位米，无点为 0

    说明：
    - 场景顺序：按 envs 列表顺序（已按目录名排序）
    - 每个场景都会产出一对文件；若某文件缺失，则写入全 0 占位
    """
    # 先删再建
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预分配“缺失时”的占位数组
    empty_in = np.zeros((TOF_H, TOF_W, 64), dtype=np.float32)
    empty_out = np.zeros((TOF_H, TOF_W), dtype=np.float32)

    total = len(envs)
    for i, env in enumerate(envs, start=1):
        # ToF input
        tof_path = env / "tof.raw"
        if tof_path.exists():
            # tof_histograms 返回 uint16，这里转 float32 作为训练输入
            hists = tof_histograms(tof_path, params=ToF3DParams()).astype(np.float32, copy=False)
            if hists.shape != (TOF_H, TOF_W, 64):
                # 防御：遇到异常形状则写占位
                hists = empty_in
        else:
            hists = empty_in

        # LiDAR output
        npz = find_points_npz(env)
        if npz and npz.exists():
            pts = load_points(npz)
            uv, zc = project_lidar_to_tof(pts, calib)
            _, dmap = aggregate_lidar_to_tof_pixels(uv, zc, pts)
            if dmap.shape != (TOF_H, TOF_W):
                dmap = empty_out
        else:
            dmap = empty_out

        in_path = out_dir / f"input_{i:05d}.npy"
        out_path = out_dir / f"output_{i:05d}.npy"
        np.save(str(in_path), hists.astype(np.float32, copy=False))
        np.save(str(out_path), dmap.astype(np.float32, copy=False))

        # 控制台进度（导出多场景时可见）
        if (i == 1) or (i == total) or (i % 25 == 0):
            print(f"[export train_data] {i}/{total} {env.name}")


def main() -> int:
    """
    主函数:
    - 加载标定
    - 列出场景目录
    - 创建 OpenCV 窗口和鼠标回调
    - 循环显示场景图像，支持切换场景和视图
    """
    # 读取标定
    if not CALIB_JSON.exists():
        raise FileNotFoundError(f"missing calib json: {CALIB_JSON}")
    calib = load_calib(CALIB_JSON)

    # 列出场景
    envs = list_env_dirs(DATA_DIR)
    if not envs:
        raise FileNotFoundError(f"no scenes found under: {DATA_DIR}")

    cv2.namedWindow("VIEW", cv2.WINDOW_AUTOSIZE)

    # 鼠标悬停：用于右下角投影图显示“距离”
    mouse_state = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event != cv2.EVENT_MOUSEMOVE:
            return
        mouse_state["x"] = x
        mouse_state["y"] = y

    cv2.setMouseCallback("VIEW", on_mouse)

    scene_cache: Dict[str, Dict[str, np.ndarray]] = {}
    idx = 0
    bottom_show_proj = True

    while True:
        env = envs[idx]
        key = str(env)
        if key not in scene_cache:
            scene_cache[key] = compute_scene(env, calib)

        cached = scene_cache[key]
        lidar_bgr = cached["lidar_bgr"]
        tof_reflect = cached["tof_reflect"]
        proj_full_color = cached["proj_full_color"]
        proj_dmap_m = cached["proj_dmap_m"]

        # 只在点云图上叠加文字, 避免遮挡右侧分析用图像
        lidar_show = lidar_bgr.copy()
        cv2.putText(
            lidar_show,
            f"{env.name}  ({idx + 1}/{len(envs)})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 拼接: 左侧整高; 右侧上下两张(带标题条); 总高度与左侧一致
        lidar_h = lidar_show.shape[0]
        header_h = TITLE_H
        content_total = max(2, lidar_h - 2 * header_h)
        top_h = content_total // 2
        bot_h = content_total - top_h

        right_top = resize_keep_aspect_to_h(tof_reflect, top_h)
        right_bottom_src = proj_full_color if bottom_show_proj else tof_reflect
        right_bottom = resize_keep_aspect_to_h(right_bottom_src, bot_h)

        # 鼠标悬停距离：仅对右下角“投影图”生效
        hover_txt = ""
        if bottom_show_proj and proj_dmap_m.size:
            view_x0 = lidar_show.shape[1]
            right_top_h_total = right_top.shape[0] + header_h

            mx, my = mouse_state["x"], mouse_state["y"]
            if view_x0 <= mx < view_x0 + right_bottom.shape[1]:
                yy0 = right_top_h_total + header_h
                yy1 = yy0 + right_bottom.shape[0]
                if yy0 <= my < yy1:
                    dx = mx - view_x0
                    dy = my - yy0
                    px, py = tof_disp_xy_to_pixel(dx, dy, right_bottom.shape[1], right_bottom.shape[0])
                    d = proj_dmap_m[py, px]
                    hover_txt = f" | hover=({px},{py}) {d:.3f}m" if d > 0 else f" | hover=({px},{py}) --"

        right_top = with_header(right_top, "TOF REFLECT", header_h)
        right_bottom_title = ("LIDAR PROJ" if bottom_show_proj else "TOF REFLECT") + hover_txt
        right_bottom = with_header(right_bottom, right_bottom_title, header_h)

        # 右侧上下两张补齐到同一宽度
        rw = max(right_top.shape[1], right_bottom.shape[1])
        if right_top.shape[1] < rw:
            pad = rw - right_top.shape[1]
            right_top = cv2.copyMakeBorder(right_top, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if right_bottom.shape[1] < rw:
            pad = rw - right_bottom.shape[1]
            right_bottom = cv2.copyMakeBorder(right_bottom, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        right_col = np.vstack([right_top, right_bottom])
        if right_col.shape[0] > lidar_h:
            right_col = right_col[:lidar_h]

        view = np.hstack([lidar_show, right_col])
        cv2.imshow("VIEW", view)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord("0"):  # Delete current scene
            if confirm_delete_scene(env):
                try:
                    shutil.rmtree(env)
                except Exception as e:
                    try:
                        ctypes.windll.user32.MessageBoxW(0, str(e), "删除失败", 0x00000000 | 0x00000010)
                    except Exception:
                        pass
                # 从列表/缓存移除，并跳到下一个
                scene_cache.pop(key, None)
                del envs[idx]
                if not envs:
                    break
                idx = idx % len(envs)
        elif k == ord("4"):  # Left
            idx = (idx - 1) % len(envs)
        elif k == ord("6"):  # Right
            idx = (idx + 1) % len(envs)
        elif k == ord(" "):  # Space
            bottom_show_proj = not bottom_show_proj
        elif k == ord("9"):  # Export all scenes to train_data
            out_dir = ROOT / "train_data"
            try:
                export_all_scenes_to_train_data(envs, calib, out_dir)
                _win_message_box("导出完成", f"已导出 {len(envs)} 组数据到:\n{out_dir}")
            except Exception as e:
                _win_message_box("导出失败", str(e))

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())