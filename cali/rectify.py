#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- 读取 cali/cali_result/calib_result.json 的标定结果(camera_matrix, rvec, tvec)
- 遍历 DATA_DIR 下每个场景文件夹, 读取:
  - points_last*.npz: LiDAR 点云(x,y,z)
  - tof.raw: ToF 直方图数据
- ToF 反射率图: 对每个像素直方图做 sum 得到强度, 再做归一化+gamma, 最后 flipV+resize 到 400x300
- LiDAR 投影图: 用 cv2.projectPoints 把 LiDAR 3D 点投到 ToF 40x30 像素坐标系
  - 过滤: 只保留落在 0<=u<40, 0<=v<30 且位于相机"前方"的点(前方 Z 符号自动选择)
  - 聚合: 同一 ToF 像素内多个点, 对 LiDAR 前向距离 x 做平均(mean)
  - 显示: 把平均 x 映射成强度 u8=round(255/x), 再用 COLORMAP_TURBO 着色, 无点为黑
"""

import json
import sys
from pathlib import Path
import cv2

import numpy as np

# 允许在 cali 目录直接运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tof3d import ToF3DParams, tof_histograms  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "../data"
CALIB_JSON = HERE / "cali_result" / "calib_result.json"

# 点云显示参数(与 cali/check.py 对齐)
LIDAR_IMG_W = 700
LIDAR_IMG_H = 700
FOV_DEG = 70.0
HALF_FOV = float(np.deg2rad(FOV_DEG / 2.0))
LIDAR_VIS_MAX_RANGE_M = 20.0
LIDAR_NEAR_SAT_M = 1.0

# ToF 显示参数(与 cali/check.py 对齐)
TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 400
TOF_SHOW_H = 300
TOF_MIN_PEAK = 100
TOF_INTEN_GAMMA = 2.2
TOF_INTEN_TARGET_MEAN = 0.18

TITLE_H = 26


def list_env_dirs(root):
    # 列出所有场景目录, 按目录名排序
    if not root.exists():
        return []
    ds = [p for p in root.iterdir() if p.is_dir()]
    ds.sort(key=lambda p: p.name)
    return ds


def find_points_npz(env_dir):
    # 找到 points_last*.npz(默认取第一个匹配)
    cands = sorted(env_dir.glob("points_last*.npz"))
    return cands[0] if cands else None


def load_points(npz_path):
    # 读取 npz 中的 x/y/z, 返回 Nx3 点云(单位米)
    d = np.load(npz_path)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack([x, y, z]).astype(np.float32, copy=False)


def tof_intensity_to_u8(intensity_sum):
    # ToF 反射率显示映射(与 cali/check.py 思路一致):
    # 1) 用整图 mean 做归一化到 target_mean
    # 2) 做 gamma 显示(1/gamma)
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


def render_lidar_gray(points_xyz):
    # LiDAR 2D 投影渲染(仅用于显示):
    # - 只取 x>0 且 yaw/pitch 在 FOV 内的点
    # - 用 x 作为"距离", 映射成 u8=round(255/x)
    if points_xyz.shape[0] == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    x = points_xyz[:, 0].astype(np.float32, copy=False)
    y = points_xyz[:, 1].astype(np.float32, copy=False)
    z = points_xyz[:, 2].astype(np.float32, copy=False)
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, yaw, pitch = x[m], yaw[m], pitch[m]
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)

    depth_m = np.clip(x.astype(np.float32, copy=False), float(LIDAR_NEAR_SAT_M), float(LIDAR_VIS_MAX_RANGE_M))
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0.0, 255.0).astype(np.uint8)

    col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_IMG_W - 1)
    row = np.clip(row, 0, LIDAR_IMG_H - 1)

    img = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
    np.maximum.at(img, (row, col), depth_u8)
    return img


def load_calib(calib_json):
    # 读取 calib_result.json
    d = json.loads(calib_json.read_text(encoding="utf-8"))
    return {
        "camera_matrix": np.asarray(d["camera_matrix"], dtype=np.float64).reshape(3, 3),
        "rvec": np.asarray(d["rvec"], dtype=np.float64).reshape(3),
        "tvec": np.asarray(d["tvec"], dtype=np.float64).reshape(3),
    }


def build_tof_reflect_view(env_dir, cv2):
    # 生成 ToF 反射率图(400x300, flipV)
    tof_bgr = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    tof_path = env_dir / "tof.raw"
    if not tof_path.exists():
        return tof_bgr

    params = ToF3DParams(min_peak_count=float(TOF_MIN_PEAK))
    hists = tof_histograms(tof_path, params=params).astype(np.float32, copy=False)  # (30,40,64)
    if hists.size == 0:
        return tof_bgr

    inten = hists.sum(axis=2).astype(np.float32, copy=False)  # (30,40)
    inten_u8 = tof_intensity_to_u8(inten)
    inten_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
    inten_big = cv2.flip(inten_big, 0)
    return cv2.cvtColor(inten_big, cv2.COLOR_GRAY2BGR)


def project_lidar_to_tof(pts_lidar_xyz, calib, cv2):
    # 用标定参数把 LiDAR 3D 点投影到 ToF 像素坐标(u,v), 同时返回相机坐标系 Z 以判断前后
    if pts_lidar_xyz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    obj = pts_lidar_xyz.astype(np.float32, copy=False).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(
        obj,
        calib["rvec"].reshape(3, 1),
        calib["tvec"].reshape(3, 1),
        calib["camera_matrix"],
        np.zeros((5, 1), dtype=np.float64),
    )
    uv = uv.reshape(-1, 2).astype(np.float64, copy=False)

    R, _ = cv2.Rodrigues(calib["rvec"].reshape(3, 1))
    t = calib["tvec"].reshape(3, 1)
    xyz = pts_lidar_xyz.astype(np.float64, copy=False).T  # (3,N)
    cam = (R @ xyz) + t
    zc = cam[2, :].astype(np.float64, copy=False)
    return uv, zc


def resize_keep_aspect_to_h(img, target_h, cv2):
    # 按高度缩放, 保持宽高比(最近邻, 避免引入插值模糊)
    if img.size == 0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    if h <= 0 or w <= 0 or int(target_h) <= 0:
        return img
    if h == int(target_h):
        return img
    scale = float(target_h) / float(h)
    target_w = int(max(1, round(float(w) * scale)))
    return cv2.resize(img, (target_w, int(target_h)), interpolation=cv2.INTER_NEAREST)


def with_header(img, header_text, header_h, cv2):
    # 在图像上方增加黑色标题条(额外区域)
    if img.size == 0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    hh = int(max(1, int(header_h)))
    out = np.zeros((h + hh, w, 3), dtype=np.uint8)
    out[hh:, :, :] = img
    cv2.putText(out, header_text, (10, int(hh - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    # 读取标定
    if not CALIB_JSON.exists():
        raise FileNotFoundError("missing calib json: %s" % str(CALIB_JSON))
    calib = load_calib(CALIB_JSON)

    # 列出场景
    envs = list_env_dirs(Path(DATA_DIR))
    if not envs:
        raise FileNotFoundError("no scenes found under: %s" % str(DATA_DIR))

    cv2.namedWindow("VIEW", cv2.WINDOW_AUTOSIZE)

    scene_cache = {}

    def compute_scene(env):
        # 点云
        npz = find_points_npz(env)
        pts = load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)
        lidar_u8 = render_lidar_gray(pts)
        lidar_bgr = cv2.applyColorMap(lidar_u8, cv2.COLORMAP_TURBO)

        # ToF 反射率
        tof_reflect = build_tof_reflect_view(env, cv2=cv2)

        # 投影到 ToF 40x30, 并按每个 ToF 像素聚合(取 LiDAR 的 x 平均值)
        uv, zc = project_lidar_to_tof(pts, calib, cv2=cv2)
        n_all = int(uv.shape[0])
        n_z_pos = int(np.count_nonzero(zc > 1e-6)) if n_all > 0 else 0
        n_z_neg = int(np.count_nonzero(zc < -1e-6)) if n_all > 0 else 0

        u8map_flat = np.zeros((TOF_H * TOF_W,), dtype=np.uint8)

        if n_all > 0 and pts.shape[0] == n_all:
            u = uv[:, 0]
            v = uv[:, 1]
            # 自动选择"前方"的 Z 符号, 避免因坐标系差异导致全被过滤
            front = (zc > 1e-6) if (n_z_pos >= n_z_neg) else (zc < -1e-6)
            in_img = (u >= 0.0) & (u < float(TOF_W)) & (v >= 0.0) & (v < float(TOF_H))
            m = front & in_img
            if np.any(m):
                uu = u[m].astype(np.float32, copy=False)
                vv = v[m].astype(np.float32, copy=False)
                x_lidar = pts[:, 0].astype(np.float32, copy=False)[m]
                x_lidar = np.clip(x_lidar, float(LIDAR_NEAR_SAT_M), float(LIDAR_VIS_MAX_RANGE_M))

                ui = np.clip(np.floor(uu).astype(np.int32, copy=False), 0, TOF_W - 1)
                vi = np.clip(np.floor(vv).astype(np.int32, copy=False), 0, TOF_H - 1)

                pix = (vi * TOF_W + ui).astype(np.int32, copy=False)
                dsum = np.zeros((TOF_H * TOF_W,), dtype=np.float32)
                dcnt = np.zeros((TOF_H * TOF_W,), dtype=np.int32)
                np.add.at(dsum, pix, x_lidar)
                np.add.at(dcnt, pix, 1)

                hit = dcnt > 0
                if np.any(hit):
                    dmean = np.zeros_like(dsum)
                    dmean[hit] = dsum[hit] / np.maximum(dcnt[hit].astype(np.float32), 1.0)
                    dm = np.clip(dmean[hit], float(LIDAR_NEAR_SAT_M), float(LIDAR_VIS_MAX_RANGE_M))
                    u8map_flat[hit] = np.clip(np.rint(255.0 / dm), 0.0, 255.0).astype(np.uint8, copy=False)

        u8map = u8map_flat.reshape((TOF_H, TOF_W))
        u8_big = cv2.resize(u8map, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
        u8_big = cv2.flip(u8_big, 0)
        proj_full_color = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)
        proj_full_color[u8_big == 0] = (0, 0, 0)

        return {
            "lidar_bgr": lidar_bgr,
            "tof_reflect": tof_reflect,
            "proj_full_color": proj_full_color,
        }

    idx = 0
    bottom_show_proj = True

    while True:
        env = envs[idx]
        key = str(env)
        if key not in scene_cache:
            scene_cache[key] = compute_scene(env)

        cached = scene_cache[key]
        lidar_bgr = cached["lidar_bgr"]
        tof_reflect = cached["tof_reflect"]
        proj_full_color = cached["proj_full_color"]

        # 只在点云图上叠加文字, 避免遮挡右侧分析用图像
        lidar_show = lidar_bgr.copy()
        cv2.putText(lidar_show, "%s  (%d/%d)" % (env.name, idx + 1, len(envs)), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 拼接: 左侧整高; 右侧上下两张(带标题条); 总高度与左侧一致
        lidar_h = int(lidar_show.shape[0])
        header_h = int(TITLE_H)
        content_total = int(max(2, lidar_h - 2 * header_h))
        top_h = int(content_total // 2)
        bot_h = int(content_total - top_h)

        right_top = resize_keep_aspect_to_h(tof_reflect, top_h, cv2=cv2)
        right_bottom_src = proj_full_color if bottom_show_proj else tof_reflect
        right_bottom = resize_keep_aspect_to_h(right_bottom_src, bot_h, cv2=cv2)

        right_top = with_header(right_top, "TOF REFLECT", header_h, cv2=cv2)
        right_bottom = with_header(right_bottom, "LIAR" if bottom_show_proj else "TOF REFLECT", header_h, cv2=cv2)

        # 右侧上下两张补齐到同一宽度
        rw = int(max(right_top.shape[1], right_bottom.shape[1]))
        if int(right_top.shape[1]) != rw:
            pad = int(rw - int(right_top.shape[1]))
            right_top = cv2.copyMakeBorder(right_top, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if int(right_bottom.shape[1]) != rw:
            pad = int(rw - int(right_bottom.shape[1]))
            right_bottom = cv2.copyMakeBorder(right_bottom, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        right_col = np.vstack([right_top, right_bottom])
        if int(right_col.shape[0]) > int(lidar_h):
            right_col = right_col[: int(lidar_h), :, :]

        view = np.hstack([lidar_show, right_col])
        cv2.imshow("VIEW", view)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:
            break
        if k == ord("4"):
            idx = (idx - 1) % len(envs)
        if k == ord("6"):
            idx = (idx + 1) % len(envs)
        if k == ord(" "):
            bottom_show_proj = not bottom_show_proj

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


