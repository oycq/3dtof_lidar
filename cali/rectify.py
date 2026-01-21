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
- 聚合: 同一 ToF 像素内多个点, 取"最近"(x 最小)的前 5%-10% 点再求均值
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


def _tof_disp_xy_to_pixel(dx: int, dy: int, *, show_w: int, show_h: int) -> tuple[int, int]:
    """
    把“显示窗口中的坐标”(已 resize 后的像素)映射回 ToF 40x30 像素坐标。
    注意：rectify.py 的 ToF 显示同样做了 flipV(上下翻转)。

    返回:
      (px, py): 原始 ToF 像素坐标（未翻转的 py）
    """
    sw = int(max(show_w, 1))
    sh = int(max(show_h, 1))
    px = int(dx * TOF_W / sw)
    py_disp = int(dy * TOF_H / sh)  # 显示坐标中的行
    px = int(np.clip(px, 0, TOF_W - 1))
    py_disp = int(np.clip(py_disp, 0, TOF_H - 1))
    py = (TOF_H - 1) - py_disp  # 还原 flipV
    return px, py


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
    # 旧实现使用 cv2.projectPoints，速度较慢；这里改成纯 numpy 投影：
    # cam = R*X + t; u = fx*Xc/Zc + cx; v = fy*Yc/Zc + cy

    def _prepare_calib_fast(calib_dict):
        # 缓存 R/t/K（每次 scene 复用），避免重复 Rodrigues/类型转换
        if "_R_fast" not in calib_dict:
            R64, _ = cv2.Rodrigues(calib_dict["rvec"].reshape(3, 1))
            calib_dict["_R_fast"] = R64.astype(np.float32, copy=False)
        if "_t_fast" not in calib_dict:
            # (1,3) 便于广播相加
            calib_dict["_t_fast"] = calib_dict["tvec"].reshape(1, 3).astype(np.float32, copy=False)
        if "_K_fast" not in calib_dict:
            calib_dict["_K_fast"] = calib_dict["camera_matrix"].astype(np.float32, copy=False)
        return calib_dict["_R_fast"], calib_dict["_t_fast"], calib_dict["_K_fast"]

    def _project_lidar_to_tof_fast(pts_lidar_xyz_, R_, t_, K_):
        if pts_lidar_xyz_.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        X = pts_lidar_xyz_.astype(np.float32, copy=False)  # (N,3)
        cam = (R_ @ X.T).T + t_  # (N,3)
        zc_ = cam[:, 2].astype(np.float32, copy=False)

        # avoid div0, keep for later filtering
        z = zc_.copy()
        z[z == 0.0] = 1e-6

        fx = float(K_[0, 0])
        fy = float(K_[1, 1])
        cx = float(K_[0, 2])
        cy = float(K_[1, 2])

        u = fx * (cam[:, 0] / z) + cx
        v = fy * (cam[:, 1] / z) + cy
        uv_ = np.stack([u, v], axis=1).astype(np.float32, copy=False)
        return uv_, zc_

    R, t, K = _prepare_calib_fast(calib)
    return _project_lidar_to_tof_fast(pts_lidar_xyz, R, t, K)


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

    # 鼠标悬停：用于右下角投影图显示“距离”
    mouse_state: dict = {"x": 0, "y": 0}

    def _on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse_state["x"] = int(x)
        mouse_state["y"] = int(y)

    cv2.setMouseCallback("VIEW", _on_mouse)

    scene_cache = {}

    def compute_scene(env):
        # 点云
        npz = find_points_npz(env)
        pts = load_points(npz) if npz is not None and npz.exists() else np.zeros((0, 3), dtype=np.float32)
        lidar_u8 = render_lidar_gray(pts)
        lidar_bgr = cv2.applyColorMap(lidar_u8, cv2.COLORMAP_TURBO)

        # ToF 反射率
        tof_reflect = build_tof_reflect_view(env, cv2=cv2)

        # 投影到 ToF 40x30, 并按每个 ToF 像素聚合:
        # - 不用全体均值, 而是取距离最近(x 最小)的前 5%-10% 点再求均值
        uv, zc = project_lidar_to_tof(pts, calib, cv2=cv2)
        n_all = int(uv.shape[0])
        n_z_pos = int(np.count_nonzero(zc > 1e-6)) if n_all > 0 else 0
        n_z_neg = int(np.count_nonzero(zc < -1e-6)) if n_all > 0 else 0

        u8map_flat = np.zeros((TOF_H * TOF_W,), dtype=np.uint8)
        # 记录“每个 ToF 像素的聚合距离（x 距离）”，用于鼠标悬停显示
        dmap_flat = np.zeros((TOF_H * TOF_W,), dtype=np.float32)

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

                TOP_FRAC = 0.10  # 0.10=最近10%; 你要更"近"就改成 0.05

                pix = (vi * TOF_W + ui).astype(np.int32, copy=False)
                order = np.argsort(pix, kind="stable")
                pix_s = pix[order]
                x_s = x_lidar[order].astype(np.float32, copy=False)

                if pix_s.size > 0:
                    cuts = np.flatnonzero(pix_s[1:] != pix_s[:-1]) + 1
                    starts = np.concatenate([np.array([0], dtype=np.int32), cuts.astype(np.int32, copy=False)])
                    ends = np.concatenate([cuts.astype(np.int32, copy=False), np.array([pix_s.size], dtype=np.int32)])
                    grp_ids = pix_s[starts]

                    for gid, s, e in zip(grp_ids.tolist(), starts.tolist(), ends.tolist()):
                        vals = x_s[int(s) : int(e)]
                        n = int(vals.size)
                        if n <= 0:
                            continue
                        k = int(max(1, int(np.ceil(float(n) * float(TOP_FRAC)))))

                        # 取最小的 k 个(最近的一小撮), 用 partition 比全排序快
                        if k >= n:
                            nearest = vals
                        else:
                            nearest = np.partition(vals, k - 1)[:k]

                        d = float(np.mean(nearest))
                        d = float(np.clip(d, float(LIDAR_NEAR_SAT_M), float(LIDAR_VIS_MAX_RANGE_M)))
                        u8map_flat[int(gid)] = np.clip(np.rint(255.0 / d), 0.0, 255.0).astype(np.uint8)
                        dmap_flat[int(gid)] = float(d)

        u8map = u8map_flat.reshape((TOF_H, TOF_W))
        dmap = dmap_flat.reshape((TOF_H, TOF_W))
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
        proj_dmap_m = cached.get("proj_dmap_m", None)

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

        # --- 鼠标悬停距离：仅对右下角“投影图”生效 ---
        hover_txt = ""
        if bool(bottom_show_proj) and isinstance(proj_dmap_m, np.ndarray) and proj_dmap_m.size:
            # view 拼接后：右侧起点 x = lidar_show.width；右上高度 = right_top(with header).height
            view_x0 = int(lidar_show.shape[1])
            # right_top 还没加 header，这里先算加 header 后高度
            right_top_h_total = int(right_top.shape[0] + header_h)

            mx = int(mouse_state.get("x", 0))
            my = int(mouse_state.get("y", 0))

            # 右下角内容区域（不含 header）：(view_x0 .. view_x0+right_bottom.w, right_top_h_total+header_h .. +right_bottom.h)
            if (mx >= view_x0) and (mx < view_x0 + int(right_bottom.shape[1])):
                yy0 = int(right_top_h_total + header_h)
                yy1 = int(right_top_h_total + header_h + int(right_bottom.shape[0]))
                if (my >= yy0) and (my < yy1):
                    dx = int(mx - view_x0)
                    dy = int(my - yy0)
                    px, py = _tof_disp_xy_to_pixel(dx, dy, show_w=int(right_bottom.shape[1]), show_h=int(right_bottom.shape[0]))
                    d = float(proj_dmap_m[py, px])
                    if d > 0.0:
                        hover_txt = f" | hover=({px},{py}) {d:.3f}m"
                    else:
                        hover_txt = f" | hover=({px},{py}) --"

        right_top = with_header(right_top, "TOF REFLECT", header_h, cv2=cv2)
        right_bottom_title = ("LIDAR PROJ" if bottom_show_proj else "TOF REFLECT") + hover_txt
        right_bottom = with_header(right_bottom, right_bottom_title, header_h, cv2=cv2)

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


