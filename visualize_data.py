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

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from tof3d import ToF3DParams, tof_distance_and_histograms
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

TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 400
TOF_SHOW_H = 300
TOF_MIN_PEAK = 100  # 峰值低于该值认为置信度不足（标黑/深度置 0）
TOF_VALID_BINS = 62  # 峰值/质心只看前 62 个 bin（与 get_tof.py/tof3d.py 对齐）

# TOF 距离补偿（标定公式）
# 约定：x/y 为毫米（mm），最终显示仍用米（m）
TOF_COMP_ENABLE = True
TOF_COMP_A = 1
TOF_COMP_B_MM = -1447

# TOF 反射率强度图：强度=直方图各 bin 求和，显示时映射到 0~1023 并做 gamma=2.2
TOF_INTEN_GAMMA = 2.2
TOF_INTEN_USE_COLORMAP = False  # 需求：黑白图即可（无需伪彩）
TOF_INTEN_TARGET_MEAN = 0.18  # 通过除以系数 k，让整张图的平均反射率变为 0.18（之后再做 gamma 显示）


def _make_hist_image(hist: np.ndarray, x: int, y: int, depth_m: float, *, low_conf: bool) -> np.ndarray:
    """
    画一个简单的直方图窗口（OpenCV BGR）。
    hist: (64,) uint16/float
    """
    w, h = 520, 260
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)

    # 坐标轴
    left, right = 45, w - 15
    top, bottom = 25, h - 35
    cv2.rectangle(img, (left, top), (right, bottom), (70, 70, 70), 1)

    hist = np.asarray(hist, dtype=np.float32).reshape(-1)
    n = int(hist.size)
    if n <= 1:
        return img

    # 需求：直方图上限固定为 1024（便于不同像素对比）
    y_max = 1024.0
    hist_clip = np.clip(hist, 0.0, y_max)

    xs = np.linspace(left, right, n).astype(np.int32)
    ys = bottom - (hist_clip / y_max * (bottom - top)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(80, 220, 255), thickness=2, lineType=cv2.LINE_AA)

    # ---- 寻峰结果可视化：在“横坐标(bin)”位置画竖线 ----
    valid_n = int(min(n, TOF_VALID_BINS))
    peak_bin0 = -1
    peak_v = 0.0
    centroid_bin0 = 0.0
    if valid_n > 0:
        peak_bin0 = int(np.argmax(hist[:valid_n]))
        peak_v = float(hist[peak_bin0])
        if peak_v > 0.0:
            # 峰位（argmax）竖线：红色（低置信度则偏灰）
            peak_x = int(xs[peak_bin0])
            peak_color = (90, 90, 255) if low_conf else (0, 0, 255)
            #cv2.line(img, (peak_x, top), (peak_x, bottom), peak_color, 1, cv2.LINE_AA)

            # 可选：峰附近质心（更稳）竖线：绿色（与 tof3d/get_tof 的“clopBinNum=4”一致）
            r = 4
            s = max(0, min(peak_bin0, valid_n - 1) - r)
            e = min(valid_n, min(peak_bin0, valid_n - 1) + r)
            if e > s + 1:
                wts = hist[s:e].astype(np.float32, copy=False)
                denom = float(np.sum(wts))
                if denom > 0.0:
                    bins = np.arange(s, e, dtype=np.float32)
                    centroid_bin0 = float(np.dot(bins, wts) / denom)
                    # 把 centroid 的 bin(float) 映射到像素横坐标
                    cx = int(np.interp(centroid_bin0, np.arange(n, dtype=np.float32), xs.astype(np.float32)))
                    centroid_color = (90, 255, 90) if low_conf else (0, 255, 0)
                    cv2.line(img, (cx, top), (cx, bottom), centroid_color, 1, cv2.LINE_AA)

    # 信息文本
    dtxt = f"{depth_m:.3f} m" if depth_m > 0 else "invalid"
    extra = "  low_conf" if low_conf else ""
    cv2.putText(
        img,
        f"TOF Pixel (x={x}, y={y})   depth={dtxt}{extra}",
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"max={hist.max():.0f}  sum={hist.sum():.0f}  peak={peak_bin0 + 1 if peak_bin0 >= 0 else 0}  centroid={centroid_bin0 + 1:.2f}",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return img


def _tof_disp_xy_to_pixel(dx: int, dy: int) -> tuple[int, int]:
    """
    把 TOF 窗口上的坐标（已放大显示后的像素）映射回 30x40 的像素坐标。
    注意：当前显示做了 flipV（上下翻转），这里会映射回“未翻转”的原始 (x,y)。
    """
    px = int(dx * TOF_W / max(TOF_SHOW_W, 1))
    py_disp = int(dy * TOF_H / max(TOF_SHOW_H, 1))  # 显示坐标中的行
    px = int(np.clip(px, 0, TOF_W - 1))
    py_disp = int(np.clip(py_disp, 0, TOF_H - 1))
    py = (TOF_H - 1) - py_disp  # 还原 flipV
    return px, py


def _tof_pixel_to_disp_xy(px: int, py: int) -> tuple[int, int]:
    """
    把原始 30x40 像素坐标映射到显示窗口坐标中心点（用于画十字/圆点）。
    """
    px = int(np.clip(px, 0, TOF_W - 1))
    py = int(np.clip(py, 0, TOF_H - 1))
    py_disp = (TOF_H - 1) - py
    dx = int((px + 0.5) * TOF_SHOW_W / TOF_W)
    dy = int((py_disp + 0.5) * TOF_SHOW_H / TOF_H)
    return dx, dy


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


def _render_lidar_gray_and_depth(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    渲染 LiDAR 2D 投影：
    - gray_u8: (H,W) uint8（反比例亮度）
    - depth_m_map: (H,W) float32（米；无效为 0；每像素取最近点）
    """
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8), np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.float32)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8), np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.float32)

    # 以 x 作为距离（与原显示逻辑一致），并做量程裁剪
    depth_m = np.clip(x.astype(np.float32, copy=False), NEAR_SAT_M, MAX_RANGE_M)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (LIDAR_IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (LIDAR_IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_IMG_W - 1)
    row = np.clip(row, 0, LIDAR_IMG_H - 1)

    # 每个像素取最近点距离
    depth_map = np.full((LIDAR_IMG_H, LIDAR_IMG_W), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (row, col), depth_m)

    valid = np.isfinite(depth_map)
    depth_m_map = np.zeros_like(depth_map, dtype=np.float32)
    depth_m_map[valid] = depth_map[valid]

    gray = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.uint8)
    if np.any(valid):
        gray[valid] = np.clip(np.rint(255.0 / depth_m_map[valid]), 0.0, 255.0).astype(np.uint8)
    return gray, depth_m_map


def _auto_expose_u8(img: np.ndarray) -> np.ndarray:
    """
    每帧独立 AE：只用当前帧非零像素的分位数做拉伸，不做跨帧平滑。
    """
    if (not AUTO_EXPOSURE) or img.size == 0:
        return img
    nz = img[img > 0]
    if nz.size < 500:
        return img

    lo = float(np.percentile(nz, AE_LOW_PCT))
    hi = float(np.percentile(nz, AE_HIGH_PCT))
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


def _tof_intensity_to_u8(intensity_sum: np.ndarray) -> np.ndarray:
    """
    把 TOF 反射率强度（直方图求和）转为 u8：
    - 输入：任意数值类型 (H,W)，强度=∑hist[bin]
    - 显示策略（按需求）：通过除以系数 k，让整张图平均反射率变为 0.18，然后 gamma=2.2（显示用 1/gamma），最后映射到 0~255
    """
    if intensity_sum.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    v = np.asarray(intensity_sum, dtype=np.float32)

    mean = float(np.mean(v)) if v.size else 0.0
    if mean <= 0.0:
        return np.zeros(v.shape, dtype=np.uint8)

    # v / k，使得 mean(v/k) = target_mean  =>  k = mean / target_mean
    k = mean / float(TOF_INTEN_TARGET_MEAN)
    k = max(k, 1e-6)
    n = v / k  # 反射率（0..1+）
    n = np.clip(n, 0.0, 1.0)
    if float(TOF_INTEN_GAMMA) > 0:
        n = np.power(n, 1.0 / float(TOF_INTEN_GAMMA))
    return np.clip(np.rint(n * 255.0), 0.0, 255.0).astype(np.uint8)


def main() -> int:

    envs = _list_env_dirs()
    if not envs:
        raise FileNotFoundError(f"data 目录下没有环境数据：{DATA_DIR}")

    idx = 0

    cv2.namedWindow("LiDAR", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_INTEN", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_HIST", cv2.WINDOW_AUTOSIZE)

    hover = {"x": TOF_W // 2, "y": TOF_H // 2}
    lidar_hover = {"x": LIDAR_IMG_W // 2, "y": LIDAR_IMG_H // 2}

    def _on_tof_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            px, py = _tof_disp_xy_to_pixel(x, y)
            hover["x"], hover["y"] = px, py

    cv2.setMouseCallback("TOF", _on_tof_mouse)
    cv2.setMouseCallback("TOF_INTEN", _on_tof_mouse)

    def _on_lidar_mouse(event, x, y, flags, param):
        # “像 TOF 一样”：鼠标移动/点击都更新
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            lx = int(np.clip(x, 0, LIDAR_IMG_W - 1))
            ly = int(np.clip(y, 0, LIDAR_IMG_H - 1))
            lidar_hover["x"], lidar_hover["y"] = lx, ly

    cv2.setMouseCallback("LiDAR", _on_lidar_mouse)

    while True:
        env = envs[idx]

        # ---- LiDAR ----
        npz_path = _find_points_npz(env)
        lidar_view = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W, 3), dtype=np.uint8)
        lidar_pts = 0
        lidar_depth_map = np.zeros((LIDAR_IMG_H, LIDAR_IMG_W), dtype=np.float32)
        if npz_path is not None and npz_path.exists():
            x, y, z = _load_points(npz_path)
            lidar_pts = int(x.size)
            g0, lidar_depth_map = _render_lidar_gray_and_depth(x, y, z)
            g = _auto_expose_u8(g0)
            lidar_view = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)

        # ---- TOF ----
        tof_path = env / "tof.raw"
        tof_view = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
        tof_inten_view = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
        hist_view = np.zeros((260, 520, 3), dtype=np.uint8)
        if tof_path.exists():
            params = ToF3DParams(
                min_peak_count=float(TOF_MIN_PEAK),
                enable_distance_compensation=bool(TOF_COMP_ENABLE),
                distance_comp_a=float(TOF_COMP_A),
                distance_comp_b_mm=float(TOF_COMP_B_MM),
            )
            depth, hists = tof_distance_and_histograms(tof_path, params=params)  # depth:(30,40), hists:(30,40,64)
            peak = hists[:, :, :62].max(axis=2)  # (30,40) uint16
            low_conf_mask = peak < int(TOF_MIN_PEAK)

            # 反射率强度：直方图所有 bin 求和（按需求 0~1023 值域 + gamma=2.2）
            inten_sum = hists.sum(axis=2).astype(np.float32, copy=False)  # (30,40)
            inten_u8 = _tof_intensity_to_u8(inten_sum)
            inten_u8_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
            inten_u8_big = cv2.flip(inten_u8_big, 0)
            if TOF_INTEN_USE_COLORMAP:
                tof_inten_view = cv2.applyColorMap(inten_u8_big, cv2.COLORMAP_TURBO)
            else:
                tof_inten_view = cv2.cvtColor(inten_u8_big, cv2.COLOR_GRAY2BGR)

            u8 = _depth_to_u8(depth)
            u8_big = cv2.resize(u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
            # “旋转 180° + 左右镜像”等价于“只做上下镜像”
            u8_big = cv2.flip(u8_big, 0)
            tof_view = cv2.applyColorMap(u8_big, cv2.COLORMAP_TURBO)

            # 低置信度点标黑（按同样的 resize + flipV 映射到显示坐标）
            low_big = cv2.resize(low_conf_mask.astype(np.uint8) * 255, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
            low_big = cv2.flip(low_big, 0)
            tof_view[low_big > 0] = (0, 0, 0)

            # hover 直方图 + 距离
            hx = int(np.clip(hover["x"], 0, TOF_W - 1))
            hy = int(np.clip(hover["y"], 0, TOF_H - 1))
            hist = hists[hy, hx, :]
            d = float(depth[hy, hx]) if depth is not None else 0.0
            hist_view = _make_hist_image(hist, hx, hy, d, low_conf=bool(low_conf_mask[hy, hx]))

            # 在 TOF 图上标出 hover 点（显示坐标）
            dx, dy = _tof_pixel_to_disp_xy(hx, hy)
            cv2.circle(tof_view, (dx, dy), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(tof_view, (dx, dy), 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(tof_inten_view, (dx, dy), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(tof_inten_view, (dx, dy), 2, (0, 0, 0), -1, cv2.LINE_AA)

        # overlay
        # LiDAR hover 距离（像 TOF 一样显示数值）
        lhx = int(np.clip(lidar_hover["x"], 0, LIDAR_IMG_W - 1))
        lhy = int(np.clip(lidar_hover["y"], 0, LIDAR_IMG_H - 1))
        ld = float(lidar_depth_map[lhy, lhx]) if lidar_depth_map is not None else 0.0
        ld_txt = f"{ld:.3f} m" if ld > 0 else "invalid"
        cv2.circle(lidar_view, (lhx, lhy), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(lidar_view, (lhx, lhy), 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            lidar_view,
            f"{env.name}  ({idx+1}/{len(envs)})  pts={lidar_pts}  hover={ld_txt}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LiDAR", lidar_view)
        cv2.imshow("TOF", tof_view)
        cv2.imshow("TOF_INTEN", tof_inten_view)
        cv2.imshow("TOF_HIST", hist_view)

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


