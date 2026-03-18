#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TOF_H = 30
TOF_W = 40
TOF_C = 64
TOTAL_PIXELS = TOF_H * TOF_W
SHOW_W = 390 * 2
SHOW_H = 520 * 2
HEADER_H = 36
# 交互预览窗口单面板尺寸（3x2 拼图），尽量保证一屏可见。
PREVIEW_W = 300
PREVIEW_H = 400

# 固定配置（按需求不走命令行）
CONF_THR = 0.5
REL_ERR_THR = 0.12
EPS = 1e-6
DEPTH_NEAR_M = 1.0
DEPTH_FAR_M = 30.0
DEPTH_GAMMA = 1.6
LIDAR_FOV_DEG = 70.0


@dataclass
class SceneStats:
    scene_name: str
    input_path: Path
    output_path: Path
    wrong_ratio: float
    wrong_count: int
    conf_count: int
    total_valid: int
    gt_mean: float
    pred_mean: float
    abs_err_mean: float
    gt_map: np.ndarray
    pred_map: np.ndarray
    conf_map: np.ndarray
    conf_mask: np.ndarray
    wrong_mask: np.ndarray


def find_pairs(train_dir: Path) -> list[tuple[Path, Path]]:
    inputs = sorted(train_dir.glob("input_*.npy"))
    pairs: list[tuple[Path, Path]] = []
    for ip in inputs:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def load_pair(ip: Path, op: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def load_network(nn_dir: Path, ckpt_path: Path):
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    net.load_state_dict(sd, strict=True)
    return net, device


def natural_key(s: str) -> list[object]:
    parts = re.split(r"(\d+)", s)
    out: list[object] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def scene_name_from_input(ip: Path) -> str:
    name = ip.stem
    if name.startswith("input_"):
        return name[len("input_") :]
    return name


def print_progress(prefix: str, idx: int, total: int) -> None:
    total_safe = max(int(total), 1)
    done = int(idx)
    ratio = float(done) / float(total_safe)
    width = 30
    fill = int(round(ratio * width))
    bar = "#" * fill + "-" * (width - fill)
    print(f"\r{prefix} [{bar}] {done}/{total_safe} ({ratio * 100.0:5.1f}%)", end="", flush=True)
    if done >= total_safe:
        print("")


def _rotate_for_display(img: np.ndarray):
    import cv2  # type: ignore

    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)


def _draw_text(img: np.ndarray, s: str, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    import cv2  # type: ignore

    out = img.copy()
    cv2.putText(out, s, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def _draw_marker(img: np.ndarray, x: int, y: int) -> np.ndarray:
    import cv2  # type: ignore

    out = img.copy()
    xx = int(np.clip(x, 0, out.shape[1] - 1))
    yy = int(np.clip(y, 0, out.shape[0] - 1))
    cv2.circle(out, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> tuple[int, int]:
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    py = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> tuple[int, int]:
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    dx = int(np.clip((py_i + 0.5) * sw / TOF_H, 0, sw - 1))
    dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    return dx, dy


def _render_input_reflect_u8(hists: np.ndarray) -> np.ndarray:
    h = np.asarray(hists, dtype=np.float32)
    inten = np.sum(h[:, :, :62], axis=2)
    if inten.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    m = np.isfinite(inten)
    if not np.any(m):
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    vmin = float(np.percentile(inten[m], 5))
    vmax = float(np.percentile(inten[m], 95))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    n = np.clip((inten - vmin) / (vmax - vmin), 0.0, 1.0)
    n = np.power(n, 1.0 / 1.2)  # 显示 gamma
    return np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)


def _render_histogram_bgr(
    bins: np.ndarray,
    w: int,
    h: int,
    title: str,
    max_bins: int | None = None,
    fixed_vmax: float | None = None,
) -> np.ndarray:
    import cv2  # type: ignore

    b = np.asarray(bins, dtype=np.float32).reshape(-1)
    if max_bins is not None:
        b = b[: int(max(0, max_bins))]
    sw = max(int(w), 1)
    sh = max(int(h), 1)
    img = np.zeros((sh, sw, 3), dtype=np.uint8)
    if b.size <= 0:
        return _draw_text(img, f"{title} empty")

    top, left, right, bottom = 34, 14, 10, 18
    x0, y0 = left, top
    x1, y1 = sw - right, sh - bottom
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return img

    if fixed_vmax is not None:
        vmax = float(fixed_vmax)
        if (not np.isfinite(vmax)) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = float(np.max(b))
        if (not np.isfinite(vmax)) or vmax <= 0.0:
            vmax = 1.0

    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    nb = int(b.size)
    bar_w = max(int(max(x1 - x0, 1) / max(nb, 1)), 1)
    for i in range(nb):
        v = float(b[i])
        if (not np.isfinite(v)) or v < 0.0:
            v = 0.0
        hh = int(np.clip(v / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        xl = x0 + i * bar_w
        xr = min(xl + bar_w, x1)
        if xr <= xl:
            continue
        yt = y1 - hh
        cv2.rectangle(img, (xl, yt), (xr, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (xl, yt), (xr, y1), (30, 30, 30), 1)

    step = max(1, nb // 6)
    for k in range(0, nb, step):
        xx = x0 + int(k * bar_w)
        cv2.line(img, (xx, y1), (xx, y1 + 4), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (xx + 2, sh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return _draw_text(img, f"{title}  max {vmax:.3f}")


def _load_points_xyz(npz_path: Path) -> np.ndarray:
    d = np.load(str(npz_path))
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.column_stack([x, y, z]).astype(np.float32, copy=False)
    finite = np.isfinite(pts).all(axis=1)
    dist2 = np.sum(pts * pts, axis=1)
    return pts[finite & (dist2 > 1e-12)]


def _render_lidar_pointcloud_bgr(points_xyz: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    import cv2  # type: ignore

    w = max(int(out_w), 1)
    h = max(int(out_h), 1)
    if points_xyz.shape[0] == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    x, y, z = points_xyz.T
    half_fov = float(np.deg2rad(LIDAR_FOV_DEG / 2.0))
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, np.hypot(x, y))
    mask = (x > 0) & (np.abs(yaw) <= half_fov) & (np.abs(pitch) <= half_fov)
    if not np.any(mask):
        return np.zeros((h, w, 3), dtype=np.uint8)

    x_m, y_m, z_m = x[mask], y[mask], z[mask]
    depth_m = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    depth_u8 = np.clip(np.rint(255.0 / np.clip(depth_m, EPS, np.inf)), 0, 255).astype(np.uint8)
    col = ((half_fov - yaw[mask]) / (2 * half_fov) * (w - 1)).astype(np.int32)
    row = ((half_fov - pitch[mask]) / (2 * half_fov) * (h - 1)).astype(np.int32)
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)

    gray = np.zeros((h, w), dtype=np.uint8)
    np.maximum.at(gray, (row, col), depth_u8)
    bgr = cv2.applyColorMap(gray, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    bgr[gray == 0] = (0, 0, 0)
    return bgr


def _render_lidar_panel_keep_aspect(points_xyz: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    # LiDAR 视场在水平/竖直方向一致，按正方形渲染可避免圆形目标被拉伸成椭圆。
    pw = max(int(panel_w), 1)
    ph = max(int(panel_h), 1)
    side = max(1, min(pw, ph))
    square = _render_lidar_pointcloud_bgr(points_xyz, side, side)
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
    x0 = (pw - side) // 2
    y0 = (ph - side) // 2
    canvas[y0 : y0 + side, x0 : x0 + side] = square
    return canvas

def _safe_index_from_scene_name(scene_name: str) -> int | None:
    m = re.fullmatch(r"0*(\d+)", scene_name.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _find_lidar_npz_for_scene(scene_name: str, data_roots: list[Path]) -> Path | None:
    # 优先按场景名精确匹配目录。
    for root in data_roots:
        if not root.exists():
            continue
        p = root / scene_name
        if p.is_dir():
            cands = sorted(p.glob("points_last*.npz"))
            if cands:
                return cands[0]

    # 若 scene_name 是 00001 这类编号，则按目录排序做索引映射（导出 train_data 常见顺序）。
    idx1 = _safe_index_from_scene_name(scene_name)
    if idx1 is not None and idx1 > 0:
        for root in data_roots:
            if not root.exists():
                continue
            dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
            if idx1 <= len(dirs):
                cands = sorted(dirs[idx1 - 1].glob("points_last*.npz"))
                if cands:
                    return cands[0]

    # 最后尝试模糊匹配（避免命名带前后缀时完全找不到）。
    sn = scene_name.lower()
    for root in data_roots:
        if not root.exists():
            continue
        for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            dn = d.name.lower()
            if (sn in dn) or (dn in sn):
                cands = sorted(d.glob("points_last*.npz"))
                if cands:
                    return cands[0]
    return None


def _load_scene_input_for_preview(item: SceneStats) -> np.ndarray:
    x, _ = load_pair(item.input_path, item.output_path)
    return x


def show_rank_preview(rows: list[SceneStats], data_roots: list[Path]) -> None:
    import cv2  # type: ignore

    wnd = "SCENE_RANK_PREVIEW"
    cv2.namedWindow(wnd, cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback(wnd, on_mouse)

    idx = 0
    hover_px, hover_py = TOF_W // 2, TOF_H // 2
    cache_idx = -1
    cache_x: np.ndarray | None = None
    cache_lidar_panel: np.ndarray | None = None

    while True:
        item = rows[idx]
        if cache_idx != idx:
            cache_x = _load_scene_input_for_preview(item=item)
            npz_path = _find_lidar_npz_for_scene(item.scene_name, data_roots=data_roots)
            if npz_path is not None and npz_path.exists():
                pts = _load_points_xyz(npz_path)
                cache_lidar_panel = _render_lidar_panel_keep_aspect(pts, PREVIEW_W, PREVIEW_H)
            else:
                miss = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                miss = _draw_text(miss, "LIDAR_POINTCLOUD missing")
                cache_lidar_panel = miss
            cache_idx = idx
        assert cache_x is not None and cache_lidar_panel is not None

        refl_u8 = _render_input_reflect_u8(cache_x)
        refl_bgr = cv2.cvtColor(refl_u8, cv2.COLOR_GRAY2BGR)
        gt_bgr = _color_depth(item.gt_map)  # LIDAR 投影真值图
        pred_bgr = _color_depth(item.pred_map)
        pred_bgr[item.conf_map < float(CONF_THR)] = (0, 0, 0)  # conf<50% 直接置黑
        wrong_bgr = pred_bgr.copy()
        wrong_bgr[item.wrong_mask] = (0, 0, 255)

        refl_show = cv2.resize(_rotate_for_display(refl_bgr), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
        gt_show = cv2.resize(_rotate_for_display(gt_bgr), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
        lidar_show = cache_lidar_panel.copy()
        pred_show = cv2.resize(_rotate_for_display(pred_bgr), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
        wrong_show = cv2.resize(_rotate_for_display(wrong_bgr), (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)

        mx = int(np.clip(mouse["x"], 0, PREVIEW_W * 3 - 1))
        my = int(np.clip(mouse["y"] - HEADER_H, 0, PREVIEW_H * 2 - 1))
        tile_col = mx // PREVIEW_W
        tile_row = my // PREVIEW_H
        if (tile_row, tile_col) in {(0, 0), (0, 1), (0, 2), (1, 1)}:
            local_x = int(mx - tile_col * PREVIEW_W)
            local_y = int(my - tile_row * PREVIEW_H)
            hover_px, hover_py = _disp_xy_to_pixel(local_x, local_y, PREVIEW_W, PREVIEW_H)
        hover_px = int(np.clip(hover_px, 0, TOF_W - 1))
        hover_py = int(np.clip(hover_py, 0, TOF_H - 1))

        dx, dy = _pixel_to_disp_xy(hover_px, hover_py, PREVIEW_W, PREVIEW_H)
        refl_show = _draw_marker(refl_show, dx, dy)
        gt_show = _draw_marker(gt_show, dx, dy)
        pred_show = _draw_marker(pred_show, dx, dy)
        wrong_show = _draw_marker(wrong_show, dx, dy)

        in_hist = _render_histogram_bgr(cache_x[hover_py, hover_px, :], PREVIEW_W, PREVIEW_H, "HIST", max_bins=62)

        refl_show = _draw_text(refl_show, "TOF_REFLECT")
        gt_show = _draw_text(gt_show, "LIDAR_GT")
        lidar_show = _draw_text(lidar_show, "LIDAR_POINTCLOUD")
        pred_show = _draw_text(pred_show, "PRED")
        wrong_show = _draw_text(wrong_show, "ERROR_ANNOTATION")

        # 2x3 固定布局：
        # 1,1=GT  1,2=PRED  1,3=ERROR
        # 2,1=LIDAR  2,2=REFLECT  2,3=HIST
        top = np.hstack([gt_show, pred_show, wrong_show])
        bot = np.hstack([lidar_show, refl_show, in_hist])
        view = np.vstack([top, bot])

        pred_v = float(item.pred_map[hover_py, hover_px])
        gt_v = float(item.gt_map[hover_py, hover_px])
        header_text = (
            f"rank {idx + 1}/{len(rows)} bad->good  scene {item.scene_name}  "
            f"wrong {item.wrong_ratio * 100.0:.2f}% ({item.wrong_count}/{TOTAL_PIXELS})  "
            f"pred {pred_v:.3f}m  gt {gt_v:.3f}m"
        )
        header = _draw_text(np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8), header_text)
        cv2.imshow(wnd, np.vstack([header, view]))

        key = int(cv2.waitKey(30) & 0xFF)
        if key == 27:  # ESC
            break
        if key == ord("4"):
            idx = (idx - 1) % len(rows)
        elif key == ord("6"):
            idx = (idx + 1) % len(rows)

    cv2.destroyWindow(wnd)


def _color_depth(depth: np.ndarray) -> np.ndarray:
    # 配色和 check.py 保持一致：inv-depth + turbo + gamma
    import cv2  # type: ignore

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    if np.any(valid):
        dc = np.clip(d[valid], DEPTH_NEAR_M, DEPTH_FAR_M)
        inv = 1.0 / np.clip(dc, EPS, np.inf)
        inv_n = 1.0 / float(DEPTH_NEAR_M)
        inv_f = 1.0 / float(DEPTH_FAR_M)
        n = (inv - inv_f) / max(inv_n - inv_f, EPS)
        n = np.power(np.clip(n, 0.0, 1.0), 1.0 / float(DEPTH_GAMMA))
        u8[valid] = np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    bgr[~valid] = (0, 0, 0)
    return bgr


def _wrong_mask_bgr(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    out[mask] = (0, 0, 255)
    return out


def _wrong_mask_with_neighbor_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    conf_pos: np.ndarray,
    rel_err_thr: float,
) -> np.ndarray:
    """
    投影存在轻微偏移时，允许与 3x3 邻域内任一 GT 匹配：
    只要最小相对误差 <= rel_err_thr，就判定为对。
    """
    h, w = gt.shape
    pred_f = np.asarray(pred, dtype=np.float32)
    gt_f = np.asarray(gt, dtype=np.float32)
    conf = np.asarray(conf_pos, dtype=bool)

    min_rel_err = np.full((h, w), np.inf, dtype=np.float32)
    has_neighbor = np.zeros((h, w), dtype=bool)

    # 用 nan pad，便于统一切片，不引入边界伪值。
    gt_pad = np.pad(gt_f, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
    valid_gt_pad = np.isfinite(gt_pad) & (gt_pad > 0.0)

    # 3x3 邻域（含自身）
    for dy in range(3):
        for dx in range(3):
            gt_nb = gt_pad[dy : dy + h, dx : dx + w]
            valid_nb = valid_gt_pad[dy : dy + h, dx : dx + w]
            rel = np.full((h, w), np.inf, dtype=np.float32)
            rel[valid_nb] = np.abs(pred_f[valid_nb] - gt_nb[valid_nb]) / np.clip(np.abs(gt_nb[valid_nb]), EPS, np.inf)
            min_rel_err = np.minimum(min_rel_err, rel)
            has_neighbor |= valid_nb

    wrong_mask = conf & has_neighbor & (min_rel_err > float(rel_err_thr))
    return wrong_mask


def save_scene_image(item: SceneStats, rank_idx: int, out_dir: Path) -> None:
    import cv2  # type: ignore

    gt_bgr = _color_depth(item.gt_map)
    pred_raw_bgr = _color_depth(item.pred_map)
    pred_raw_bgr[~item.conf_mask] = (0, 0, 0)  # conf<50% 不参与绘制
    pred_mark_bgr = pred_raw_bgr.copy()
    pred_mark_bgr[item.wrong_mask] = (0, 0, 255)  # 错误像素标红

    pred_raw_show = cv2.resize(_rotate_for_display(pred_raw_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    gt_show = cv2.resize(_rotate_for_display(gt_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    pred_mark_show = cv2.resize(_rotate_for_display(pred_mark_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

    pred_raw_show = _draw_text(pred_raw_show, "PRED_RAW")
    gt_show = _draw_text(gt_show, "GT")
    pred_mark_show = _draw_text(pred_mark_show, "PRED_RED(wrong, conf>=50%)", (0, 0, 255))

    view = np.hstack([pred_raw_show, gt_show, pred_mark_show])
    header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
    header_text = (
        f"rank {rank_idx + 1}  scene {item.scene_name}  "
        f"wrong_rate {item.wrong_ratio:.6f} ({item.wrong_count}/{TOTAL_PIXELS})  "
        f"gt {item.gt_mean:.3f}m  pred {item.pred_mean:.3f}m  abs_err {item.abs_err_mean:.3f}m"
    )
    header = _draw_text(header, header_text)
    canvas = np.vstack([header, view])

    safe_scene = re.sub(r"[^\w\-\.]+", "_", item.scene_name)
    out_path = out_dir / f"{rank_idx + 1:03d}_{safe_scene}.png"
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"failed to write image: {out_path}")


def show_error_rate_hist(rows: list[SceneStats]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency matplotlib, run: py -m pip install matplotlib") from e

    # 兼容 Windows 常见中文字体，避免中文乱码；并修复负号显示为方块的问题。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    err_pct = np.array([100.0 * float(r.wrong_ratio) for r in rows], dtype=np.float32)
    # 聚焦低错误率区间，便于比较接近的场景。
    err_pct = np.clip(err_pct, 0.0, 40.0)

    plt.figure(figsize=(10, 6))
    # 0~40% 每 1% 一个 bin。
    bins = np.arange(0.0, 41.0, 1.0)
    plt.hist(err_pct, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.title("各场景错误率分布直方图")
    plt.xlabel("错误率（%）")
    plt.ylabel("场景数量")
    plt.xlim(0.0, 40.0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()


def main() -> int:
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    here = Path(__file__).resolve().parent
    nn_dir = here.parent
    train_dir = nn_dir / "train_data"
    ckpt_path = nn_dir / "model_last.pt"
    out_dir = here / "rank"

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    net, device = load_network(nn_dir=nn_dir, ckpt_path=ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧结果，避免上次遗留图片干扰排序观察。
    for p in out_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass
    old_csv = out_dir / "ranking.csv"
    if old_csv.exists():
        try:
            old_csv.unlink()
        except OSError:
            pass

    rows: list[SceneStats] = []
    conf_thr = float(CONF_THR)
    rel_err_thr = float(REL_ERR_THR)

    with torch.no_grad():
        total_pairs = len(pairs)
        for idx, (ip, op) in enumerate(pairs, start=1):
            x, gt = load_pair(ip, op)
            inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            out = net.forward_train(inp)
            pred = out["dist"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            conf = out["conf"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

            valid = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(conf) & (gt > 0.0)
            total_valid = int(np.count_nonzero(valid))
            conf_pos = valid & (conf >= conf_thr)  # conf<50% 不参与统计
            abs_err_map = np.abs(pred - gt).astype(np.float32, copy=False)
            wrong_mask = _wrong_mask_with_neighbor_gt(
                pred=pred,
                gt=gt,
                conf_pos=conf_pos,
                rel_err_thr=rel_err_thr,
            )
            conf_count = int(np.count_nonzero(conf_pos))
            wrong_count = int(np.count_nonzero(wrong_mask))
            wrong_ratio = float(wrong_count) / float(TOTAL_PIXELS)

            scene_name = scene_name_from_input(ip)
            gt_v = gt[conf_pos]
            pred_v = pred[conf_pos]
            err_v = abs_err_map[conf_pos]
            if total_valid == 0 and conf_count == 0:
                continue
            rows.append(
                SceneStats(
                    scene_name=scene_name,
                    input_path=ip,
                    output_path=op,
                    wrong_ratio=wrong_ratio,
                    wrong_count=wrong_count,
                    conf_count=conf_count,
                    total_valid=total_valid,
                    gt_mean=float(np.mean(gt_v)) if gt_v.size else float("nan"),
                    pred_mean=float(np.mean(pred_v)) if pred_v.size else float("nan"),
                    abs_err_mean=float(np.mean(err_v)) if err_v.size else float("nan"),
                    gt_map=gt,
                    pred_map=pred,
                    conf_map=conf,
                    conf_mask=conf_pos,
                    wrong_mask=wrong_mask,
                )
            )
            print_progress("统计进度", idx, total_pairs)

    if not rows:
        raise RuntimeError("no valid scene found after filtering")

    rows.sort(
        key=lambda r: (
            -r.wrong_ratio,
            -r.wrong_count,
            natural_key(r.scene_name),
        )
    )
    # 校验：rank 越往后，错误率不升高（允许持平）。
    for i in range(1, len(rows)):
        if rows[i].wrong_ratio > rows[i - 1].wrong_ratio:
            raise RuntimeError(
                "rank order invalid: later rank has higher wrong_ratio "
                f"(rank={i+1}, prev={rows[i-1].wrong_ratio:.8f}, curr={rows[i].wrong_ratio:.8f})"
            )

    total_ranked = len(rows)
    for i, item in enumerate(rows, start=1):
        save_scene_image(item=item, rank_idx=i - 1, out_dir=out_dir)
        print_progress("出图进度", i, total_ranked)

    csv_path = out_dir / "ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "scene_name",
                "wrong_ratio_wrong_over_1200",
                "wrong_count",
                "conf_count",
                "valid_count",
                "gt_mean_m",
                "pred_mean_m",
                "abs_err_mean_m",
                "input_file",
                "output_file",
            ]
        )
        for i, item in enumerate(rows, start=1):
            writer.writerow(
                [
                    i,
                    item.scene_name,
                    f"{item.wrong_ratio:.6f}",
                    item.wrong_count,
                    item.conf_count,
                    item.total_valid,
                    f"{item.gt_mean:.6f}" if math.isfinite(item.gt_mean) else "nan",
                    f"{item.pred_mean:.6f}" if math.isfinite(item.pred_mean) else "nan",
                    f"{item.abs_err_mean:.6f}" if math.isfinite(item.abs_err_mean) else "nan",
                    item.input_path.name,
                    item.output_path.name,
                ]
            )

    print(f"[done] pairs={len(pairs)}")
    print(f"[done] ranked scenes={len(rows)}")
    print(f"[done] rank dir: {out_dir}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] conf threshold: >= {conf_thr:.2f}")
    print(f"[done] wrong criterion: |pred-gt|/gt > {rel_err_thr * 100.0:.2f}%")
    print(f"[done] wrong ratio definition: wrong_count/{TOTAL_PIXELS}")
    project_root = nn_dir.parent
    data_roots = [project_root / "data", project_root / "cali" / "data"]
    show_rank_preview(rows, data_roots=data_roots)
    show_error_rate_hist(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

