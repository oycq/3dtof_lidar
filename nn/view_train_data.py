#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
view_train_data.py

用于快速检查 rectify.py 导出的 train_data：
- input_00001.npy: ToF 直方图 (30,40,64) float32
- output_00001.npy: 聚合 LiDAR 距离图 (30,40) float32，单位米，无点为 0

交互：
- 4：上一帧
- 6：下一帧
- ESC：退出
- 鼠标悬停在 input 图：下方显示该像素直方图
- 鼠标悬停在 output 图：标题显示该像素距离
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


TOF_W = 40
TOF_H = 30
TOF_BINS = 64

SHOW_W = 400
SHOW_H = 300

HIST_H = 220
TITLE_H = 28


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标（显示做了 flipV，因此这里需要还原）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
    py_disp = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    py = (TOF_H - 1) - py_disp
    return px, py


def _render_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """把 (H,W,64) 直方图转成强度灰度图 (H,W) uint8。"""
    inten = np.sum(hists.astype(np.float32, copy=False), axis=2)  # (30,40)
    if inten.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    vmax = float(np.max(inten))
    if vmax <= 0.0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    img = np.clip(np.rint(inten / vmax * 255.0), 0, 255).astype(np.uint8)
    return img


def _render_hist_plot(hist: np.ndarray, w: int, h: int) -> np.ndarray:
    """渲染单个像素的 64-bin 直方图到 BGR 图。"""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # 边框/坐标系
    pad_l, pad_r, pad_t, pad_b = 40, 10, 12, 22
    x0, x1 = pad_l, w - pad_r
    y0, y1 = pad_t, h - pad_b
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 60, 60), 1, cv2.LINE_AA)

    hist = np.asarray(hist, dtype=np.float32).reshape(-1)
    if hist.size != TOF_BINS:
        cv2.putText(canvas, "hist shape invalid", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    vmax = float(np.max(hist))
    if vmax <= 0.0:
        cv2.putText(canvas, "hist = all zeros", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return canvas

    xs = np.linspace(x0, x1, TOF_BINS).astype(np.int32)
    ys = (y1 - (hist / vmax) * (y1 - y0)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=False, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

    # 标注
    cv2.putText(canvas, f"max={vmax:.0f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(canvas, "bin", (x1 - 60, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H,W) float32 米 -> BGR 伪彩，0 为黑。"""
    d = np.asarray(depth_m, dtype=np.float32)
    if d.size == 0:
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    dv = d[valid]
    vmin = float(np.min(dv))
    vmax = float(np.max(dv))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    u8[valid] = np.clip(np.rint((d[valid] - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _with_header(img: np.ndarray, text: str, header_h: int) -> np.ndarray:
    hh = max(int(header_h), 1)
    h, w = img.shape[:2]
    out = np.zeros((h + hh, w, 3), dtype=np.uint8)
    out[hh:, :] = img
    cv2.putText(out, text, (10, hh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    inputs = sorted(train_dir.glob("input_*.npy"))
    pairs: List[Tuple[Path, Path]] = []
    for ip in inputs:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="train_data", help="train_data 目录路径")
    args = ap.parse_args()

    train_dir = Path(args.dir).resolve()
    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    cv2.namedWindow("TRAIN_DATA", cv2.WINDOW_AUTOSIZE)

    state = {
        "mx": 0,
        "my": 0,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event != cv2.EVENT_MOUSEMOVE:
            return
        state["mx"] = x
        state["my"] = y

    cv2.setMouseCallback("TRAIN_DATA", on_mouse)

    idx = 0
    cached_in: np.ndarray | None = None
    cached_out: np.ndarray | None = None
    cached_idx = -1

    while True:
        ip, op = pairs[idx]

        if cached_idx != idx:
            cached_in = np.load(str(ip))  # (30,40,64)
            cached_out = np.load(str(op))  # (30,40)
            cached_idx = idx

        assert cached_in is not None and cached_out is not None
        if cached_in.shape != (TOF_H, TOF_W, TOF_BINS):
            raise ValueError(f"bad input shape: {cached_in.shape} ({ip.name})")
        if cached_out.shape != (TOF_H, TOF_W):
            raise ValueError(f"bad output shape: {cached_out.shape} ({op.name})")

        inten_u8 = _render_intensity_u8(cached_in)
        in_big = cv2.resize(inten_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        in_big = cv2.flip(in_big, 0)
        in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

        out_bgr_small = _colorize_depth(cached_out)
        out_big = cv2.resize(out_bgr_small, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        out_big = cv2.flip(out_big, 0)

        # hover 解析
        mx, my = int(state["mx"]), int(state["my"])
        hover_hist = np.zeros((TOF_BINS,), dtype=np.float32)
        hover_txt_in = ""
        hover_txt_out = ""

        # 布局：上面一行两张图（带标题条），下面一行直方图
        x_in0 = 0
        x_out0 = SHOW_W
        y_img0 = TITLE_H
        y_img1 = TITLE_H + SHOW_H

        if y_img0 <= my < y_img1:
            if x_in0 <= mx < x_in0 + SHOW_W:
                px, py = _disp_xy_to_pixel(mx - x_in0, my - y_img0, SHOW_W, SHOW_H)
                hover_hist = cached_in[py, px, :].astype(np.float32, copy=False)
                hover_txt_in = f" | hover=({px},{py})"
            elif x_out0 <= mx < x_out0 + SHOW_W:
                px, py = _disp_xy_to_pixel(mx - x_out0, my - y_img0, SHOW_W, SHOW_H)
                d = float(cached_out[py, px])
                hover_txt_out = f" | hover=({px},{py}) {d:.3f}m" if d > 0 else f" | hover=({px},{py}) --"

        hist_img = _render_hist_plot(hover_hist, w=SHOW_W * 2, h=HIST_H)
        hist_img = _with_header(hist_img, "HISTOGRAM (input pixel)", TITLE_H)

        left = _with_header(in_bgr, f"INPUT (ToF intensity){hover_txt_in}  {ip.name}  ({idx+1}/{len(pairs)})", TITLE_H)
        right = _with_header(out_big, f"OUTPUT (depth m){hover_txt_out}  {op.name}", TITLE_H)
        top = np.hstack([left, right])
        view = np.vstack([top, hist_img])

        cv2.imshow("TRAIN_DATA", view)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            break
        elif k == ord("4"):  # prev
            idx = (idx - 1) % len(pairs)
        elif k == ord("6"):  # next
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


