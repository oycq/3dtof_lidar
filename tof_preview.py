#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof_preview.py

独立实时预览 ToF 数据：
- TOF_PREVIEW：实时反射率图（40x30 放大显示）
- TOF_HIST：鼠标悬停像素的单点直方图（前 62 bins）

交互：
- 鼠标移动/左键拖动：更新 hover 像素与直方图
- ESC：退出
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from tof3d import ToF3DParams, tof_distance_matrix_from_u16, tof_histograms_from_u16, tof_reflectance_mean3_max
from tof_server import ToFRealtimeServer

TOF_W = 40
TOF_H = 30
TOF_C = 64

SHOW_W = 300
SHOW_H = 400
HIST_SHOW_BINS = 62
MIN_PEAK = 100


def _tof_intensity_to_u8(intensity_sum: np.ndarray, *, gamma: float = 2.2, target_mean: float = 0.18) -> np.ndarray:
    if intensity_sum.size == 0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    v = np.asarray(intensity_sum, dtype=np.float32)
    mean = float(np.mean(v)) if v.size else 0.0
    if mean <= 0.0:
        return np.zeros(v.shape, dtype=np.uint8)
    k = max(mean / float(target_mean), 1e-6)
    n = np.clip(v / k, 0.0, 1.0)
    if float(gamma) > 0.0:
        n = np.power(n, 1.0 / float(gamma))
    return np.clip(np.rint(n * 255.0), 0.0, 255.0).astype(np.uint8)


def _disp_xy_to_pixel(dx: int, dy: int) -> tuple[int, int]:
    # 与 run.py 对齐：显示时做 rot90CW + flipH，映射后 display x -> 原始 py，display y -> 原始 px
    py = int(np.clip(dx * TOF_H / max(SHOW_W, 1), 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / max(SHOW_H, 1), 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int) -> tuple[int, int]:
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    dx = int(np.clip((py_i + 0.5) * SHOW_W / TOF_H, 0, SHOW_W - 1))
    dy = int(np.clip((px_i + 0.5) * SHOW_H / TOF_W, 0, SHOW_H - 1))
    return dx, dy


def _make_hist_image(hist: np.ndarray, px: int, py: int, depth_m: float, *, low_conf: bool) -> np.ndarray:
    w, h = 520, 260
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)

    left, right = 45, w - 15
    top, bottom = 25, h - 35
    cv2.rectangle(img, (left, top), (right, bottom), (70, 70, 70), 1)

    v = np.asarray(hist, dtype=np.float32).reshape(-1)
    n = int(min(v.size, HIST_SHOW_BINS))
    if n <= 1:
        return img

    # 只统计前 62 个 bin（等价于忽略 64-bin 数据中的最后两个 bin）。
    v = v[:n]
    y_max = 1024.0
    v_clip = np.clip(v, 0.0, y_max)
    plot_w = max(right - left, 1)
    bar_step = plot_w / float(n)
    for i in range(n):
        x0 = int(left + i * bar_step)
        x1 = int(left + (i + 1) * bar_step) - 1
        if x1 <= x0:
            x1 = x0 + 1
        y = int(bottom - (v_clip[i] / y_max * (bottom - top)))
        cv2.rectangle(img, (x0, y), (x1, bottom), (80, 220, 255), -1, cv2.LINE_AA)

    valid_n = int(n)
    peak_bin = -1
    centroid = 0.0
    peak_v = 0.0
    if valid_n > 0:
        peak_bin = int(np.argmax(v[:valid_n]))
        peak_v = float(v[peak_bin])
        if peak_v > 0.0:
            r = 4
            s = max(0, min(peak_bin, valid_n - 1) - r)
            e = min(valid_n, min(peak_bin, valid_n - 1) + r)
            if e > s + 1:
                wts = v[s:e].astype(np.float32, copy=False)
                denom = float(np.sum(wts))
                if denom > 0.0:
                    bins = np.arange(s, e, dtype=np.float32)
                    centroid = float(np.dot(bins, wts) / denom)
                    cx = int(left + (centroid + 0.5) * bar_step)
                    cx = int(np.clip(cx, left, right))
                    cv2.line(img, (cx, top), (cx, bottom), (0, 255, 0), 1, cv2.LINE_AA)

    d_txt = f"{depth_m:.3f}m" if depth_m > 0 else "invalid"
    conf_txt = " low_conf" if low_conf else ""
    cv2.putText(
        img,
        f"TOF Pixel (x={px}, y={py}) depth={d_txt}{conf_txt}",
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"max={v.max():.0f} sum62={v.sum():.0f} peak={peak_bin + 1 if peak_bin >= 0 else 0} centroid={centroid + 1:.2f}",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> int:
    cv2.namedWindow("TOF_PREVIEW", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_HIST", cv2.WINDOW_AUTOSIZE)

    hover = {"x": TOF_W // 2, "y": TOF_H // 2}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) not in (int(cv2.EVENT_MOUSEMOVE), int(cv2.EVENT_LBUTTONDOWN)):
            return
        px, py = _disp_xy_to_pixel(int(x), int(y))
        hover["x"], hover["y"] = int(px), int(py)

    cv2.setMouseCallback("TOF_PREVIEW", on_mouse)

    params = ToF3DParams(min_peak_count=float(MIN_PEAK))
    tof_srv = ToFRealtimeServer(queue_maxlen=5, min_peak_count=float(MIN_PEAK), target_fps=10.0)
    tof_srv.start()

    last_ts = 0.0
    cached_hists = np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    cached_depth = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    cached_inten = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    fps_show = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                raw_u16 = np.frombuffer(frame.raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16, params=params)
                if hists.shape == (TOF_H, TOF_W, TOF_C):
                    cached_hists = hists
                    cached_depth = tof_distance_matrix_from_u16(raw_u16, params=params)
                    cached_inten = tof_reflectance_mean3_max(hists)
                    last_ts = float(frame.ts)

                    t_now = time.perf_counter()
                    dt = max(t_now - t_prev, 1e-6)
                    fps_show = 1.0 / dt
                    t_prev = t_now

            inten_u8 = _tof_intensity_to_u8(cached_inten)
            disp_u8 = cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE)
            disp_u8 = cv2.flip(disp_u8, 1)
            disp_big = cv2.resize(disp_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            preview = cv2.cvtColor(disp_big, cv2.COLOR_GRAY2BGR)

            px = int(np.clip(hover["x"], 0, TOF_W - 1))
            py = int(np.clip(hover["y"], 0, TOF_H - 1))
            hist = cached_hists[py, px, :]
            depth_m = float(cached_depth[py, px])
            valid_n = int(min(hist.size, HIST_SHOW_BINS))
            peak = float(np.max(hist[:valid_n])) if valid_n > 0 else 0.0
            low_conf = bool(peak < float(MIN_PEAK))

            dx, dy = _pixel_to_disp_xy(px, py)
            cv2.circle(preview, (dx, dy), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(preview, (dx, dy), 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(
                preview,
                f"hover=({px},{py}) depth={'%.3fm' % depth_m if depth_m > 0 else 'invalid'} fps={fps_show:.1f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"hover=({px},{py}) depth={'%.3fm' % depth_m if depth_m > 0 else 'invalid'} fps={fps_show:.1f}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

            hist_view = _make_hist_image(hist, px, py, depth_m, low_conf=low_conf)
            cv2.imshow("TOF_PREVIEW", preview)
            cv2.imshow("TOF_HIST", hist_view)

            k = int(cv2.waitKey(1) & 0xFF)
            if k == 27:
                break
    finally:
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


