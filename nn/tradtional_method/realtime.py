#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

TOF_H, TOF_W, TOF_C = 30, 40, 64
HIST_BINS = 62
SHOW_W, SHOW_H, HEADER_H = 390, 520, 32
INVALID_BIN = 63
LOG_BASE = 1.06
EPS = 1e-6
DISP_GAMMA = 1.2
DEPTH_GAMMA = 1.6
CONF_MEAN_BIAS = 20.0
BIN_TO_DIST_M = 0.6
DIST_OFFSET_M = 0.5
DEPTH_NEAR_M = 1.0
DEPTH_FAR_M = 30.0


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    py = int(np.clip(dx * TOF_H / max(show_w, 1), 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / max(show_h, 1), 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> Tuple[int, int]:
    dx = int(np.clip((py + 0.5) * show_w / TOF_H, 0, show_w - 1))
    dy = int(np.clip((px + 0.5) * show_h / TOF_W, 0, show_h - 1))
    return dx, dy


def _depth_from_peak3_centroid(hists: np.ndarray) -> np.ndarray:
    bins62 = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    k = np.argmax(bins62, axis=2).astype(np.int32)
    kl = np.clip(k - 1, 0, HIST_BINS - 1)
    kr = np.clip(k + 1, 0, HIST_BINS - 1)
    rows = np.arange(TOF_H, dtype=np.int32)[:, None]
    cols = np.arange(TOF_W, dtype=np.int32)[None, :]
    wl, wc, wr = bins62[rows, cols, kl], bins62[rows, cols, k], bins62[rows, cols, kr]
    den = wl + wc + wr
    num = kl.astype(np.float32) * wl + k.astype(np.float32) * wc + kr.astype(np.float32) * wr
    centroid = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = den > 0.0
    centroid[m] = num[m] / den[m]
    return centroid * float(BIN_TO_DIST_M) + float(DIST_OFFSET_M)


def _conf_from_input(hists: np.ndarray) -> np.ndarray:
    b = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    y = np.max(b, axis=2)
    x = np.mean(b, axis=2) + float(CONF_MEAN_BIAS)
    conf = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = y > 0.0
    conf[m] = 1.0 - (x[m] / y[m])
    return np.clip(conf, 0.0, 1.0)


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python") from e

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tof3d import tof_histograms_from_u16  # noqa: E402
    from tof_server import ToFRealtimeServer  # noqa: E402

    cv2.namedWindow("TRADITIONAL_REALTIME", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("pThr%", "TRADITIONAL_REALTIME", 50, 100, lambda _: None)
    mouse = {"x": 0, "y": 0}

    def draw_text(img: np.ndarray, s: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, s, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def draw_marker(img: np.ndarray, x: int, y: int) -> np.ndarray:
        out = img.copy()
        cv2.circle(out, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def input_u8(h: np.ndarray) -> np.ndarray:
        inten = np.sum(h[:, :, :HIST_BINS].astype(np.float32), axis=2)
        m = np.isfinite(inten)
        if not np.any(m):
            return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
        vmin = float(np.percentile(inten[m], 5))
        vmax = float(np.percentile(inten[m], 95))
        if vmax <= vmin:
            return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
        n = np.clip((inten - vmin) / (vmax - vmin), 0.0, 1.0)
        n = np.power(n, 1.0 / float(DISP_GAMMA))
        return np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)

    def color_depth(depth: np.ndarray) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        valid = d > 0
        u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
        if np.any(valid):
            dc = np.clip(d[valid], DEPTH_NEAR_M, DEPTH_FAR_M)
            inv = 1.0 / np.clip(dc, EPS, np.inf)
            inv_n, inv_f = 1.0 / DEPTH_NEAR_M, 1.0 / DEPTH_FAR_M
            n = (inv - inv_f) / max(inv_n - inv_f, EPS)
            n = np.power(np.clip(n, 0.0, 1.0), 1.0 / float(DEPTH_GAMMA))
            u8[valid] = np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
        bgr[~valid] = (0, 0, 0)
        return bgr

    def color_conf(conf: np.ndarray) -> np.ndarray:
        n = np.power(np.clip(conf.astype(np.float32), 0.0, 1.0), 1.0 / float(DISP_GAMMA))
        u8 = np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=2)

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) == int(cv2.EVENT_MOUSEMOVE):
            mouse["x"] = int(x)
            mouse["y"] = int(y)

    cv2.setMouseCallback("TRADITIONAL_REALTIME", on_mouse)
    srv = ToFRealtimeServer(queue_maxlen=5, min_peak_count=100.0, target_fps=10.0)
    srv.start()

    ln_base = float(np.log(LOG_BASE))
    last_ts = 0.0
    cached_in = cached_pred = cached_conf = None
    try:
        while True:
            frame = srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                h = tof_histograms_from_u16(np.frombuffer(frame.raw_bytes, dtype=np.uint16))
                if h.shape == (TOF_H, TOF_W, TOF_C):
                    cached_in = h
                    cached_pred = _depth_from_peak3_centroid(h)
                    cached_conf = _conf_from_input(h)
                    last_ts = float(frame.ts)

            if cached_in is None or cached_pred is None or cached_conf is None:
                if int(cv2.waitKey(5) & 0xFF) == 27:
                    break
                continue

            p_thr = float(cv2.getTrackbarPos("pThr%", "TRADITIONAL_REALTIME")) / 100.0
            in_bgr = cv2.cvtColor(cv2.resize(cv2.flip(cv2.rotate(input_u8(cached_in), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            pred = color_depth(cached_pred)
            pred = pred.copy()
            pred[cached_conf < p_thr] = (0, 0, 0)
            pred_bgr = cv2.resize(cv2.flip(cv2.rotate(pred, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            conf_bgr = cv2.resize(cv2.flip(cv2.rotate(color_conf(cached_conf), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

            mx = int(np.clip(mouse["x"], 0, SHOW_W * 3 - 1))
            my = int(np.clip(mouse["y"] - HEADER_H, 0, SHOW_H - 1))
            tile_x0 = 0 if mx < SHOW_W else (SHOW_W if mx < SHOW_W * 2 else SHOW_W * 2)
            px, py = _disp_xy_to_pixel(mx - tile_x0, my, SHOW_W, SHOW_H)
            bins_px = np.asarray(cached_in[py, px, :HIST_BINS], dtype=np.float32)
            pred_v = float(cached_pred[py, px])
            conf_v = float(np.clip(cached_conf[py, px], 0.0, 1.0))
            refl = float(np.sum(bins_px))
            if bins_px.size == 0 or float(np.max(bins_px)) <= 0.0:
                hover = f"pred {pred_v:.3f}m  bin[{INVALID_BIN:02d}] INVALID  conf {conf_v:.2f}  refl {refl:.1f}"
            else:
                k = int(np.argmax(bins_px))
                a, b = float(np.exp((k - 0.5) * ln_base)), float(np.exp((k + 0.5) * ln_base))
                hover = f"pred {pred_v:.3f}m  bin[{k:02d}] {a:.2f}-{b:.2f}m  conf {conf_v:.2f}  refl {refl:.1f}"

            in_bgr = draw_text(in_bgr, f"INPUT(refl=bins[0..{HIST_BINS-1}], gamma={DISP_GAMMA:g})")
            pred_bgr = draw_text(pred_bgr, f"PRED(inv-depth+turbo, {DEPTH_NEAR_M:g}-{DEPTH_FAR_M:g}m, gamma={DEPTH_GAMMA:g})")
            conf_bgr = draw_text(conf_bgr, "CONF(1-(mean+bias)/max)")
            dx, dy = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)
            in_bgr = draw_marker(in_bgr, dx, dy)
            pred_bgr = draw_marker(pred_bgr, dx, dy)
            conf_bgr = draw_marker(conf_bgr, dx, dy)

            view = np.hstack([in_bgr, pred_bgr, conf_bgr])
            header = draw_text(np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8), hover)
            cv2.imshow("TRADITIONAL_REALTIME", np.vstack([header, view]))

            if int(cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
