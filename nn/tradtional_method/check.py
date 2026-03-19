#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

TOF_H, TOF_W, TOF_C = 30, 40, 64
HIST_BINS = 62
SHOW_W, SHOW_H, HEADER_H = 390, 520, 32
HIST_W, HIST_H = 620, 260
INVALID_BIN = 63
MAX_GT_SHOW_M = 30.0
EPS = 1e-6
DISP_GAMMA = 1.2
DEPTH_GAMMA = 1.6
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


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    out = []
    for ip in ins:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            out.append((ip, op))
    return out


def _load_pair(ip: Path, op: Path) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


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
    k = np.argmax(b, axis=2).astype(np.int32)
    y = np.max(b, axis=2)
    x = np.mean(b, axis=2)
    conf = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = y > 0.0
    conf[m] = 1.0 - (x[m] / y[m])
    low_peak = ((k < 15) & (y < 40.0)) | ((k >= 15) & (y < 20.0))
    conf[low_peak] = 0.0
    return np.clip(conf, 0.0, 1.0)


def _guess_train_dir(this_dir: Path) -> Path:
    for p in [this_dir / "train_data", this_dir.parent / "train_data"]:
        if p.exists():
            return p
    return this_dir / "train_data"


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python") from e

    train_dir = _guess_train_dir(Path(__file__).resolve().parent)
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")
    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

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

    def render_raw_histogram(bins: np.ndarray) -> np.ndarray:
        b = np.asarray(bins, dtype=np.float32).reshape(-1)
        nb = int(b.shape[0])
        img = np.zeros((HIST_H, HIST_W, 3), dtype=np.uint8)
        if nb <= 0:
            return img
        x0, y0 = 14, 34
        x1, y1 = HIST_W - 10, HIST_H - 18
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return img
        vmax = float(np.max(b))
        if (not np.isfinite(vmax)) or vmax <= 0.0:
            vmax = 1.0
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
        bar_w = max(int(max(x1 - x0, 1) / nb), 1)
        for i in range(nb):
            v = float(b[i]) if np.isfinite(b[i]) and b[i] > 0.0 else 0.0
            hh = int(np.clip(v / vmax, 0.0, 1.0) * (y1 - y0 - 1))
            xl = x0 + i * bar_w
            xr = min(xl + bar_w, x1)
            if xr <= xl:
                continue
            yt = y1 - hh
            cv2.rectangle(img, (xl, yt), (xr, y1), (255, 220, 0), -1)
            cv2.rectangle(img, (xl, yt), (xr, y1), (30, 30, 30), 1)
        return draw_text(img, f"RAW_HIST bins[0..{nb-1}] max {vmax:.3f} sum {float(np.sum(b)):.3f}")

    cv2.namedWindow("CHECK_TRADITIONAL", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("pThr%", "CHECK_TRADITIONAL", 50, 100, lambda _: None)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) == int(cv2.EVENT_MOUSEMOVE):
            mouse["x"] = int(x)
            mouse["y"] = int(y)

    cv2.setMouseCallback("CHECK_TRADITIONAL", on_mouse)
    idx = 0

    while True:
        x, gt = _load_pair(*pairs[idx])
        pred = _depth_from_peak3_centroid(x)
        conf = _conf_from_input(x)

        p_thr = float(cv2.getTrackbarPos("pThr%", "CHECK_TRADITIONAL")) / 100.0
        in_bgr = cv2.cvtColor(cv2.resize(cv2.flip(cv2.rotate(input_u8(x), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        gt_clip = np.where((gt > 0.0) & (gt <= MAX_GT_SHOW_M) & np.isfinite(gt), gt, 0.0)
        gt_bgr = cv2.resize(cv2.flip(cv2.rotate(color_depth(gt_clip), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        pred_map = color_depth(pred)
        pred_map[conf < p_thr] = (0, 0, 0)
        pred_bgr = cv2.resize(cv2.flip(cv2.rotate(pred_map, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        conf_bgr = cv2.resize(cv2.flip(cv2.rotate(color_conf(conf), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

        mx = int(np.clip(mouse["x"], 0, SHOW_W * 2 - 1))
        my = int(np.clip(mouse["y"] - HEADER_H, 0, SHOW_H * 2 - 1))
        tile_x0 = 0 if mx < SHOW_W else SHOW_W
        tile_y0 = 0 if my < SHOW_H else SHOW_H
        px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, SHOW_W, SHOW_H)

        bins_px = np.asarray(x[py, px, :HIST_BINS], dtype=np.float32)
        pr_v, gt_v, cf_v = float(pred[py, px]), float(gt[py, px]), float(np.clip(conf[py, px], 0.0, 1.0))
        refl = float(np.sum(bins_px))
        hover = f"pred {pr_v:.3f}m  gt {gt_v:.3f}m  conf {cf_v:.2f}  refl {refl:.1f}"

        in_bgr = draw_text(in_bgr, f"INPUT(refl=bins[0..{HIST_BINS-1}], gamma={DISP_GAMMA:g})")
        gt_bgr = draw_text(gt_bgr, "GT")
        pred_bgr = draw_text(pred_bgr, f"PRED(inv-depth+turbo, {DEPTH_NEAR_M:g}-{DEPTH_FAR_M:g}m, gamma={DEPTH_GAMMA:g})")
        conf_bgr = draw_text(conf_bgr, "CONF(1-mean/max, k<15:y<40->0 else y<20->0)")
        dx, dy = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)
        in_bgr, gt_bgr, pred_bgr, conf_bgr = draw_marker(in_bgr, dx, dy), draw_marker(gt_bgr, dx, dy), draw_marker(pred_bgr, dx, dy), draw_marker(conf_bgr, dx, dy)

        top = np.hstack([in_bgr, gt_bgr])
        bot = np.hstack([pred_bgr, conf_bgr])
        view = np.vstack([top, bot])
        header = draw_text(np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8), f"sample {idx + 1}/{len(pairs)}  |  {hover}")
        cv2.imshow("CHECK_TRADITIONAL", np.vstack([header, view]))
        cv2.imshow("HIST", render_raw_histogram(x[py, px, :HIST_BINS]))

        key = int(cv2.waitKey(30) & 0xFF)
        if key == 27:
            break
        if key == ord("4"):
            idx = (idx - 1) % len(pairs)
        elif key == ord("6"):
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
