#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


TOF_H = 30
TOF_W = 40
TOF_C = 64
EPS = 1e-6
TAIL_BASE = 1024.0
PULSES = 50000.0
MIN_SHOW_M = 1.0
MAX_GT_SHOW_M = 30.0
SHOW_W = 390
SHOW_H = 520
HEADER_H = 56
HIST_W = 640
HIST_H = 280


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    py = int(np.clip(dx * TOF_H / max(int(show_w), 1), 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / max(int(show_h), 1), 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> Tuple[int, int]:
    dx = int(np.clip((int(py) + 0.5) * max(int(show_w), 1) / TOF_H, 0, max(int(show_w), 1) - 1))
    dy = int(np.clip((int(px) + 0.5) * max(int(show_h), 1) / TOF_W, 0, max(int(show_h), 1) - 1))
    return dx, dy


def _with_text(img_bgr: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    cv2.putText(out, text, (10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    xx = int(np.clip(x, 0, out.shape[1] - 1))
    yy = int(np.clip(y, 0, out.shape[0] - 1))
    cv2.circle(out, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_histogram_bgr(bins: np.ndarray, w: int = HIST_W, h: int = HIST_H) -> np.ndarray:
    import cv2  # type: ignore

    b = np.asarray(bins, dtype=np.float32).reshape(-1)[:TOF_C]
    img = np.zeros((max(int(h), 1), max(int(w), 1), 3), dtype=np.uint8)
    if b.size <= 0:
        return img
    b_draw = b[:62]
    tail_63 = float(b[62]) if b.size > 62 else 0.0
    tail_64 = float(b[63]) if b.size > 63 else 0.0
    sat_value = tail_64 * TAIL_BASE + tail_63
    if sat_value == 0.0:
        sat_value = PULSES
    vmax_raw = float(np.max(b_draw)) if b_draw.size > 0 else 0.0
    vmax_eq_sat = vmax_raw * PULSES / sat_value

    x0, y0 = 14, 128
    x1, y1 = img.shape[1] - 10, img.shape[0] - 18
    vmax = 1.0 if (not np.isfinite(vmax_raw) or vmax_raw <= 0.0) else vmax_raw
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    bar_w = max(int((x1 - x0) / max(b_draw.size, 1)), 1)
    for i, v in enumerate(b_draw):
        vv = float(v) if np.isfinite(v) and float(v) > 0.0 else 0.0
        hh = int(np.clip(vv / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        xl = x0 + i * bar_w
        xr = min(xl + bar_w, x1)
        if xr <= xl:
            continue
        yt = y1 - hh
        cv2.rectangle(img, (xl, yt), (xr, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (xl, yt), (xr, y1), (30, 30, 30), 1)
    img = _with_text(img, "RAW_HIST (only 0-61 bins)", y=24)
    img = _with_text(img, f"max={vmax_raw:.3f}", y=48)
    img = _with_text(img, f"sat_value={sat_value:.3f}", y=72)
    img = _with_text(img, f"max_eq_sat={vmax_eq_sat:.3f}", y=96)
    return img


def _inv_depth_range_from_depth(depth_m: np.ndarray) -> Tuple[float, float] | None:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d >= float(MIN_SHOW_M))
    if not np.any(valid):
        return None
    inv_v = 1.0 / np.clip(d[valid], EPS, np.inf)
    vmin = float(np.min(inv_v))
    vmax = float(np.max(inv_v))
    return (vmin, vmax if vmax > vmin else vmin + 1e-6)


def _colorize_depth_with_range(depth_m: np.ndarray, inv_vmin: float, inv_vmax: float) -> np.ndarray:
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d >= float(MIN_SHOW_M))
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    inv = np.zeros_like(d, dtype=np.float32)
    inv[valid] = 1.0 / np.clip(d[valid], EPS, np.inf)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    u8[valid] = np.clip(np.rint((inv[valid] - inv_vmin) / (inv_vmax - inv_vmin) * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_gray01(x: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=2)


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for ip in sorted(train_dir.glob("input_*.npy")):
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def _load_pair(ip: Path, op: Path) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("missing dependency torch") from e

    nn_dir = Path(__file__).resolve().parent
    train_dir = nn_dir / "train_data"
    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    cv2.namedWindow("CHECK_NET", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) == int(cv2.EVENT_MOUSEMOVE):
            mouse["x"] = int(x)
            mouse["y"] = int(y)

    cv2.setMouseCallback("CHECK_NET", on_mouse)

    idx = 0
    while True:
        ip, op = pairs[idx]
        x, gt = _load_pair(ip, op)
        with torch.no_grad():
            inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            pred_depth_t, snr_t, reflectance_t, conf_t = net(inp)
            pred_depth = pred_depth_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            snr_map = snr_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            reflectance_map = reflectance_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            conf_map = conf_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

        gt_for_disp = np.where(np.isfinite(gt) & (gt >= float(MIN_SHOW_M)) & (gt <= float(MAX_GT_SHOW_M)), gt, 0.0)
        inv_range = _inv_depth_range_from_depth(gt_for_disp)
        if inv_range is None:
            gt_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
            pred_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
        else:
            gt_bgr = _colorize_depth_with_range(gt_for_disp, inv_range[0], inv_range[1])
            pred_bgr = _colorize_depth_with_range(pred_depth, inv_range[0], inv_range[1])
        pred_bgr = pred_bgr.copy()
        pred_bgr[conf_map <= 0.5] = (0, 0, 0)

        refl_bgr = cv2.cvtColor(cv2.resize(cv2.flip(cv2.rotate(np.clip(np.rint(np.clip(reflectance_map, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        gt_big = cv2.resize(cv2.flip(cv2.rotate(gt_bgr, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        pred_big = cv2.resize(cv2.flip(cv2.rotate(pred_bgr, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        conf_big = cv2.resize(cv2.flip(cv2.rotate(_colorize_gray01(conf_map), cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

        mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 2 - 1))
        my = int(np.clip(int(mouse.get("y", 0)) - int(HEADER_H), 0, SHOW_H * 2 - 1))
        tile_x0 = 0 if mx < SHOW_W else SHOW_W
        tile_y0 = 0 if my < SHOW_H else SHOW_H
        px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, SHOW_W, SHOW_H)
        dx, dy = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)

        for img in [refl_bgr, gt_big, pred_big, conf_big]:
            marked = _draw_marker(img, dx, dy)
            img[:] = marked

        hover1 = f"pred {float(pred_depth[py, px]):.3f}m  gt {float(gt[py, px]):.3f}m  conf {float(conf_map[py, px]):.0f}  snr {float(snr_map[py, px]):.3f}"
        hover2 = f"reflectance {float(reflectance_map[py, px]) * 100.0:.3f}%"

        refl_bgr = _with_text(refl_bgr, "REFLECTANCE")
        gt_big = _with_text(gt_big, "GT")
        pred_big = _with_text(pred_big, "PRED (conf==1)")
        conf_big = _with_text(conf_big, "CONF")
        view = np.vstack([np.zeros((HEADER_H, SHOW_W * 2, 3), dtype=np.uint8), np.hstack([refl_bgr, gt_big]), np.hstack([pred_big, conf_big])])
        view[:HEADER_H] = _with_text(view[:HEADER_H], f"sample {idx + 1}/{len(pairs)}  |  {hover1}", y=22)
        view[:HEADER_H] = _with_text(view[:HEADER_H], hover2, y=46)
        cv2.imshow("CHECK_NET", view)

        hist_img = _render_histogram_bgr(x[py, px, :])
        cv2.imshow("HIST", hist_img)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:
            break
        if k == ord("4"):
            idx = (idx - 1) % len(pairs)
        if k == ord("6"):
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
