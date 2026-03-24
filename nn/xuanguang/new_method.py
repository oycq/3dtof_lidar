#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

TOF_H = 30
TOF_W = 40
TOF_C = 64

NUM_BINS = 64
HIST_BINS = 62
SHOW_W = 390
SHOW_H = 520
HEADER_H = 56
HIST_W = 620
HIST_H = 260

EPS = 1e-6
DEPTH_FAR_M = 35.0
DEPTH_MAP_CLIP_MIN = 1.5
DEPTH_MAP_SCALE = 1.5
DISP_GAMMA = 1.2
SUM_GATE_MAX = 20000.0
SUM_GATE_SNR_DIV = 3.0
PEAK_GATE_MIN = 30.0
SNR_GATE_LE3M = 5.5
SNR_GATE_3TO5M = 5.0
SNR_GATE_5TO8M = 4.5
SNR_GATE_GT8M = 4.0
SNR_SHOW_MAX = 10.0
REFLECT_SAT_VALUE = 1023.0
REFLECT_SAT_SCALE = 50000.0
REFLECT_DENOM = 156250.0
REFLECT_THRESH = 0.025
REFLECT_SHOW_GAMMA = 2.2

LIDAR_W = 700
LIDAR_H = 700
LIDAR_FOV_DEG = 70.0
LIDAR_HALF_FOV = np.deg2rad(LIDAR_FOV_DEG / 2.0)
LIDAR_MAX_RANGE_M = 20.0
LIDAR_NEAR_SAT_M = 1.0

MAIN_WINDOW = "XUANGUANG_SCENES"


@dataclass
class SceneState:
    scene_dir: Path
    scene_name: str
    hists: np.ndarray
    pred_depth: np.ndarray
    out_probs: np.ndarray
    snr: np.ndarray
    conf_mask: np.ndarray
    reflect_peak: np.ndarray
    reflectance: np.ndarray
    reflect_valid_mask: np.ndarray
    reflect_black_mask: np.ndarray
    conf_mask_after_reflect: np.ndarray
    points: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None
    lidar_view: np.ndarray | None


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    py = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> Tuple[int, int]:
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    dx = int(np.clip((py_i + 0.5) * sw / TOF_H, 0, sw - 1))
    dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    return dx, dy


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    xx = int(np.clip(x, 0, out.shape[1] - 1))
    yy = int(np.clip(y, 0, out.shape[0] - 1))
    cv2.circle(out, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _with_text(img_bgr: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    cv2.putText(out, text, (10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _render_histogram_bgr(
    bins: np.ndarray,
    w: int = HIST_W,
    h: int = HIST_H,
    max_bins: int | None = None,
    title: str = "HIST",
    fixed_vmax: float | None = None,
) -> np.ndarray:
    import cv2  # type: ignore

    b = np.asarray(bins, dtype=np.float32).reshape(-1)
    if max_bins is None:
        max_bins = int(HIST_BINS)
    nb = int(min(int(max_bins), b.shape[0]))
    if nb <= 0:
        return np.zeros((max(int(h), 1), max(int(w), 1), 3), dtype=np.uint8)

    b = b[:nb]
    sw = max(int(w), 1)
    sh = max(int(h), 1)
    img = np.zeros((sh, sw, 3), dtype=np.uint8)

    top = 76
    left = 14
    right = 10
    bottom = 18
    x0, y0 = left, top
    x1, y1 = sw - right, sh - bottom
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return img

    if fixed_vmax is not None:
        vmax = float(fixed_vmax)
        if (not np.isfinite(vmax)) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = float(np.max(b)) if b.size else 0.0
        if (not np.isfinite(vmax)) or vmax <= 0.0:
            vmax = 1.0

    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)

    bar_area_w = max(x1 - x0, 1)
    bar_w = max(int(bar_area_w / nb), 1)
    for i in range(nb):
        v = float(b[i])
        if (not np.isfinite(v)) or v < 0.0:
            v = 0.0
        hh = int(np.clip(v / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        x_l = x0 + i * bar_w
        x_r = min(x_l + bar_w, x1)
        if x_r <= x_l:
            continue
        y_t = y1 - hh
        cv2.rectangle(img, (x_l, y_t), (x_r, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (x_l, y_t), (x_r, y1), (30, 30, 30), 1)

    for k in range(0, nb, 10):
        xx = x0 + int(k * bar_w)
        cv2.line(img, (xx, y1), (xx, y1 + 4), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (xx + 2, sh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    img = _with_text(img, f"{title} bins[0..{nb-1}]  max {vmax:.3f}  sum {float(np.sum(b)):.3f}")
    return img


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    inten = np.sum(hists.astype(np.float32, copy=False), axis=2)
    vmax = float(np.max(inten)) if inten.size else 0.0
    if vmax <= 0.0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    return np.clip(np.rint(inten / vmax * 255.0), 0, 255).astype(np.uint8)


def _compute_snr_from_input(hists: np.ndarray) -> np.ndarray:
    h = np.asarray(hists, dtype=np.float32)
    if h.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad hists shape: {h.shape}")

    src = h[:, :, :HIST_BINS]
    vmax = np.max(src, axis=2)
    vsum = np.sum(src, axis=2, dtype=np.float32)
    mean = np.mean(src, axis=2, dtype=np.float32)
    std = np.std(src, axis=2, dtype=np.float32)
    snr = (vmax - mean) / np.maximum(std, 1e-6)

    snr = np.where(vsum > float(SUM_GATE_MAX), snr / float(max(SUM_GATE_SNR_DIV, 1.0)), snr)
    snr = np.where(vmax < float(PEAK_GATE_MIN), 0.0, snr)
    return snr.astype(np.float32, copy=False)


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    d_safe = np.maximum(d[valid], EPS)
    y = float(DEPTH_MAP_SCALE) / d_safe
    y_min = float(DEPTH_MAP_SCALE) / float(max(DEPTH_FAR_M, DEPTH_MAP_CLIP_MIN))
    y_max = float(DEPTH_MAP_SCALE) / float(max(DEPTH_MAP_CLIP_MIN, EPS))
    norm = (y - y_min) / max(y_max - y_min, EPS)
    u8[valid] = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m] / float(SNR_SHOW_MAX), 0.0, 1.0), 1.0 / float(DISP_GAMMA))
    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _adjust_reflect_peak(peak: np.ndarray | float, bin_63: np.ndarray | float, bin_64: np.ndarray | float) -> np.ndarray:
    peak_arr = np.asarray(peak, dtype=np.float32)
    bin_63_arr = np.asarray(bin_63, dtype=np.float32)
    bin_64_arr = np.asarray(bin_64, dtype=np.float32)
    denom = bin_64_arr * float(REFLECT_SAT_VALUE) + bin_63_arr
    sat_mask = np.isfinite(peak_arr) & (peak_arr == float(REFLECT_SAT_VALUE))
    valid_mask = sat_mask & np.isfinite(denom) & (denom > 0.0)
    adjusted = np.where(
        valid_mask,
        float(REFLECT_SAT_VALUE) * float(REFLECT_SAT_SCALE) / denom,
        peak_arr,
    )
    return adjusted.astype(np.float32, copy=False)


def _compute_reflectance(hists: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    peak = np.max(src, axis=2)
    bin_63 = np.asarray(hists[:, :, HIST_BINS], dtype=np.float32) if hists.shape[2] > HIST_BINS else np.zeros_like(peak, dtype=np.float32)
    bin_64 = (
        np.asarray(hists[:, :, HIST_BINS + 1], dtype=np.float32)
        if hists.shape[2] > (HIST_BINS + 1)
        else np.zeros_like(peak, dtype=np.float32)
    )
    peak = _adjust_reflect_peak(peak, bin_63, bin_64)
    valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(peak) & (peak >= 0.0)
    reflectance = np.full(depth.shape, np.nan, dtype=np.float32)
    reflectance[valid] = peak[valid] * depth[valid] * depth[valid] / float(REFLECT_DENOM)
    black_mask = valid & (reflectance < float(REFLECT_THRESH))
    return peak.astype(np.float32, copy=False), reflectance, black_mask


def _colorize_reflectance(reflectance: np.ndarray, valid: np.ndarray) -> np.ndarray:
    r = np.asarray(reflectance, dtype=np.float32)
    m = valid.astype(bool) & np.isfinite(r)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    disp = np.clip(r[m], 0.0, 1.0)
    disp = np.power(disp, 1.0 / float(REFLECT_SHOW_GAMMA))
    u8[m] = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _make_range_snr_mask(depth_m: np.ndarray, snr: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    s = np.asarray(snr, dtype=np.float32)
    valid = np.isfinite(d) & np.isfinite(s) & (d > 0.0)
    thr = np.full(d.shape, float(SNR_GATE_GT8M), dtype=np.float32)
    thr = np.where(d <= 8.0, float(SNR_GATE_5TO8M), thr)
    thr = np.where(d <= 5.0, float(SNR_GATE_3TO5M), thr)
    thr = np.where(d <= 3.0, float(SNR_GATE_LE3M), thr)
    return valid & (s > thr)


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode():
        inp = torch.from_numpy(hists).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        out = net.forward_train(inp)
        logits_t = out["bin_logits"]
        probs_t = torch.softmax(logits_t, dim=1)
        dist_t = out["dist"]
        pred_depth = dist_t[:, 0, :, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        out_probs = probs_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)

    invalid = (~np.isfinite(pred_depth)) | (pred_depth <= 0.0)
    if np.any(invalid):
        pred_depth = pred_depth.copy()
        out_probs = out_probs.copy()
        pred_depth[invalid] = 0.0
        out_probs[invalid, :] = 0.0
    return pred_depth, out_probs


def _load_scene_points(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None:
    if not npz_path.exists():
        return None
    d = np.load(str(npz_path))
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"], dtype=np.float32)
    z = np.asarray(d["z"], dtype=np.float32)
    meta = {
        "capture_seconds": float(d["capture_seconds"]) if "capture_seconds" in d else 0.0,
        "saved_unix_ts": float(d["saved_unix_ts"]) if "saved_unix_ts" in d else 0.0,
    }
    return x, y, z, meta


def _find_points_npz(scene_dir: Path) -> Path | None:
    files = sorted(scene_dir.glob("points_last*.npz"))
    return files[0] if files else None


def _render_lidar_view(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    if x.size == 0:
        return np.zeros((LIDAR_H, LIDAR_W, 3), dtype=np.uint8)

    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    m = (x > 0.0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    m &= (np.abs(yaw) <= LIDAR_HALF_FOV) & (np.abs(pitch) <= LIDAR_HALF_FOV)
    if not np.any(m):
        return np.zeros((LIDAR_H, LIDAR_W, 3), dtype=np.uint8)

    xv = x[m]
    yv = y[m]
    zv = z[m]
    yaw = np.arctan2(yv, xv)
    pitch = np.arctan2(zv, xv)
    depth_m = np.clip(xv, LIDAR_NEAR_SAT_M, LIDAR_MAX_RANGE_M)
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0.0, 255.0).astype(np.uint8)

    col = ((LIDAR_HALF_FOV - yaw) / (2.0 * LIDAR_HALF_FOV) * (LIDAR_W - 1)).astype(np.int32)
    row = ((LIDAR_HALF_FOV - pitch) / (2.0 * LIDAR_HALF_FOV) * (LIDAR_H - 1)).astype(np.int32)
    col = np.clip(col, 0, LIDAR_W - 1)
    row = np.clip(row, 0, LIDAR_H - 1)

    img = np.zeros((LIDAR_H, LIDAR_W), dtype=np.uint8)
    np.maximum.at(img, (row, col), depth_u8)
    bgr = cv2.applyColorMap(img, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    bgr[img == 0] = (0, 0, 0)
    return bgr


def _scene_stats_text(pred_depth: np.ndarray, snr: np.ndarray, conf_mask: np.ndarray, points: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]] | None) -> list[str]:
    valid_depth = pred_depth[np.isfinite(pred_depth) & (pred_depth > 0.0)]
    valid_snr = snr[np.isfinite(snr)]
    kept_ratio = float(np.mean(conf_mask.astype(np.float32))) if conf_mask.size else 0.0
    lines = [
        f"scene=20260318_191533  depth_valid={int(valid_depth.size)}/{pred_depth.size}  snr_kept={kept_ratio * 100.0:.1f}%",
        (
            "depth(m): "
            f"min={float(np.min(valid_depth)):.3f}  "
            f"mean={float(np.mean(valid_depth)):.3f}  "
            f"max={float(np.max(valid_depth)):.3f}"
            if valid_depth.size
            else "depth(m): no valid prediction"
        ),
        (
            "snr: "
            f"min={float(np.min(valid_snr)):.3f}  "
            f"mean={float(np.mean(valid_snr)):.3f}  "
            f"max={float(np.max(valid_snr)):.3f}"
            if valid_snr.size
            else "snr: no valid value"
        ),
    ]
    if points is not None:
        x, y, z, meta = points
        lines.append(
            "lidar: "
            f"pts={int(x.size)}  "
            f"last={meta['capture_seconds']:.1f}s  "
            f"x=[{float(np.min(x)):.2f},{float(np.max(x)):.2f}]  "
            f"y=[{float(np.min(y)):.2f},{float(np.max(y)):.2f}]  "
            f"z=[{float(np.min(z)):.2f},{float(np.max(z)):.2f}]"
        )
    else:
        lines.append("lidar: points_last2.0s.npz missing")
    return lines


def _scene_compare_text(
    pred_depth: np.ndarray,
    reflectance: np.ndarray,
    reflect_valid_mask: np.ndarray,
    reflect_black_mask: np.ndarray,
    conf_mask: np.ndarray,
    conf_mask_after_reflect: np.ndarray,
) -> list[str]:
    valid_reflect = reflectance[reflect_valid_mask & np.isfinite(reflectance)]
    before_count = int(np.sum(conf_mask))
    after_count = int(np.sum(conf_mask_after_reflect))
    removed_count = int(np.sum(conf_mask & reflect_black_mask))
    total_count = int(conf_mask.size)
    lines = [
        f"compare: before={before_count}/{total_count}  after={after_count}/{total_count}  reflect_removed={removed_count}",
        (
            f"reflectance: valid={int(np.sum(reflect_valid_mask))}/{total_count}  "
            f"black<{REFLECT_THRESH * 100.0:.1f}%={int(np.sum(reflect_black_mask))}"
        ),
    ]
    if valid_reflect.size:
        lines.append(
            "reflectance stats(%): "
            f"min={float(np.min(valid_reflect) * 100.0):.3f}  "
            f"mean={float(np.mean(valid_reflect) * 100.0):.3f}  "
            f"max={float(np.max(valid_reflect) * 100.0):.3f}"
        )
    else:
        lines.append("reflectance stats(%): no valid reflectance")
    return lines


def _discover_scene_dirs(scene_root: Path) -> list[Path]:
    if not scene_root.exists():
        return []

    scene_dirs = {tof_path.parent for tof_path in scene_root.rglob("tof.raw") if tof_path.is_file()}
    return sorted(scene_dirs)


def _build_scene_state(scene_dir: Path, net, device) -> SceneState:
    from tof3d import tof_histograms_from_u16  # noqa: E402

    tof_path = scene_dir / "tof.raw"
    if not tof_path.exists():
        raise FileNotFoundError(f"missing scene tof.raw: {tof_path}")

    raw_u16 = np.fromfile(str(tof_path), dtype=np.uint16)
    hists = tof_histograms_from_u16(raw_u16)
    if hists.shape != (TOF_H, TOF_W, TOF_C):
        raise RuntimeError(f"unexpected histogram shape: {hists.shape} @ {scene_dir}")

    pred_depth, out_probs = _run_infer(net, device, hists)
    snr = _compute_snr_from_input(hists)
    conf_mask = _make_range_snr_mask(pred_depth, snr)
    reflect_peak, reflectance, reflect_black_mask = _compute_reflectance(hists, pred_depth)
    reflect_valid_mask = np.isfinite(reflectance)
    conf_mask_after_reflect = conf_mask & (~reflect_black_mask)

    points_path = _find_points_npz(scene_dir)
    points = _load_scene_points(points_path) if points_path is not None else None
    if points is not None:
        lidar_view = _render_lidar_view(points[0], points[1], points[2])
        lidar_view = _with_text(lidar_view, "LIDAR last 2.0s", y=28)
    else:
        lidar_view = None

    return SceneState(
        scene_dir=scene_dir,
        scene_name=scene_dir.name,
        hists=hists,
        pred_depth=pred_depth,
        out_probs=out_probs,
        snr=snr,
        conf_mask=conf_mask,
        reflect_peak=reflect_peak,
        reflectance=reflectance,
        reflect_valid_mask=reflect_valid_mask,
        reflect_black_mask=reflect_black_mask,
        conf_mask_after_reflect=conf_mask_after_reflect,
        points=points,
        lidar_view=lidar_view,
    )


def _print_scene_stats(state: SceneState) -> None:
    for line in _scene_stats_text(state.pred_depth, state.snr, state.conf_mask, state.points):
        if line.startswith("scene="):
            print(line.replace("scene=20260318_191533", f"scene={state.scene_name}", 1))
        else:
            print(line)
    for line in _scene_compare_text(
        state.pred_depth,
        state.reflectance,
        state.reflect_valid_mask,
        state.reflect_black_mask,
        state.conf_mask,
        state.conf_mask_after_reflect,
    ):
        print(line)


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("missing dependency torch") from e

    script_dir = Path(__file__).resolve().parent
    nn_dir = script_dir.parent if script_dir.name == "xuanguang" else script_dir
    scene_root = script_dir if script_dir.name == "xuanguang" else (nn_dir / "xuanguang")
    root = nn_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))

    from net import Network  # noqa: E402
    ckpt_path = nn_dir / "model_last.pt"
    scene_dirs = _discover_scene_dirs(scene_root)
    if not scene_dirs:
        raise FileNotFoundError(f"no scene with tof.raw found under: {scene_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()

    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        try:
            net.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed, fallback strict=False: {e}")
            net.load_state_dict(sd, strict=False)
        print(f"[load] {ckpt_path}")
    else:
        print(f"[warn] missing checkpoint: {ckpt_path} (use random weights)")

    scene_idx = 0
    state = _build_scene_state(scene_dirs[scene_idx], net, device)
    _print_scene_stats(state)

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_AUTOSIZE)

    mouse = {"x": 0, "y": 0}
    picked = {"px": 0, "py": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback(MAIN_WINDOW, on_mouse)

    def build_views(cur: SceneState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_before_bgr = _colorize_depth(cur.pred_depth)
        pred_before_bgr = pred_before_bgr.copy()
        pred_before_bgr[~cur.conf_mask] = (0, 0, 0)
        pred_before_big_base = cv2.resize(
            cv2.flip(cv2.rotate(pred_before_bgr, cv2.ROTATE_90_CLOCKWISE), 1),
            (SHOW_W, SHOW_H),
            interpolation=cv2.INTER_NEAREST,
        )

        pred_after_bgr = _colorize_depth(cur.pred_depth)
        pred_after_bgr = pred_after_bgr.copy()
        pred_after_bgr[~cur.conf_mask_after_reflect] = (0, 0, 0)
        pred_after_big_base = cv2.resize(
            cv2.flip(cv2.rotate(pred_after_bgr, cv2.ROTATE_90_CLOCKWISE), 1),
            (SHOW_W, SHOW_H),
            interpolation=cv2.INTER_NEAREST,
        )

        reflect_bgr = _colorize_reflectance(cur.reflectance, cur.reflect_valid_mask)
        reflect_big_base = cv2.resize(
            cv2.flip(cv2.rotate(reflect_bgr, cv2.ROTATE_90_CLOCKWISE), 1),
            (SHOW_W, SHOW_H),
            interpolation=cv2.INTER_NEAREST,
        )
        return pred_before_big_base, pred_after_big_base, reflect_big_base

    pred_before_big_base, pred_after_big_base, reflect_big_base = build_views(state)

    while True:
        top_row_w = SHOW_W * 3
        mx = int(np.clip(mouse.get("x", 0), 0, top_row_w - 1))
        my_view = int(mouse.get("y", 0)) - int(HEADER_H)
        if 0 <= my_view < SHOW_H and mx < top_row_w:
            my = int(my_view)
            if mx < SHOW_W:
                tile_x0 = 0
            elif mx < SHOW_W * 2:
                tile_x0 = SHOW_W
            else:
                tile_x0 = SHOW_W * 2
            px, py = _disp_xy_to_pixel(mx - tile_x0, my, SHOW_W, SHOW_H)
            picked["px"] = int(px)
            picked["py"] = int(py)
        px = int(np.clip(picked["px"], 0, TOF_W - 1))
        py = int(np.clip(picked["py"], 0, TOF_H - 1))

        pr_v = float(state.pred_depth[py, px])
        snr_v = float(state.snr[py, px])
        peak_v = float(state.reflect_peak[py, px])
        hbins_hover = state.hists[py, px, :]
        bin_63_v = float(hbins_hover[HIST_BINS]) if hbins_hover.shape[0] > HIST_BINS else 0.0
        bin_64_v = float(hbins_hover[HIST_BINS + 1]) if hbins_hover.shape[0] > (HIST_BINS + 1) else 0.0
        sat_v = float(bin_64_v * float(REFLECT_SAT_VALUE) + bin_63_v)
        refl_v = float(state.reflectance[py, px]) if bool(state.reflect_valid_mask[py, px]) else float("nan")
        keep_before_v = bool(state.conf_mask[py, px])
        keep_after_v = bool(state.conf_mask_after_reflect[py, px])
        refl_black_v = bool(state.reflect_black_mask[py, px])
        if pr_v <= 0.0 or (not np.isfinite(pr_v)):
            hover_txt = (
                f"pred --  snr {snr_v:.2f}  peak {peak_v:.1f}  sat {sat_v:.1f}  refl --  "
                f"keep {int(keep_before_v)}->{int(keep_after_v)}  black={int(refl_black_v)}  px=({px},{py})"
            )
        else:
            refl_txt = f"{refl_v * 100.0:.3f}%" if np.isfinite(refl_v) else "--"
            hover_txt = (
                f"pred {pr_v:.3f}m  snr {snr_v:.2f}  peak {peak_v:.1f}  sat {sat_v:.1f}  refl {refl_txt}  "
                f"keep {int(keep_before_v)}->{int(keep_after_v)}  black={int(refl_black_v)}  px=({px},{py})"
            )

        pred_before_big = _with_text(pred_before_big_base, "DEPTH_OLD (range-SNR gated)")
        pred_after_big = _with_text(pred_after_big_base, f"DEPTH_NEW (refl>={REFLECT_THRESH * 100.0:.1f}% keep)")
        reflect_big = _with_text(reflect_big_base, f"REFLECTANCE (0..1, gamma {REFLECT_SHOW_GAMMA:.1f})")

        dx_m, dy_m = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)
        pred_before_big = _draw_marker(pred_before_big, dx_m, dy_m)
        pred_after_big = _draw_marker(pred_after_big, dx_m, dy_m)
        reflect_big = _draw_marker(reflect_big, dx_m, dy_m)
        top_row = np.hstack([pred_before_big, pred_after_big, reflect_big])

        hbins = state.hists[py, px, :]
        hist_img = _render_histogram_bgr(hbins, w=HIST_W, h=HIST_H, max_bins=HIST_BINS, title="HIST")
        hsrc = np.asarray(hbins[:HIST_BINS], dtype=np.float32)
        if hsrc.size:
            h_peak_idx = int(np.argmax(hsrc))
            h_max_raw = float(hsrc[h_peak_idx])
            h_bin_63 = float(hbins[HIST_BINS]) if hbins.shape[0] > HIST_BINS else 0.0
            h_bin_64 = float(hbins[HIST_BINS + 1]) if hbins.shape[0] > (HIST_BINS + 1) else 0.0
            h_sat = float(h_bin_64 * float(REFLECT_SAT_VALUE) + h_bin_63)
            h_max = float(_adjust_reflect_peak(h_max_raw, h_bin_63, h_bin_64))
            h_mean = float(np.mean(hsrc, dtype=np.float32))
            h_std = float(np.std(hsrc, dtype=np.float32))
            h_snr = float((h_max - h_mean) / max(h_std, 1e-6))
            h_refl = float(h_max * pr_v * pr_v / float(REFLECT_DENOM)) if np.isfinite(pr_v) and pr_v > 0.0 else float("nan")
        else:
            h_peak_idx = 0
            h_max = 0.0
            h_mean = 0.0
            h_std = 0.0
            h_snr = 0.0
            h_refl = float("nan")
            h_sat = 0.0
        cv2.putText(hist_img, "snr = (max - mean) / std, using bins[0..61]", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(hist_img, f"max[{h_peak_idx}]={h_max:.3f}  mean={h_mean:.3f}  std={h_std:.3f}  snr={h_snr:.4f}", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(hist_img, f"sat=bin64*1023+bin63 = {h_sat:.3f}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        refl_hist_txt = (
            f"reflect={h_refl * 100.0:.4f}%"
            if np.isfinite(h_refl)
            else "reflect=-- (pred invalid)"
        )
        cv2.putText(hist_img, refl_hist_txt, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        bottom_row = np.zeros((HIST_H, top_row.shape[1], 3), dtype=np.uint8)
        hist_x0 = max((top_row.shape[1] - HIST_W) // 2, 0)
        bottom_row[:, hist_x0 : hist_x0 + HIST_W] = hist_img
        view = np.vstack([top_row, bottom_row])
        header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
        header = _with_text(
            header,
            f"scene={state.scene_name}  [{scene_idx + 1}/{len(scene_dirs)}]  key:4 prev, 6 next, q/esc quit",
            y=22,
        )
        header = _with_text(header, hover_txt, y=46)
        view = np.vstack([header, view])
        cv2.imshow(MAIN_WINDOW, view)

        k = int(cv2.waitKey(16) & 0xFF)
        if k in (27, ord("q"), ord("Q")):
            break
        if k == ord("4") and scene_idx > 0:
            scene_idx -= 1
            state = _build_scene_state(scene_dirs[scene_idx], net, device)
            _print_scene_stats(state)
            pred_before_big_base, pred_after_big_base, reflect_big_base = build_views(state)
        elif k == ord("6") and scene_idx < len(scene_dirs) - 1:
            scene_idx += 1
            state = _build_scene_state(scene_dirs[scene_idx], net, device)
            _print_scene_stats(state)
            pred_before_big_base, pred_after_big_base, reflect_big_base = build_views(state)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
