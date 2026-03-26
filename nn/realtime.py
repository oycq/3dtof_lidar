#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/realtime.py

实时读取 tof.raw（通过 tof_server.py 的 ToFRealtimeServer），
运行模型并实时显示 3 张图：
- INPUT: ToF 强度（直方图求和）
- PRED: 预测深度（伪彩）
- PROB: SNR 灰度图
- HIST: 鼠标悬停点的输入直方图（实时刷新）

交互：
- 鼠标悬停：显示 pred/bin_range/snr/conf/reflectance
- PRED 使用按 pred 距离分段 SNR 卡控：
  <=3m: snr>5.5, 3~5m: snr>5, 5~8m: snr>4.5, 8m+: snr>4
- ESC 退出
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

TOF_H = 30
TOF_W = 40
TOF_C = 64

# 输出分类配置（与 train.py / net.py 对齐）
NUM_BINS = 64
VALID_BINS = 63
INVALID_BIN = 63
MAX_VALID_M = 35.0
LOG_BASE = 1.06

# 单图显示长短边；是否旋转时会自动交换宽高
SHOW_LONG = 520
SHOW_SHORT = 390
ROTATE_90 = False
HEADER_H = 56
HIST_BINS = 62
HIST_W = 640
HIST_H = 280

EPS = 1e-6
TAIL_BASE = 1024.0
PULSES = 50000.0
DEPTH_NEAR_M = 0.8
DEPTH_FAR_M = 35.0
DEPTH_MAP_CLIP_MIN = 1.5
DEPTH_MAP_SCALE = 1.5
DISP_GAMMA = 1.2
SUM_GATE_MAX = 20000.0
SUM_GATE_SNR_DIV = 3.0
PEAK_GATE_MIN = 30.0
SNR_GATE_LE3M = 5.5      # <=3m
SNR_GATE_3TO5M = 5.0     # (3,5]m
SNR_GATE_5TO8M = 4.5     # (5,8]m
SNR_GATE_GT8M = 4.0      # >8m
SNR_SHOW_MAX = 10.0


def _get_show_size(rotate_90: bool) -> Tuple[int, int]:
    """返回单图显示尺寸；旋转后同步切换宽高比例。"""
    if rotate_90:
        return SHOW_SHORT, SHOW_LONG
    return SHOW_LONG, SHOW_SHORT


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int, rotate_90: bool) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    if rotate_90:
        # 与项目里旧版四宫格一致：rot90CW + flipH，等价于转置。
        py = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
        px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    else:
        px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
        py = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int, rotate_90: bool) -> Tuple[int, int]:
    """ToF 像素坐标 -> 显示坐标。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    if rotate_90:
        dx = int(np.clip((py_i + 0.5) * sw / TOF_H, 0, sw - 1))
        dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    else:
        dx = int(np.clip((px_i + 0.5) * sw / TOF_W, 0, sw - 1))
        dy = int(np.clip((py_i + 0.5) * sh / TOF_H, 0, sh - 1))
    return dx, dy


def _orient_for_display(img: np.ndarray, rotate_90: bool) -> np.ndarray:
    """按显示选项变换方向；旋转模式与旧版四宫格保持一致。"""
    if not rotate_90:
        return img

    import cv2  # type: ignore

    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    """在图上画一个小圆点（黑边白心），用于标记 hover 像素。"""
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
    cv2.putText(out, text, (10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_histogram_bgr(
    bins: np.ndarray,
    w: int = HIST_W,
    h: int = HIST_H,
) -> np.ndarray:
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
    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return None
    inv_v = 1.0 / np.clip(d[valid], EPS, np.inf)
    vmin = float(np.min(inv_v))
    vmax = float(np.max(inv_v))
    return (vmin, vmax if vmax > vmin else vmin + 1e-6)


def _colorize_depth_with_range(depth_m: np.ndarray, inv_vmin: float, inv_vmax: float) -> np.ndarray:
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
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


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """(H,W,64) -> (H,W) uint8 intensity (简单按 max 归一化)."""
    inten = np.sum(hists.astype(np.float32, copy=False), axis=2)
    vmax = float(np.max(inten)) if inten.size else 0.0
    if vmax <= 0.0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    return np.clip(np.rint(inten / vmax * 255.0), 0, 255).astype(np.uint8)


def _compute_snr_from_input(hists: np.ndarray) -> np.ndarray:
    """按输入前 62 个 bin 计算 SNR 图：snr=(max-mean)/std。"""
    h = np.asarray(hists, dtype=np.float32)
    if h.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad hists shape: {h.shape}")

    src = h[:, :, :HIST_BINS]
    vmax = np.max(src, axis=2)
    vsum = np.sum(src, axis=2, dtype=np.float32)
    mean = np.mean(src, axis=2, dtype=np.float32)
    std = np.std(src, axis=2, dtype=np.float32)
    snr = (vmax - mean) / np.maximum(std, 1e-6)

    div = float(max(SUM_GATE_SNR_DIV, 1.0))
    snr = np.where(vsum > float(SUM_GATE_MAX), snr / div, snr)
    snr = np.where(vmax < float(PEAK_GATE_MIN), 0.0, snr)
    return snr.astype(np.float32, copy=False)


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H,W) depth(m) -> BGR (JET), 映射 y=1.5/clip(x,1.5,+inf)。"""
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    clip_min = float(max(DEPTH_MAP_CLIP_MIN, EPS))
    scale = float(DEPTH_MAP_SCALE)
    far_m = float(max(DEPTH_FAR_M, clip_min))
    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    d_safe = np.maximum(d[valid], EPS)
    y = scale / d_safe
    y_min = scale / far_m
    y_max = scale / clip_min
    norm = (y - y_min) / max(y_max - y_min, EPS)
    u8[valid] = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """SNR 标量图 -> 灰度 BGR（gamma=DISP_GAMMA），invalid 为黑。"""
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)

    gamma = float(DISP_GAMMA)
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m] / float(SNR_SHOW_MAX), 0.0, 1.0), 1.0 / gamma)

    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _make_range_snr_mask(depth_m: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """按 pred 距离段应用 SNR 阈值：

    - <=3m:   snr > 5.5
    - 3~5m:   snr > 5
    - 5~8m:   snr > 4.5
    - >8m:    snr > 4
    """
    d = np.asarray(depth_m, dtype=np.float32)
    s = np.asarray(snr, dtype=np.float32)
    valid = np.isfinite(d) & np.isfinite(s) & (d > 0.0)
    thr = np.full(d.shape, float(SNR_GATE_GT8M), dtype=np.float32)
    thr = np.where(d <= 8.0, float(SNR_GATE_5TO8M), thr)
    thr = np.where(d <= 5.0, float(SNR_GATE_3TO5M), thr)
    thr = np.where(d <= 3.0, float(SNR_GATE_LE3M), thr)
    return valid & (s > thr)


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hist (H,W,64) -> pred_depth, snr, reflectance, conf."""
    import torch

    h = np.array(hists, dtype=np.float32, copy=True)
    with torch.inference_mode():
        inp = torch.from_numpy(h).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32)
        dist_t, snr_t, reflectance_t, conf_t = net(inp)
        pred_depth = dist_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        snr = snr_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        reflectance = reflectance_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        conf = conf_t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    invalid = (~np.isfinite(pred_depth)) | (pred_depth <= 0.0)
    if np.any(invalid):
        pred_depth = pred_depth.copy()
        snr = snr.copy()
        reflectance = reflectance.copy()
        conf = conf.copy()
        pred_depth[invalid] = 0.0
        snr[invalid] = 0.0
        reflectance[invalid] = 0.0
        conf[invalid] = 0.0

    return pred_depth, snr, reflectance, conf


def main() -> int:
    rotate_90 = bool(ROTATE_90)
    show_w, show_h = _get_show_size(rotate_90)

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("missing dependency torch") from e

    nn_dir = Path(__file__).resolve().parent
    root = nn_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))

    from net import Network  # noqa: E402
    from tof3d import tof_histograms_from_u16  # noqa: E402
    from tof_server import ToFRealtimeServer  # noqa: E402

    ckpt_path = nn_dir / "model_last.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
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

    cv2.namedWindow("NN_REALTIME", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback("NN_REALTIME", on_mouse)

    tof_srv = ToFRealtimeServer(queue_maxlen=5, min_peak_count=100.0, target_fps=10.0)
    tof_srv.start()

    last_ts = 0.0
    cached_in: np.ndarray | None = None
    cached_pred_depth: np.ndarray | None = None
    cached_snr: np.ndarray | None = None
    cached_reflectance: np.ndarray | None = None
    cached_conf: np.ndarray | None = None

    try:
        while True:
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                raw_u16 = np.frombuffer(frame.raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16)
                if hists.shape == (TOF_H, TOF_W, TOF_C):
                    pred_depth, snr, reflectance, conf = _run_infer(net, device, hists)
                    cached_in = hists
                    cached_pred_depth = pred_depth
                    cached_snr = snr
                    cached_reflectance = reflectance
                    cached_conf = conf
                    last_ts = float(frame.ts)

            if cached_in is None or cached_pred_depth is None or cached_snr is None or cached_reflectance is None or cached_conf is None:
                k = int(cv2.waitKey(5) & 0xFF)
                if k == 27:
                    break
                continue

            refl_u8 = np.clip(np.rint(np.clip(cached_reflectance, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
            refl_bgr = cv2.cvtColor(
                cv2.resize(_orient_for_display(refl_u8, rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )
            input_bgr = cv2.cvtColor(
                cv2.resize(_orient_for_display(_render_input_intensity_u8(cached_in), rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR,
            )

            pred_for_disp = np.where(cached_conf > 0.5, cached_pred_depth, 0.0)
            inv_range = _inv_depth_range_from_depth(cached_pred_depth)
            if inv_range is None:
                pred_src = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
            else:
                pred_src = _colorize_depth_with_range(pred_for_disp, inv_range[0], inv_range[1])
            pred_big = cv2.resize(_orient_for_display(pred_src, rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST)
            conf_big = cv2.resize(_orient_for_display(_colorize_gray01(cached_conf), rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST)

            mx = int(np.clip(mouse.get("x", 0), 0, show_w * 2 - 1))
            my = int(np.clip(int(mouse.get("y", 0)) - int(HEADER_H), 0, show_h * 2 - 1))
            tile_x0 = 0 if mx < show_w else show_w
            tile_y0 = 0 if my < show_h else show_h
            px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, show_w, show_h, rotate_90)
            dx_m, dy_m = _pixel_to_disp_xy(px, py, show_w, show_h, rotate_90)

            for img in [refl_bgr, input_bgr, pred_big, conf_big]:
                marked = _draw_marker(img, dx_m, dy_m)
                img[:] = marked

            hover1 = f"pred {float(cached_pred_depth[py, px]):.3f}m  conf {float(cached_conf[py, px]):.0f}  snr {float(cached_snr[py, px]):.3f}"
            hover2 = f"reflectance {float(cached_reflectance[py, px]) * 100.0:.3f}%"

            refl_bgr = _with_text(refl_bgr, "REFLECTANCE")
            input_bgr = _with_text(input_bgr, "INPUT")
            pred_big = _with_text(pred_big, "PRED (conf==1)")
            conf_big = _with_text(conf_big, "CONF")
            view = np.vstack([
                np.zeros((HEADER_H, show_w * 2, 3), dtype=np.uint8),
                np.hstack([refl_bgr, input_bgr]),
                np.hstack([pred_big, conf_big]),
            ])
            view[:HEADER_H] = _with_text(view[:HEADER_H], hover1, y=22)
            view[:HEADER_H] = _with_text(view[:HEADER_H], hover2, y=46)

            cv2.imshow("NN_REALTIME", view)

            hbins = cached_in[py, px, :]
            cv2.imshow("HIST", _render_histogram_bgr(hbins))

            k = int(cv2.waitKey(1) & 0xFF)
            if k == 27:
                break
    finally:
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

