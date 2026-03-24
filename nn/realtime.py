#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/realtime.py

实时读取 tof.raw（通过 tof_server.py 的 ToFRealtimeServer），
运行深度学习模型并实时显示 3 张图：
- INPUT: ToF 强度（直方图求和）
- PRED: 预测深度（伪彩）
- PROB: 预测置信度图（conf 分支输出）
- HIST: 鼠标悬停点的输入直方图（实时刷新）
- OUT_HIST: 鼠标悬停点的输出 bin 概率直方图（实时刷新）

交互：
- 鼠标悬停：显示 pred/bin_range/prob
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

# 与 check.py 对齐：显示方向使用 rot90CW + flipH（等价转置），单图按 3:4 竖屏显示
SHOW_W = 390
SHOW_H = 520
HEADER_H = 32
HIST_BINS = 62
HIST_W = 620
HIST_H = 260

EPS = 1e-6
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


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标（显示做了 rot90CW + flipH）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    # 与 check.py 一致：该组合等价于转置，display x -> 原始 py，display y -> 原始 px
    py = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
    px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """ToF 像素坐标 -> 显示坐标（显示做了 rot90CW + flipH）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    dx = int(np.clip((py_i + 0.5) * sw / TOF_H, 0, sw - 1))
    dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    return dx, dy


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    """在图上画一个小圆点（黑边白心），用于标记 hover 像素。"""
    import cv2  # type: ignore

    out = img_bgr.copy()
    xx = int(np.clip(x, 0, out.shape[1] - 1))
    yy = int(np.clip(y, 0, out.shape[0] - 1))
    cv2.circle(out, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(out, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _with_text(img_bgr: np.ndarray, text: str) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _render_histogram_bgr(
    bins: np.ndarray,
    w: int = HIST_W,
    h: int = HIST_H,
    max_bins: int | None = None,
    title: str = "HIST",
    fixed_vmax: float | None = None,
) -> np.ndarray:
    """把一维 bins 渲染成柱状直方图（BGR uint8）。"""
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
        xL = x0 + i * bar_w
        xR = min(xL + bar_w, x1)
        if xR <= xL:
            continue
        yT = y1 - hh
        cv2.rectangle(img, (xL, yT), (xR, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (xL, yT), (xR, y1), (30, 30, 30), 1)

    step = 10
    for k in range(0, nb, step):
        xx = x0 + int(k * bar_w)
        cv2.line(img, (xx, y1), (xx, y1 + 4), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (xx + 2, sh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    ssum = float(np.sum(b))
    img = _with_text(img, f"{title} bins[0..{nb-1}]  max {vmax:.3f}  sum {ssum:.3f}")
    return img


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


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hist (H,W,64) -> pred_depth, out_probs."""
    import torch

    with torch.inference_mode():
        inp = torch.from_numpy(hists).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        out = net.forward_train(inp)
        logits_t = out["bin_logits"]  # (1,64,H,W)
        probs_t = torch.softmax(logits_t, dim=1)
        dist_t = out["dist"]  # (1,1,H,W)
        pred_depth = dist_t[:, 0, :, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        out_probs = probs_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)

    invalid = (~np.isfinite(pred_depth)) | (pred_depth <= 0.0)
    if np.any(invalid):
        pred_depth = pred_depth.copy()
        out_probs = out_probs.copy()
        pred_depth[invalid] = 0.0
        out_probs[invalid, :] = 0.0

    return pred_depth, out_probs


def main() -> int:
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

    cv2.namedWindow("NN_REALTIME", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("OUT_HIST", cv2.WINDOW_AUTOSIZE)

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
    cached_out_probs: np.ndarray | None = None

    try:
        while True:
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                raw_u16 = np.frombuffer(frame.raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16)
                if hists.shape == (TOF_H, TOF_W, TOF_C):
                    pred_depth, out_probs = _run_infer(net, device, hists)
                    snr = _compute_snr_from_input(hists)
                    cached_in = hists
                    cached_pred_depth = pred_depth
                    cached_snr = snr
                    cached_out_probs = out_probs
                    last_ts = float(frame.ts)

            if cached_in is None or cached_pred_depth is None or cached_snr is None or cached_out_probs is None:
                k = int(cv2.waitKey(5) & 0xFF)
                if k == 27:
                    break
                continue

            # INPUT intensity
            inten_u8 = _render_input_intensity_u8(cached_in)
            in_u8 = cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE)
            in_u8 = cv2.flip(in_u8, 1)
            in_big = cv2.resize(in_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

            # PRED depth
            pred_bgr = _colorize_depth(cached_pred_depth)
            conf_mask = _make_range_snr_mask(cached_pred_depth, cached_snr)
            if np.any(~conf_mask):
                pred_bgr = pred_bgr.copy()
                pred_bgr[~conf_mask] = (0, 0, 0)
            pred_bgr = cv2.flip(cv2.rotate(pred_bgr, cv2.ROTATE_90_CLOCKWISE), 1)
            pred_big = cv2.resize(pred_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

            # PROB
            valid = np.ones((TOF_H, TOF_W), dtype=bool)
            prob_bgr = _colorize_prob(cached_snr, valid)
            prob_bgr = cv2.flip(cv2.rotate(prob_bgr, cv2.ROTATE_90_CLOCKWISE), 1)
            prob_big = cv2.resize(prob_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

            # hover info（单行）
            mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 3 - 1))
            my_view = int(mouse.get("y", 0)) - int(HEADER_H)
            my = int(np.clip(my_view, 0, SHOW_H - 1))
            tile_x0 = 0 if mx < SHOW_W else (SHOW_W if mx < SHOW_W * 2 else SHOW_W * 2)
            px, py = _disp_xy_to_pixel(mx - tile_x0, my, SHOW_W, SHOW_H)

            pr_v = float(cached_pred_depth[py, px])
            pb_v = float(cached_snr[py, px])
            if pr_v <= 0.0 or (not np.isfinite(pr_v)):
                hover_txt = f"pred --  snr {pb_v:.2f}"
            else:
                hover_txt = f"pred {pr_v:.3f}m  snr {pb_v:.2f}"

            # 标题文字
            in_bgr = _with_text(in_bgr, "INPUT")
            pred_big = _with_text(pred_big, "PRED (<=3m>5.5, 3-5m>5, 5-8m>4.5, 8m+>4)")
            prob_big = _with_text(prob_big, f"SNR(white@{SNR_SHOW_MAX:g})")

            # 画 hover 点
            dx_m, dy_m = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)
            in_bgr = _draw_marker(in_bgr, dx_m, dy_m)
            pred_big = _draw_marker(pred_big, dx_m, dy_m)
            prob_big = _draw_marker(prob_big, dx_m, dy_m)

            view = np.hstack([in_bgr, pred_big, prob_big])
            header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
            header = _with_text(header, hover_txt)
            view = np.vstack([header, view])

            cv2.imshow("NN_REALTIME", view)

            # hovered point histogram（输入前 62 个 bin）
            hbins = cached_in[py, px, :]
            hist_img = _render_histogram_bgr(hbins, w=HIST_W, h=HIST_H, max_bins=HIST_BINS, title="IN_HIST")
            hsrc = np.asarray(hbins[:HIST_BINS], dtype=np.float32)
            if hsrc.size:
                h_peak_idx = int(np.argmax(hsrc))
                h_max = float(hsrc[h_peak_idx])
                h_mean = float(np.mean(hsrc, dtype=np.float32))
                h_std = float(np.std(hsrc, dtype=np.float32))
                h_snr = float((h_max - h_mean) / max(h_std, 1e-6))
            else:
                h_peak_idx = 0
                h_max = 0.0
                h_mean = 0.0
                h_std = 0.0
                h_snr = 0.0
            cv2.putText(
                hist_img,
                "snr = (max - mean) / std, using bins[0..61]",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                hist_img,
                f"max[{h_peak_idx}]={h_max:.3f}  mean={h_mean:.3f}  std={h_std:.3f}  snr={h_snr:.4f}",
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow("HIST", hist_img)

            # hovered point output histogram（网络输出 NUM_BINS 个概率）
            obins = cached_out_probs[py, px, :]
            out_hist_img = _render_histogram_bgr(
                obins, w=HIST_W, h=HIST_H, max_bins=NUM_BINS, title="OUT_HIST", fixed_vmax=1.0
            )
            cv2.imshow("OUT_HIST", out_hist_img)

            k = int(cv2.waitKey(1) & 0xFF)
            if k == 27:
                break
    finally:
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

