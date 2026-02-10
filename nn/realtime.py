#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/realtime.py

实时读取 tof.raw（通过 tof_server.py 的 ToFRealtimeServer），
运行深度学习模型并实时显示 3 张图：
- INPUT: ToF 强度（直方图求和）
- PRED: 预测深度（伪彩）
- PROB: 预测置信度图（top1 区间概率）

交互：
- 鼠标悬停：显示 pred/bin_range/prob
- pThr% 滑动条：低于阈值的像素直接置黑（PRED 视图）
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
BIN_M = 0.15 * 4
MIN_RANGE_M = BIN_M
MAX_RANGE_M = float(NUM_BINS) * float(BIN_M)

SHOW_W = 400
SHOW_H = 300
HEADER_H = 32

EPS = 1e-6
DEPTH_NEAR_M = 0.5
DEPTH_FAR_M = 10.0


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标（显示做了 flipV，因此这里需要还原）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
    py_disp = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    py = (TOF_H - 1) - py_disp
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """ToF 像素坐标 -> 显示坐标（显示做了 flipV）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    py_disp = (TOF_H - 1) - py_i
    dx = int(np.clip((px_i + 0.5) * sw / TOF_W, 0, sw - 1))
    dy = int(np.clip((py_disp + 0.5) * sh / TOF_H, 0, sh - 1))
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


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """(H,W,64) -> (H,W) uint8 intensity (简单按 max 归一化)."""
    inten = np.sum(hists.astype(np.float32, copy=False), axis=2)
    vmax = float(np.max(inten)) if inten.size else 0.0
    if vmax <= 0.0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    return np.clip(np.rint(inten / vmax * 255.0), 0, 255).astype(np.uint8)


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H,W) depth(m) -> BGR (0.5m red, 6m blue), 0 为黑。"""
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    d_clamped = np.clip(d[valid], DEPTH_NEAR_M, DEPTH_FAR_M)
    norm = (DEPTH_FAR_M - d_clamped) / max(DEPTH_FAR_M - DEPTH_NEAR_M, EPS)
    u8[valid] = np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """0~1 的标量图 -> 灰度 BGR（gamma=2.2），invalid 为黑。"""
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)

    gamma = 2.2
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m], 0.0, 1.0), 1.0 / gamma)

    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """hist (H,W,64) -> pred_depth, prob (H,W)."""
    import torch

    with torch.no_grad():
        inp = torch.from_numpy(hists).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        logits = net(inp)  # (1,NUM_BINS,H,W)
        probs_t = torch.softmax(logits, dim=1)  # (1,NUM_BINS,H,W)
        top_prob_t, top_idx_t = torch.max(probs_t, dim=1)  # (1,H,W)

        # pred depth:
        # - bin0 => invalid => depth=0
        # - bin k(1..63) => interval [k*BIN_M,(k+1)*BIN_M) => center=(k+0.5)*BIN_M
        idx_f = top_idx_t.to(dtype=torch.float32)
        pred_depth_t = torch.where(top_idx_t == 0, torch.zeros_like(idx_f), (idx_f + 0.5) * float(BIN_M))  # (1,H,W)
        pred_depth = pred_depth_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        prob = top_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    return pred_depth, prob


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
    net = Network(in_channels=TOF_C, out_bins=NUM_BINS).to(device)
    net.eval()

    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        net.load_state_dict(sd, strict=True)
        print(f"[load] {ckpt_path}")
    else:
        print(f"[warn] missing checkpoint: {ckpt_path} (use random weights)")

    cv2.namedWindow("NN_REALTIME", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("pThr%", "NN_REALTIME", 50, 100, lambda _: None)

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
    cached_prob: np.ndarray | None = None

    try:
        while True:
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                raw_u16 = np.frombuffer(frame.raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16)
                if hists.shape == (TOF_H, TOF_W, TOF_C):
                    pred_depth, prob = _run_infer(net, device, hists)
                    cached_in = hists
                    cached_pred_depth = pred_depth
                    cached_prob = prob
                    last_ts = float(frame.ts)

            if cached_in is None or cached_pred_depth is None or cached_prob is None:
                k = int(cv2.waitKey(5) & 0xFF)
                if k == 27:
                    break
                continue

            p_thr = float(cv2.getTrackbarPos("pThr%", "NN_REALTIME")) / 100.0

            # INPUT intensity
            inten_u8 = _render_input_intensity_u8(cached_in)
            in_big = cv2.resize(inten_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            in_big = cv2.flip(in_big, 0)
            in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

            # PRED depth
            pred_bgr = _colorize_depth(cached_pred_depth)
            conf_mask = cached_prob >= p_thr
            if np.any(~conf_mask):
                pred_bgr = pred_bgr.copy()
                pred_bgr[~conf_mask] = (0, 0, 0)
            pred_big = cv2.flip(cv2.resize(pred_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), 0)

            # PROB
            valid = cached_pred_depth > 0
            prob_bgr = _colorize_prob(cached_prob, valid)
            prob_big = cv2.flip(cv2.resize(prob_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), 0)

            # hover info（单行）
            mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 3 - 1))
            my_view = int(mouse.get("y", 0)) - int(HEADER_H)
            my = int(np.clip(my_view, 0, SHOW_H - 1))
            tile_x0 = 0 if mx < SHOW_W else (SHOW_W if mx < SHOW_W * 2 else SHOW_W * 2)
            px, py = _disp_xy_to_pixel(mx - tile_x0, my, SHOW_W, SHOW_H)

            pr_v = float(cached_pred_depth[py, px])
            pb_v = float(np.clip(cached_prob[py, px], 0.0, 1.0))
            # hover 显示用的 bin：
            if pr_v <= 0.0 or (not np.isfinite(pr_v)):
                hover_txt = f"pred --  bin[00] INVALID  p {pb_v:.2f}"
            else:
                k_bin = int(np.floor(pr_v / float(BIN_M)))  # 0..63
                k_bin = int(np.clip(k_bin, 0, NUM_BINS - 1))
                a_m = float(k_bin) * float(BIN_M)
                b_m = float(k_bin + 1) * float(BIN_M)
                hover_txt = f"pred {pr_v:.3f}m  bin[{k_bin:02d}] {a_m:.2f}-{b_m:.2f}m  p {pb_v:.2f}"

            # 标题文字
            in_bgr = _with_text(in_bgr, "INPUT")
            pred_big = _with_text(pred_big, "PRED")
            prob_big = _with_text(prob_big, "PROB")

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

            k = int(cv2.waitKey(1) & 0xFF)
            if k == 27:
                break
    finally:
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

