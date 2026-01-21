#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_train_effect.py

用途：检测 nn/train.py 训练后的效果
- 输入：nn/train_data/input_*.npy  (30,40,64) float32
- 真值：nn/train_data/output_*.npy (30,40) float32, 单位米；无效为 0 或 <=0
- 模型：nn/model_last.pt

可视化风格参考 cali/check.py：
- cv2.imshow
- 4/6 切换样本，ESC 退出
- ToF 强度与深度图做 resize + flipV 以对齐项目里的显示习惯
- 鼠标悬停显示：pred/gt/prob（三个值，单行 ASCII，避免 cv2 putText 乱码）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


TOF_H = 30
TOF_W = 40
TOF_C = 64

SHOW_W = 400
SHOW_H = 300
HEADER_H = 32

EPS = 1e-6


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
    # disp 上的 y 对应 py_disp（0=top）: py = (TOF_H-1) - py_disp
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
    # 小号细圆环：尽量不遮挡单个像素
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
    """(H,W) depth(m) -> BGR (near red, far blue), 0 为黑.

    实现：用 inv_depth = 1/depth 做归一化，再用 JET（低=蓝，高=红）。
    """
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    inv = np.zeros_like(d, dtype=np.float32)
    inv[valid] = 1.0 / np.clip(d[valid], EPS, np.inf)
    iv = inv[valid]
    vmin = float(np.min(iv))
    vmax = float(np.max(iv))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    u8[valid] = np.clip(np.rint((inv[valid] - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_error(err_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """abs error (m) -> BGR INFERNO, invalid 为黑。"""
    import cv2  # type: ignore

    e = np.asarray(err_m, dtype=np.float32)
    m = (valid.astype(bool)) & np.isfinite(e)
    if not np.any(m):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    ev = e[m]
    vmax = float(np.percentile(ev, 95)) if ev.size else 0.0
    vmax = max(vmax, 1e-6)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    u8[m] = np.clip(np.rint(np.minimum(e[m], vmax) / vmax * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    bgr[~m] = (0, 0, 0)
    return bgr


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """prob 0~1 -> 灰度 BGR（gamma=2.2），invalid 为黑。"""
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)

    # 仅用于显示：先 clamp 到 0~1，再做 gamma 映射（更易观察低值层次）
    gamma = 2.2
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m], 0.0, 1.0), 1.0 / gamma)

    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    pairs: List[Tuple[Path, Path]] = []
    for ip in ins:
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
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("missing dependency torch") from e

    # 本文件已移动到 nn/ 下：nn_dir = .../nn
    nn_dir = Path(__file__).resolve().parent
    root = nn_dir.parent
    train_dir = nn_dir / "train_data"
    ckpt_path = nn_dir / "model_last.pt"

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    # import 网络：把 nn 加到 sys.path，直接 import net.py
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()

    if ckpt_path.exists():
        # 优先使用 weights_only=True（新版本 torch 支持），避免 pickle 风险提示
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        net.load_state_dict(sd, strict=True)
        print(f"[load] {ckpt_path}")
    else:
        print(f"[warn] missing checkpoint: {ckpt_path} (use random weights)")

    cv2.namedWindow("CHECK_TRAIN", cv2.WINDOW_AUTOSIZE)

    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback("CHECK_TRAIN", on_mouse)

    idx = 0
    cached_idx = -1
    cached_in: np.ndarray | None = None
    cached_gt: np.ndarray | None = None
    cached_pred_depth: np.ndarray | None = None
    cached_prob: np.ndarray | None = None

    while True:
        ip, op = pairs[idx]
        if cached_idx != idx:
            x, gt = _load_pair(ip, op)
            valid = gt > 0

            # run net
            with torch.no_grad():
                inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                out = net(inp)  # (1,2,H,W)
                pred_inv = out[:, 0:1].clamp(min=EPS)
                pred_prob = out[:, 1:2]
                pred_depth = (1.0 / pred_inv).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                prob = pred_prob.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            cached_in, cached_gt = x, gt
            cached_pred_depth, cached_prob = pred_depth, prob
            cached_idx = idx

        assert cached_in is not None and cached_gt is not None and cached_pred_depth is not None and cached_prob is not None

        valid = cached_gt > 0
        # INPUT intensity
        inten_u8 = _render_input_intensity_u8(cached_in)
        in_big = cv2.resize(inten_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        in_big = cv2.flip(in_big, 0)
        in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

        # GT / PRED / PROB
        gt_bgr = _colorize_depth(cached_gt)
        pred_bgr = _colorize_depth(cached_pred_depth)
        prob_bgr = _colorize_prob(cached_prob, valid)

        gt_big = cv2.flip(cv2.resize(gt_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), 0)
        pred_big = cv2.flip(cv2.resize(pred_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), 0)
        prob_big = cv2.flip(cv2.resize(prob_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST), 0)

        # hover info（单窗口拼图：2x2 + 顶部 header，需要先扣掉 header 高度）
        mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 2 - 1))
        my_view = int(mouse.get("y", 0)) - int(HEADER_H)
        my = int(np.clip(my_view, 0, SHOW_H * 2 - 1))
        tile_x0 = 0 if mx < SHOW_W else SHOW_W
        tile_y0 = 0 if my < SHOW_H else SHOW_H
        px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, SHOW_W, SHOW_H)
        gt_v = float(cached_gt[py, px])
        pr_v = float(cached_pred_depth[py, px])
        pb_v = float(np.clip(cached_prob[py, px], 0.0, 1.0))
        # 仅显示三项，且全 ASCII，避免 putText 乱码
        hover_txt = f"pred {pr_v:.3f}  gt {gt_v:.3f}  prob {pb_v:.2f}" if gt_v > 0 else f"pred {pr_v:.3f}  gt --  prob {pb_v:.2f}"

        # 单窗口拼图，更方便截图
        in_bgr = _with_text(in_bgr, "INPUT")
        gt_big = _with_text(gt_big, "GT")
        pred_big = _with_text(pred_big, "PRED")
        prob_big = _with_text(prob_big, "PROB")

        # 四张图同步绘制鼠标指向的点
        dx_m, dy_m = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)
        in_bgr = _draw_marker(in_bgr, dx_m, dy_m)
        gt_big = _draw_marker(gt_big, dx_m, dy_m)
        pred_big = _draw_marker(pred_big, dx_m, dy_m)
        prob_big = _draw_marker(prob_big, dx_m, dy_m)

        top = np.hstack([in_bgr, gt_big])
        bot = np.hstack([pred_big, prob_big])
        view = np.vstack([top, bot])
        # hover 行单独放到顶部标题栏，避免和 tile 内的 "INPUT/GT/..." 文本重叠
        header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
        header = _with_text(header, hover_txt)
        view = np.vstack([header, view])

        cv2.imshow("CHECK_TRAIN", view)

        k = int(cv2.waitKey(30) & 0xFF)
        if k == 27:  # ESC
            break
        elif k == ord("4"):
            idx = (idx - 1) % len(pairs)
        elif k == ord("6"):
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


