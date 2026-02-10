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
- ToF 强度与深度图做 resize + (rot90CW + flipH) 以对齐项目里的显示习惯（同 cali/check.py）
- 鼠标悬停显示：pred/gt/bin_range/prob（单行 ASCII，避免 cv2 putText 乱码）

模型输出语义（见 nn/net.py）：
- 每个像素输出 64 个 logits（softmax 后为 64 个类别概率）
  - 类别 0：无效（GT 超出量程/NA 等，训练时会落到该类）
  - 类别 1~63：距离区间概率（有效距离从 bin1 开始，bin0 专用于“无效概率”）
    - 类别 k 表示区间 [k*0.15m, (k+1)*0.15m)
- check 的 pred 距离：
  - top1 类别为 0 时视为无效，显示深度为 0（黑色）
  - 否则取 top1 区间中心值用于显示
- check 的 prob = top1 probability（当 top1=0 时即“无效概率”）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


TOF_H = 30
TOF_W = 40
TOF_C = 64

# 输出分类配置（与 train.py 对齐）
NUM_BINS = 64
BIN_M = 0.15
MIN_RANGE_M = BIN_M
MAX_RANGE_M = float(NUM_BINS) * float(BIN_M)

# 显示方向使用 rot90CW + flipH 后，原始 (H,W)=(30,40) 会变成 (40,30)，
# 因此显示的宽高比例应为 W:H = 30:40 = 3:4（竖屏）。
# 与 cali/check.py 对齐：300x400
SHOW_W = 390
SHOW_H = 520
HEADER_H = 32

# 直方图显示：展示输入的前 62 个 bin（0~61），去掉最后两个 bin（通常为特殊/保留）
HIST_BINS = 62
HIST_W = 620
HIST_H = 260

EPS = 1e-6


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标。

    显示方向对齐 cali/check.py：rot90CW + flipH。
    该组合等价于转置：变换后 (row, col) = (px, py)。
    因此 display 的 x 对应原始 py，display 的 y 对应原始 px。
    """
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
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
    # 小号细圆环：尽量不遮挡单个像素
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
    """把一维 bins 渲染成柱状直方图（BGR uint8）。

    - 默认显示输入的前 HIST_BINS 个 bin（0~61），最后两个 bin 不显示
    - 为了便于观察，按当前 bins 的最大值做线性归一化
    """
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

    # 画图区（留边距给文字和坐标）
    top = 34
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
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

    # 坐标框
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)

    # 柱子宽度（至少 1px）
    bar_area_w = max(x1 - x0, 1)
    bar_w = max(int(bar_area_w / nb), 1)
    gap = 0  # 可按需调成 1

    for i in range(nb):
        v = float(b[i])
        if not np.isfinite(v) or v < 0:
            v = 0.0
        hh = int(np.clip(v / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        xL = x0 + i * bar_w
        xR = min(xL + bar_w - gap, x1)
        if xR <= xL:
            continue
        yT = y1 - hh
        # 填充柱子（青色）+ 细边
        cv2.rectangle(img, (xL, yT), (xR, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (xL, yT), (xR, y1), (30, 30, 30), 1)

    # x 轴刻度（每 10 个 bin 一根）
    step = 10
    for k in range(0, nb, step):
        xx = x0 + int(k * bar_w)
        cv2.line(img, (xx, y1), (xx, y1 + 4), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, str(k), (xx + 2, sh - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # 标题（ASCII，避免乱码）
    ssum = float(np.sum(b))
    title2 = f"{title} bins[0..{nb-1}]  max {vmax:.3f}  sum {ssum:.3f}"
    img = _with_text(img, title2)
    return img


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """(H,W,64) -> (H,W) uint8 intensity (简单按 max 归一化)."""
    inten = np.sum(hists.astype(np.float32, copy=False), axis=2)
    vmax = float(np.max(inten)) if inten.size else 0.0
    if vmax <= 0.0:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    return np.clip(np.rint(inten / vmax * 255.0), 0, 255).astype(np.uint8)


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H,W) depth(m) -> BGR (near red, far blue), 0 为黑。

    注意：这个函数会“自适应”本张图的动态范围（vmin/vmax），因此如果要让
    GT 和 PRED 使用同一套色彩映射，请改用 `_colorize_depth_with_range`，
    并用 GT 计算出来的 (vmin,vmax) 去给 PRED 上色。
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


def _inv_depth_range_from_depth(depth_m: np.ndarray) -> tuple[float, float] | None:
    """从 depth(m) 计算 inv_depth 的 (vmin,vmax)；无有效像素返回 None。"""
    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return None
    inv_v = 1.0 / np.clip(d[valid], EPS, np.inf)
    vmin = float(np.min(inv_v))
    vmax = float(np.max(inv_v))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _colorize_depth_with_range(depth_m: np.ndarray, inv_vmin: float, inv_vmax: float) -> np.ndarray:
    """用指定的 inv_depth 范围给 depth(m) 上色（GT/PRED 统一色彩系统用）。"""
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    inv = np.zeros_like(d, dtype=np.float32)
    inv[valid] = 1.0 / np.clip(d[valid], EPS, np.inf)

    vmin = float(inv_vmin)
    vmax = float(inv_vmax)
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
    """0~1 的标量图 -> 灰度 BGR（gamma=2.2），invalid 为黑。

    历史上这里用于显示 prob；现在用于显示 “P(gt 在 pred±5% 区间)” 的概率图。
    """
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
    net = Network(in_channels=TOF_C, out_bins=NUM_BINS).to(device)
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
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("OUT_HIST", cv2.WINDOW_AUTOSIZE)
    # 概率阈值滑动条：低于该阈值的像素，PRED 直接置黑不显示
    # 概率 = top1 区间的 softmax 概率
    cv2.createTrackbar("pThr%", "CHECK_TRAIN", 50, 100, lambda _: None)

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
    cached_out_probs: np.ndarray | None = None  # (H,W,NUM_BINS) float32

    while True:
        ip, op = pairs[idx]
        if cached_idx != idx:
            x, gt = _load_pair(ip, op)
            # 与训练对齐的 GT 有效性（用于显示/hover 文案）
            valid_gt = np.isfinite(gt) & (gt >= float(MIN_RANGE_M)) & (gt < float(MAX_RANGE_M))

            # run net
            with torch.no_grad():
                inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                logits = net(inp)  # (1,NUM_BINS,H,W)
                probs_t = torch.softmax(logits, dim=1)  # (1,NUM_BINS,H,W)
                top_prob_t, top_idx_t = torch.max(probs_t, dim=1)  # (1,H,W)

                # pred depth:
                # - class0 => invalid => depth=0
                # - class k(1..63) => interval [k*BIN_M,(k+1)*BIN_M) => center=(k+0.5)*BIN_M
                idx_f = top_idx_t.to(dtype=torch.float32)
                pred_depth_t = torch.where(top_idx_t == 0, torch.zeros_like(idx_f), (idx_f + 0.5) * float(BIN_M))  # (1,H,W)

                pred_depth = (
                    pred_depth_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                )  # (H,W)
                prob = top_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)  # (H,W)
                out_probs = (
                    probs_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32, copy=False)
                )  # (H,W,NUM_BINS)

            cached_in, cached_gt = x, gt
            cached_pred_depth, cached_prob, cached_out_probs = pred_depth, prob, out_probs
            cached_idx = idx

        assert (
            cached_in is not None
            and cached_gt is not None
            and cached_pred_depth is not None
            and cached_prob is not None
            and cached_out_probs is not None
        )

        valid_gt = np.isfinite(cached_gt) & (cached_gt >= float(MIN_RANGE_M)) & (cached_gt < float(MAX_RANGE_M))
        p_thr = float(cv2.getTrackbarPos("pThr%", "CHECK_TRAIN")) / 100.0
        # INPUT intensity
        inten_u8 = _render_input_intensity_u8(cached_in)
        # 显示方向对齐 cali/check.py：向右旋转90° + 水平翻转
        inten_u8 = cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE)
        inten_u8 = cv2.flip(inten_u8, 1)
        in_big = cv2.resize(inten_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

        # GT / PRED / PROB(±5%)
        # 关键：以 GT 的动态范围建立“距离→颜色”映射系统，PRED 跟随同一套映射
        inv_range = _inv_depth_range_from_depth(cached_gt)
        if inv_range is None:
            gt_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
            pred_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
        else:
            inv_vmin, inv_vmax = inv_range
            gt_bgr = _colorize_depth_with_range(cached_gt, inv_vmin, inv_vmax)
            pred_bgr = _colorize_depth_with_range(cached_pred_depth, inv_vmin, inv_vmax)

        # 概率过滤：低概率像素不显示（置黑）
        conf_mask = cached_prob >= p_thr
        if np.any(~conf_mask):
            pred_bgr = pred_bgr.copy()
            pred_bgr[~conf_mask] = (0, 0, 0)

        # PROB 图：显示 top1 probability（不依赖 GT）
        valid_all = np.ones((TOF_H, TOF_W), dtype=bool)
        prob_bgr = _colorize_prob(cached_prob, valid_all)

        gt_bgr = cv2.flip(cv2.rotate(gt_bgr, cv2.ROTATE_90_CLOCKWISE), 1)
        pred_bgr = cv2.flip(cv2.rotate(pred_bgr, cv2.ROTATE_90_CLOCKWISE), 1)
        prob_bgr = cv2.flip(cv2.rotate(prob_bgr, cv2.ROTATE_90_CLOCKWISE), 1)

        gt_big = cv2.resize(gt_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        pred_big = cv2.resize(pred_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
        prob_big = cv2.resize(prob_bgr, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

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

        # bin range
        k_top = int(np.argmax(cached_out_probs[py, px, :]))
        # 仅显示三项，且全 ASCII，避免 putText 乱码
        if k_top == 0:
            hover_txt = (
                f"pred --  gt {gt_v:.3f}m  bin[00] INVALID  p {pb_v:.2f}"
                if gt_v > 0
                else f"pred --  gt --  bin[00] INVALID  p {pb_v:.2f}"
            )
        else:
            a_m = float(k_top) * float(BIN_M)
            b_m = float(k_top + 1) * float(BIN_M)
            hover_txt = (
                f"pred {pr_v:.3f}m  gt {gt_v:.3f}m  bin[{k_top:02d}] {a_m:.2f}-{b_m:.2f}m  p {pb_v:.2f}"
                if gt_v > 0
                else f"pred {pr_v:.3f}m  gt --  bin[{k_top:02d}] {a_m:.2f}-{b_m:.2f}m  p {pb_v:.2f}"
            )

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

        # hovered point histogram（取输入的前 62 个 bin：0~61）
        try:
            hbins = cached_in[py, px, :]
        except Exception:
            hbins = np.zeros((TOF_C,), dtype=np.float32)
        hist_img = _render_histogram_bgr(hbins, w=HIST_W, h=HIST_H, max_bins=HIST_BINS, title="IN_HIST")
        cv2.imshow("HIST", hist_img)

        # hovered point output histogram（网络输出 NUM_CLASSES 个概率）
        try:
            obins = cached_out_probs[py, px, :]
        except Exception:
            obins = np.zeros((NUM_BINS,), dtype=np.float32)
        # 概率直方图 y 轴固定到 1.0（概率范围）
        out_hist_img = _render_histogram_bgr(
            obins, w=HIST_W, h=HIST_H, max_bins=NUM_BINS, title="OUT_HIST", fixed_vmax=1.0
        )
        cv2.imshow("OUT_HIST", out_hist_img)

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


