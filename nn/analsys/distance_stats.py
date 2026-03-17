#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 GT 距离分桶统计（0m~29m，1m 一个桶）：
- 准确率：conf>50% 且 |pred-gt|/gt<=ACC_REL_THR 的比例
- 错误率：conf>50% 且 |pred-gt|/gt>ACC_REL_THR 的比例

数据来源：
- 输入：nn/train_data/input_*.npy   (30,40,64)
- 真值：nn/train_data/output_*.npy  (30,40), 单位米
- 模型：nn/model_last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import torch


TOF_H = 30
TOF_W = 40
TOF_C = 64

DIST_MIN_M = 0
DIST_MAX_M = 29
# 宏定义：conf 大于该阈值（百分比）才参与“准确率/错误率”统计。
CONF_POS_PCT = 50.0
ACC_REL_THR = 0.07
EPS = 1e-6

FIG_SIZE = (13.8, 7.2)
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 12


def find_pairs(train_dir: Path) -> list[tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    pairs: list[tuple[Path, Path]] = []
    for ip in ins:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def load_pair(ip: Path, op: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def format_float(v: float) -> str:
    if np.isnan(v):
        return "nan"
    return f"{v:.6f}"


def get_chinese_font() -> tuple[FontProperties | None, str]:
    """通过字体文件路径强制加载中文字体，避免仅按字体名匹配失败。"""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),  # 微软雅黑
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),  # 黑体
        Path("C:/Windows/Fonts/simsun.ttc"),  # 宋体
        Path("C:/Windows/Fonts/simkai.ttf"),  # 楷体
        Path("C:/Windows/Fonts/STSONG.TTF"),
    ]
    for fp in candidates:
        if fp.exists():
            return FontProperties(fname=str(fp)), fp.name
    return None, ""


def set_text_font(
    text_obj: object,
    font_prop: FontProperties | None,
    size: int | float | None = None,
    weight: str | None = None,
) -> None:
    """统一设置文本字体，避免在每个位置写重复代码。"""
    if font_prop is not None and hasattr(text_obj, "set_fontproperties"):
        text_obj.set_fontproperties(font_prop)
    if size is not None and hasattr(text_obj, "set_fontsize"):
        text_obj.set_fontsize(size)
    if weight is not None and hasattr(text_obj, "set_fontweight"):
        text_obj.set_fontweight(weight)


def apply_axis_font(ax: plt.Axes, font_prop: FontProperties | None) -> None:
    for t in ax.get_xticklabels():
        set_text_font(t, font_prop, size=TICK_SIZE)
    for t in ax.get_yticklabels():
        set_text_font(t, font_prop, size=TICK_SIZE)


def save_plot(
    dist_axis: np.ndarray,
    acc_rate: np.ndarray,
    err_rate: np.ndarray,
) -> None:
    # 更简洁的双曲线风格：准确率 vs 错误率。
    font_prop, font_name = get_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=True)

    ax.plot(
        dist_axis,
        acc_rate,
        color="#16a34a",
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"准确率（conf>{CONF_POS_PCT:.0f}% 且预测正确）",
        zorder=3,
    )
    ax.fill_between(dist_axis, 0.0, acc_rate, color="#22c55e", alpha=0.12, zorder=1)

    ax.plot(
        dist_axis,
        err_rate,
        color="#dc2626",
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"错误率（conf>{CONF_POS_PCT:.0f}% 且预测错误）",
        zorder=3,
    )
    ax.fill_between(dist_axis, 0.0, err_rate, color="#ef4444", alpha=0.10, zorder=1)

    title_obj = ax.set_title("不同距离检测准确率", pad=16)
    set_text_font(title_obj, font_prop, size=20, weight="bold")
    ax.set_xlabel("距离（米）", fontproperties=font_prop)
    ax.set_ylabel("比例", fontproperties=font_prop)
    set_text_font(ax.xaxis.label, font_prop, size=LABEL_SIZE)
    set_text_font(ax.yaxis.label, font_prop, size=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(dist_axis[0]) - 0.5, float(dist_axis[-1]) + 0.5)
    ax.set_xticks(dist_axis)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="upper right", frameon=True, framealpha=0.9, prop=font_prop, fontsize=LEGEND_SIZE)
    if legend is not None:
        for t in legend.get_texts():
            set_text_font(t, font_prop, size=LEGEND_SIZE)
    apply_axis_font(ax, font_prop)

    if not font_name:
        print("[warn] 未找到中文字体文件，图表中文可能仍显示异常。")

    plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="统计 0~29m 各距离桶准确率/错误率，并绘图")
    parser.add_argument("--train-dir", type=str, default=None, help="训练数据目录，默认 nn/train_data")
    parser.add_argument("--ckpt", type=str, default=None, help="模型权重路径，默认 nn/model_last.pt")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    nn_dir = here.parent
    train_dir = Path(args.train_dir) if args.train_dir else (nn_dir / "train_data")
    ckpt_path = Path(args.ckpt) if args.ckpt else (nn_dir / "model_last.pt")

    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")
    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()

    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    net.load_state_dict(sd, strict=True)

    # 每个桶对应 [d, d+1)，d in [0..29]。
    n_bins = DIST_MAX_M - DIST_MIN_M + 1
    cnt = np.zeros((n_bins,), dtype=np.int64)
    acc_cnt = np.zeros((n_bins,), dtype=np.int64)
    err_cnt = np.zeros((n_bins,), dtype=np.int64)
    conf_thr = float(CONF_POS_PCT) / 100.0

    with torch.no_grad():
        for ip, op in pairs:
            x, gt = load_pair(ip, op)
            inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            out = net.forward_train(inp)
            pred = out["dist"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            conf = out["conf"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

            valid = np.isfinite(gt) & (gt > 0.0) & (gt <= float(DIST_MAX_M))
            if not np.any(valid):
                continue

            gt_v = gt[valid]
            pred_v = pred[valid]
            conf_v = conf[valid]

            # floor 分桶，范围限制到 [0..29]
            bucket = np.floor(gt_v).astype(np.int32)
            bucket = np.clip(bucket, DIST_MIN_M, DIST_MAX_M)
            bi = bucket - DIST_MIN_M

            rel_err = np.abs(pred_v - gt_v) / np.clip(gt_v, EPS, np.inf)
            conf_pos = conf_v > conf_thr
            ok = rel_err <= float(ACC_REL_THR)
            acc_hit = conf_pos & ok
            err_hit = conf_pos & (~ok)

            np.add.at(cnt, bi, 1)
            np.add.at(acc_cnt, bi, acc_hit.astype(np.int64))
            np.add.at(err_cnt, bi, err_hit.astype(np.int64))

    dist_axis = np.arange(DIST_MIN_M, DIST_MAX_M + 1, dtype=np.int32)
    accuracy = np.full((n_bins,), np.nan, dtype=np.float64)
    error_rate = np.full((n_bins,), np.nan, dtype=np.float64)
    nz = cnt > 0
    accuracy[nz] = acc_cnt[nz] / cnt[nz]
    error_rate[nz] = err_cnt[nz] / cnt[nz]
    save_plot(dist_axis=dist_axis, acc_rate=accuracy, err_rate=error_rate)

    print(f"[done] pairs={len(pairs)}")
    print("[done] plot shown on screen")
    print(f"[done] conf threshold: > {CONF_POS_PCT:.1f}%")
    print("distance_bin_m,count,accuracy,error_rate")
    for d in range(DIST_MIN_M, DIST_MAX_M + 1):
        i = d - DIST_MIN_M
        c = int(cnt[i])
        acc = (float(acc_cnt[i]) / float(c)) if c > 0 else float("nan")
        err = (float(err_cnt[i]) / float(c)) if c > 0 else float("nan")
        print(f"{d},{c},{format_float(acc)},{format_float(err)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

