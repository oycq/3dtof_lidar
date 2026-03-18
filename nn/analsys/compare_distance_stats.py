#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import torch

TOF_H, TOF_W, TOF_C = 30, 40, 64
HIST_BINS = 62
CONF_MEAN_BIAS = 20.0
BIN_TO_DIST_M = 0.6
DIST_OFFSET_M = 0.5
EPS = 1e-6

DEFAULT_DIST_MIN_M = 2
DEFAULT_DIST_MAX_M = 29
DEFAULT_CONF_THR = 0.5
DEFAULT_REL_ACC_THR = 0.12

LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 11


def find_pairs(train_dir: Path) -> list[tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    out: list[tuple[Path, Path]] = []
    for ip in ins:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            out.append((ip, op))
    return out


def load_pair(ip: Path, op: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def traditional_depth_from_peak3_centroid(hists: np.ndarray) -> np.ndarray:
    bins62 = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    k = np.argmax(bins62, axis=2).astype(np.int32)
    kl = np.clip(k - 1, 0, HIST_BINS - 1)
    kr = np.clip(k + 1, 0, HIST_BINS - 1)
    rows = np.arange(TOF_H, dtype=np.int32)[:, None]
    cols = np.arange(TOF_W, dtype=np.int32)[None, :]
    wl = bins62[rows, cols, kl]
    wc = bins62[rows, cols, k]
    wr = bins62[rows, cols, kr]
    den = wl + wc + wr
    num = kl.astype(np.float32) * wl + k.astype(np.float32) * wc + kr.astype(np.float32) * wr
    centroid = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = den > 0.0
    centroid[m] = num[m] / den[m]
    return centroid * float(BIN_TO_DIST_M) + float(DIST_OFFSET_M)


def traditional_conf_from_input(hists: np.ndarray) -> np.ndarray:
    b = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    y = np.max(b, axis=2)
    x = np.mean(b, axis=2) + float(CONF_MEAN_BIAS)
    conf = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = y > 0.0
    conf[m] = 1.0 - (x[m] / y[m])
    return np.clip(conf, 0.0, 1.0)


def infer_nn(net: torch.nn.Module, device: torch.device, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    out = net.forward_train(inp)
    pred = out["dist"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    conf = out["conf"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return pred, conf


def format_float(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.6f}"


def init_bins(dist_min: int, dist_max: int) -> np.ndarray:
    n_bins = dist_max - dist_min + 1
    cnt = np.zeros((n_bins,), dtype=np.int64)
    return cnt


def update_count_bins(
    cnt: np.ndarray,
    gt: np.ndarray,
    dist_min: int,
    dist_max: int,
) -> None:
    bucket = np.floor(gt).astype(np.int32)
    bucket = np.clip(bucket, dist_min, dist_max)
    bi = bucket - dist_min
    np.add.at(cnt, bi, 1)


def update_hit_bins(
    acc_cnt: np.ndarray,
    err_cnt: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    dist_min: int,
    dist_max: int,
    conf_thr: float,
    rel_acc_thr: float,
) -> None:
    bucket = np.floor(gt).astype(np.int32)
    bucket = np.clip(bucket, dist_min, dist_max)
    bi = bucket - dist_min
    rel_err = np.abs(pred - gt) / np.clip(gt, EPS, np.inf)
    conf_pos = conf > conf_thr
    ok = rel_err <= rel_acc_thr
    acc_hit = conf_pos & ok
    err_hit = conf_pos & (~ok)

    np.add.at(acc_cnt, bi, acc_hit.astype(np.int64))
    np.add.at(err_cnt, bi, err_hit.astype(np.int64))


def calc_rates(den_cnt: np.ndarray, acc_cnt: np.ndarray, err_cnt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bins = den_cnt.size
    acc = np.full((n_bins,), np.nan, dtype=np.float64)
    err = np.full((n_bins,), np.nan, dtype=np.float64)
    nz = den_cnt > 0
    acc[nz] = acc_cnt[nz] / den_cnt[nz]
    err[nz] = err_cnt[nz] / den_cnt[nz]
    return acc, err


def get_chinese_font() -> tuple[FontProperties | None, str]:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simkai.ttf"),
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


def show_compare_plot(
    dist_axis: np.ndarray,
    trad_acc: np.ndarray,
    trad_err: np.ndarray,
    nn_acc: np.ndarray,
    nn_err: np.ndarray,
    shared_cnt: np.ndarray,
    conf_thr: float,
    rel_acc_thr: float,
) -> None:
    font_prop, font_name = get_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, (ax_acc, ax_err, ax_cnt) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14.2, 12.4),
        constrained_layout=True,
        sharex=True,
    )

    ax_acc.plot(
        dist_axis,
        nn_acc,
        color="#2563eb",
        linewidth=2.4,
        linestyle="-",
        marker="s",
        markersize=4.8,
        label="NN准确率（实线）",
    )
    ax_acc.plot(
        dist_axis,
        trad_acc,
        color="#60a5fa",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=4.6,
        label="传统准确率（虚线）",
    )

    title = ax_acc.set_title(
        f"传统方法 vs NN 方法 测距精度对比（conf>{conf_thr*100:.0f}% | rel<={rel_acc_thr*100:.1f}%）",
        pad=10,
    )
    set_text_font(title, font_prop, size=15, weight="bold")
    ax_acc.set_ylabel("准确率", fontproperties=font_prop)
    set_text_font(ax_acc.yaxis.label, font_prop, size=LABEL_SIZE)
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_xlim(float(dist_axis[0]) - 0.6, float(dist_axis[-1]) + 0.6)
    ax_acc.grid(axis="y", linestyle="--", alpha=0.35)
    ax_acc.grid(axis="x", linestyle=":", alpha=0.15)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    legend_acc = ax_acc.legend(loc="upper right", frameon=True, framealpha=0.9, prop=font_prop, fontsize=LEGEND_SIZE)
    if legend_acc is not None:
        for t in legend_acc.get_texts():
            set_text_font(t, font_prop, size=LEGEND_SIZE)
    apply_axis_font(ax_acc, font_prop)

    ax_err.plot(
        dist_axis,
        nn_err,
        color="#dc2626",
        linewidth=2.4,
        linestyle="-",
        marker="s",
        markersize=4.8,
        label="NN错误率（实线）",
    )
    ax_err.plot(
        dist_axis,
        trad_err,
        color="#f87171",
        linewidth=2.4,
        linestyle="--",
        marker="o",
        markersize=4.6,
        label="传统错误率（虚线）",
    )
    ax_err.set_ylabel("错误率", fontproperties=font_prop)
    set_text_font(ax_err.yaxis.label, font_prop, size=LABEL_SIZE)
    ax_err.set_ylim(0.0, 1.0)
    ax_err.grid(axis="y", linestyle="--", alpha=0.35)
    ax_err.grid(axis="x", linestyle=":", alpha=0.15)
    ax_err.spines["top"].set_visible(False)
    ax_err.spines["right"].set_visible(False)
    legend_err = ax_err.legend(loc="upper right", frameon=True, framealpha=0.9, prop=font_prop, fontsize=LEGEND_SIZE)
    if legend_err is not None:
        for t in legend_err.get_texts():
            set_text_font(t, font_prop, size=LEGEND_SIZE)
    apply_axis_font(ax_err, font_prop)

    ax_cnt.bar(
        dist_axis,
        shared_cnt,
        width=0.76,
        color="#38bdf8",
        edgecolor="#0369a1",
        alpha=0.92,
        label="不同距离的点的数量",
    )
    ax_cnt.set_xlabel("距离（米）", fontproperties=font_prop)
    ax_cnt.set_ylabel("个", fontproperties=font_prop)
    set_text_font(ax_cnt.xaxis.label, font_prop, size=LABEL_SIZE)
    set_text_font(ax_cnt.yaxis.label, font_prop, size=LABEL_SIZE)
    ax_cnt.set_xticks(dist_axis)
    ax_cnt.grid(axis="y", linestyle="--", alpha=0.35)
    ax_cnt.grid(axis="x", linestyle=":", alpha=0.15)
    ax_cnt.spines["top"].set_visible(False)
    ax_cnt.spines["right"].set_visible(False)
    legend_cnt = ax_cnt.legend(loc="upper right", frameon=True, framealpha=0.9, prop=font_prop, fontsize=LEGEND_SIZE)
    if legend_cnt is not None:
        for t in legend_cnt.get_texts():
            set_text_font(t, font_prop, size=LEGEND_SIZE)
    apply_axis_font(ax_cnt, font_prop)

    if not font_name:
        print("[warn] 未找到中文字体文件，图表中文可能显示异常。")
    plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比传统方法与NN方法的距离分桶准确率/错误率")
    parser.add_argument("--train-dir", type=str, default=None, help="默认 nn/train_data")
    parser.add_argument("--ckpt", type=str, default=None, help="默认 nn/model_last.pt")
    parser.add_argument("--dist-min", type=int, default=DEFAULT_DIST_MIN_M, help="最小GT距离（米）")
    parser.add_argument("--dist-max", type=int, default=DEFAULT_DIST_MAX_M, help="最大GT距离（米）")
    parser.add_argument("--conf-thr", type=float, default=DEFAULT_CONF_THR, help="置信度阈值[0,1]")
    parser.add_argument("--rel-acc-thr", type=float, default=DEFAULT_REL_ACC_THR, help="准确率相对误差阈值")
    args = parser.parse_args()

    if args.dist_max < args.dist_min:
        raise ValueError("--dist-max must be >= --dist-min")

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

    shared_cnt = init_bins(args.dist_min, args.dist_max)
    trad_acc_cnt = init_bins(args.dist_min, args.dist_max)
    trad_err_cnt = init_bins(args.dist_min, args.dist_max)
    nn_acc_cnt = init_bins(args.dist_min, args.dist_max)
    nn_err_cnt = init_bins(args.dist_min, args.dist_max)

    with torch.no_grad():
        for ip, op in pairs:
            x, gt = load_pair(ip, op)
            gt_ok = np.isfinite(gt) & (gt >= float(args.dist_min)) & (gt <= float(args.dist_max))
            if not np.any(gt_ok):
                continue
            update_count_bins(
                cnt=shared_cnt,
                gt=gt[gt_ok].reshape(-1),
                dist_min=args.dist_min,
                dist_max=args.dist_max,
            )

            trad_pred = traditional_depth_from_peak3_centroid(x)
            trad_conf = traditional_conf_from_input(x)
            trad_valid = gt_ok & np.isfinite(trad_pred) & np.isfinite(trad_conf)
            if np.any(trad_valid):
                update_hit_bins(
                    acc_cnt=trad_acc_cnt,
                    err_cnt=trad_err_cnt,
                    gt=gt[trad_valid].reshape(-1),
                    pred=trad_pred[trad_valid].reshape(-1),
                    conf=trad_conf[trad_valid].reshape(-1),
                    dist_min=args.dist_min,
                    dist_max=args.dist_max,
                    conf_thr=float(args.conf_thr),
                    rel_acc_thr=float(args.rel_acc_thr),
                )

            nn_pred, nn_conf = infer_nn(net=net, device=device, x=x)
            nn_valid = gt_ok & np.isfinite(nn_pred) & np.isfinite(nn_conf)
            if np.any(nn_valid):
                update_hit_bins(
                    acc_cnt=nn_acc_cnt,
                    err_cnt=nn_err_cnt,
                    gt=gt[nn_valid].reshape(-1),
                    pred=nn_pred[nn_valid].reshape(-1),
                    conf=nn_conf[nn_valid].reshape(-1),
                    dist_min=args.dist_min,
                    dist_max=args.dist_max,
                    conf_thr=float(args.conf_thr),
                    rel_acc_thr=float(args.rel_acc_thr),
                )

    dist_axis = np.arange(args.dist_min, args.dist_max + 1, dtype=np.int32)
    trad_acc, trad_err = calc_rates(shared_cnt, trad_acc_cnt, trad_err_cnt)
    nn_acc, nn_err = calc_rates(shared_cnt, nn_acc_cnt, nn_err_cnt)

    print(f"[done] pairs={len(pairs)}")
    print(f"[done] train_dir={train_dir}")
    print(f"[done] ckpt={ckpt_path}")
    print(
        f"[done] filter: gt in [{args.dist_min}, {args.dist_max}]m, "
        f"conf>{args.conf_thr*100:.1f}%, rel<={args.rel_acc_thr*100:.1f}%"
    )
    print("distance_bin_m,shared_count,trad_accuracy,trad_error_rate,nn_accuracy,nn_error_rate")
    for d in range(args.dist_min, args.dist_max + 1):
        i = d - args.dist_min
        print(
            f"{d},{int(shared_cnt[i])},{format_float(float(trad_acc[i]))},{format_float(float(trad_err[i]))},"
            f"{format_float(float(nn_acc[i]))},{format_float(float(nn_err[i]))}"
        )

    show_compare_plot(
        dist_axis=dist_axis,
        trad_acc=trad_acc,
        trad_err=trad_err,
        nn_acc=nn_acc,
        nn_err=nn_err,
        shared_cnt=shared_cnt,
        conf_thr=float(args.conf_thr),
        rel_acc_thr=float(args.rel_acc_thr),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
