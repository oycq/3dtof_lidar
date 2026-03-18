#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

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

FIG_SIZE = (13.8, 7.2)
LABEL_SIZE = 14
TICK_SIZE = 11
LEGEND_SIZE = 12


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    out: List[Tuple[Path, Path]] = []
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
    wl = bins62[rows, cols, kl]
    wc = bins62[rows, cols, k]
    wr = bins62[rows, cols, kr]
    den = wl + wc + wr
    num = kl.astype(np.float32) * wl + k.astype(np.float32) * wc + kr.astype(np.float32) * wr
    centroid = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = den > 0.0
    centroid[m] = num[m] / den[m]
    return centroid * float(BIN_TO_DIST_M) + float(DIST_OFFSET_M)


def _conf_from_input(hists: np.ndarray) -> np.ndarray:
    b = np.asarray(hists[:, :, :HIST_BINS], dtype=np.float32)
    y = np.max(b, axis=2)
    x = np.mean(b, axis=2) + float(CONF_MEAN_BIAS)
    conf = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    m = y > 0.0
    conf[m] = 1.0 - (x[m] / y[m])
    return np.clip(conf, 0.0, 1.0)


def _guess_train_dir(this_dir: Path) -> Path:
    for p in [this_dir / "train_data", this_dir.parent / "train_data"]:
        if p.exists():
            return p
    return this_dir / "train_data"


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _fmt(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.6f}"


def _get_chinese_font() -> tuple[FontProperties | None, str]:
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


def _get_title_font(base_font_prop: FontProperties | None) -> FontProperties | None:
    yahei_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
    ]
    for fp in yahei_candidates:
        if fp.exists():
            return FontProperties(fname=str(fp))
    return base_font_prop


def _set_text_font(
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


def _apply_axis_font(ax: plt.Axes, font_prop: FontProperties | None) -> None:
    for t in ax.get_xticklabels():
        _set_text_font(t, font_prop, size=TICK_SIZE)
    for t in ax.get_yticklabels():
        _set_text_font(t, font_prop, size=TICK_SIZE)


def _print_metrics(title: str, pred: np.ndarray, gt: np.ndarray, rel_acc_thr: float) -> None:
    err = pred - gt
    abs_err = np.abs(err)
    rel_err = abs_err / np.clip(gt, EPS, np.inf)
    mae = _safe_mean(abs_err)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size > 0 else float("nan")
    bias = _safe_mean(err)
    mre = _safe_mean(rel_err)
    p50_ae = _safe_quantile(abs_err, 0.50)
    p90_ae = _safe_quantile(abs_err, 0.90)
    acc = _safe_mean((rel_err <= rel_acc_thr).astype(np.float32))
    print(f"[{title}] count={pred.size}")
    print(
        "  "
        f"MAE={_fmt(mae)}m  RMSE={_fmt(rmse)}m  BIAS={_fmt(bias)}m  "
        f"MRE={_fmt(mre)}  P50_AE={_fmt(p50_ae)}m  P90_AE={_fmt(p90_ae)}m  "
        f"ACC(rel<={rel_acc_thr*100:.1f}%)={_fmt(acc)}"
    )


def _show_plot(
    dist_axis: np.ndarray,
    acc_rate: np.ndarray,
    err_rate: np.ndarray,
    cnt: np.ndarray,
    conf_thr: float,
    rel_acc_thr: float,
) -> None:
    font_prop, font_name = _get_chinese_font()
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(FIG_SIZE[0], 10.2),
        constrained_layout=True,
        sharex=True,
    )

    ax_top.plot(
        dist_axis,
        acc_rate,
        color="#16a34a",
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"准确率（conf>{conf_thr*100:.0f}% 误差<={rel_acc_thr*100:.1f}%）",
        zorder=3,
    )
    ax_top.fill_between(dist_axis, 0.0, acc_rate, color="#22c55e", alpha=0.12, zorder=1)

    ax_top.plot(
        dist_axis,
        err_rate,
        color="#dc2626",
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"错误率（conf>{conf_thr*100:.0f}% 误差>{rel_acc_thr*100:.1f}%）",
        zorder=3,
    )
    ax_top.fill_between(dist_axis, 0.0, err_rate, color="#ef4444", alpha=0.10, zorder=1)

    title_font = _get_title_font(font_prop)
    title_obj = ax_top.set_title("不同距离检测准确率", pad=10)
    _set_text_font(title_obj, title_font, size=16, weight="bold")
    ax_top.set_xlabel("距离（米）", fontproperties=font_prop)
    ax_top.set_ylabel("比例", fontproperties=font_prop)
    _set_text_font(ax_top.xaxis.label, font_prop, size=LABEL_SIZE)
    _set_text_font(ax_top.yaxis.label, font_prop, size=LABEL_SIZE)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_xlim(float(dist_axis[0]) - 0.5, float(dist_axis[-1]) + 0.5)
    ax_top.set_xticks(dist_axis)
    ax_top.tick_params(axis="x", labelbottom=True)
    ax_top.grid(axis="y", linestyle="--", alpha=0.35)
    ax_top.grid(axis="x", linestyle=":", alpha=0.12)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    legend = ax_top.legend(loc="upper right", frameon=True, framealpha=0.9, prop=font_prop, fontsize=LEGEND_SIZE)
    if legend is not None:
        for t in legend.get_texts():
            _set_text_font(t, font_prop, size=LEGEND_SIZE)
    _apply_axis_font(ax_top, font_prop)

    ax_bottom.bar(
        dist_axis,
        cnt,
        width=0.82,
        color="#2563eb",
        edgecolor="#1d4ed8",
        alpha=0.88,
        zorder=3,
    )
    title_obj = ax_bottom.set_title("不同距离下数据点数量", pad=10)
    _set_text_font(title_obj, title_font, size=16, weight="bold")
    ax_bottom.set_xlabel("距离（米）", fontproperties=font_prop)
    ax_bottom.set_ylabel("数据点数量", fontproperties=font_prop)
    _set_text_font(ax_bottom.xaxis.label, font_prop, size=LABEL_SIZE)
    _set_text_font(ax_bottom.yaxis.label, font_prop, size=LABEL_SIZE)
    ax_bottom.set_xlim(float(dist_axis[0]) - 0.5, float(dist_axis[-1]) + 0.5)
    ax_bottom.set_xticks(dist_axis)
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.35)
    ax_bottom.grid(axis="x", linestyle=":", alpha=0.12)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    _apply_axis_font(ax_bottom, font_prop)

    if not font_name:
        print("[warn] 未找到中文字体文件，图表中文可能显示异常。")
    plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Traditional method distance precision statistics")
    parser.add_argument("--train-dir", type=str, default=None, help="default: auto guess train_data")
    parser.add_argument("--dist-min", type=float, default=DEFAULT_DIST_MIN_M, help="GT min distance (m)")
    parser.add_argument("--dist-max", type=float, default=DEFAULT_DIST_MAX_M, help="GT max distance (m)")
    parser.add_argument("--conf-thr", type=float, default=DEFAULT_CONF_THR, help="confidence threshold [0,1]")
    parser.add_argument(
        "--rel-acc-thr",
        type=float,
        default=DEFAULT_REL_ACC_THR,
        help="relative error threshold for accuracy, e.g. 0.07 means 7%%",
    )
    args = parser.parse_args()

    if args.dist_max < args.dist_min:
        raise ValueError("--dist-max must be >= --dist-min")

    this_dir = Path(__file__).resolve().parent
    train_dir = Path(args.train_dir) if args.train_dir else _guess_train_dir(this_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")

    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    pred_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []

    for ip, op in pairs:
        x, gt = _load_pair(ip, op)
        pred = _depth_from_peak3_centroid(x)
        conf = _conf_from_input(x)
        valid = (
            np.isfinite(gt)
            & np.isfinite(pred)
            & (gt >= float(args.dist_min))
            & (gt <= float(args.dist_max))
            & (gt > 0.0)
        )
        if not np.any(valid):
            continue
        pred_all.append(pred[valid].reshape(-1))
        gt_all.append(gt[valid].reshape(-1))
        conf_all.append(conf[valid].reshape(-1))

    if not pred_all:
        raise RuntimeError("no valid points found after filtering")

    pred_v = np.concatenate(pred_all, axis=0)
    gt_v = np.concatenate(gt_all, axis=0)
    conf_v = np.concatenate(conf_all, axis=0)

    print(f"[info] train_dir={train_dir}")
    print(f"[info] pairs={len(pairs)}")
    print(
        f"[info] filter: gt in [{args.dist_min:.3f}, {args.dist_max:.3f}]m, "
        f"conf_thr={args.conf_thr:.3f}, rel_acc_thr={args.rel_acc_thr*100:.2f}%"
    )
    _print_metrics("ALL_VALID", pred_v, gt_v, float(args.rel_acc_thr))

    use = conf_v >= float(args.conf_thr)
    _print_metrics("CONF_VALID", pred_v[use], gt_v[use], float(args.rel_acc_thr))

    # 按整米分桶，区间是 [d, d+1)。
    d0 = int(np.floor(float(args.dist_min)))
    d1 = int(np.floor(float(args.dist_max)))
    n_bins = d1 - d0 + 1
    cnt = np.zeros((n_bins,), dtype=np.int64)
    acc_cnt = np.zeros((n_bins,), dtype=np.int64)
    err_cnt = np.zeros((n_bins,), dtype=np.int64)

    abs_err = np.abs(pred_v - gt_v)
    rel_err = abs_err / np.clip(gt_v, EPS, np.inf)
    bucket = np.floor(gt_v).astype(np.int32)
    bucket = np.clip(bucket, d0, d1)
    bi = bucket - d0
    conf_ok = conf_v >= float(args.conf_thr)
    rel_ok = rel_err <= float(args.rel_acc_thr)
    np.add.at(cnt, bi, 1)
    np.add.at(acc_cnt, bi, (conf_ok & rel_ok).astype(np.int64))
    np.add.at(err_cnt, bi, (conf_ok & (~rel_ok)).astype(np.int64))

    print("distance_bin_m,count,accuracy,error_rate")
    acc_arr = np.full((n_bins,), np.nan, dtype=np.float64)
    err_arr = np.full((n_bins,), np.nan, dtype=np.float64)
    for d in range(d0, d1 + 1):
        i = d - d0
        c = int(cnt[i])
        acc = (float(acc_cnt[i]) / c) if c > 0 else float("nan")
        err = (float(err_cnt[i]) / c) if c > 0 else float("nan")
        acc_arr[i] = acc
        err_arr[i] = err
        print(f"{d},{c},{_fmt(acc)},{_fmt(err)}")

    dist_axis = np.arange(d0, d1 + 1, dtype=np.int32)
    _show_plot(
        dist_axis=dist_axis,
        acc_rate=acc_arr,
        err_rate=err_arr,
        cnt=cnt,
        conf_thr=float(args.conf_thr),
        rel_acc_thr=float(args.rel_acc_thr),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
