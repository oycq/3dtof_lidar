#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 net.py 开/关 QUANT_INT16 在 cali_data 上的差异。

只调用 Network.forward 的输出 (dist/conf/peak/reflectance/snr)，
不复制 net.py 内部公式。net 改了门控或计算，这里自动跟着变。

1) 以 float 的 conf 为基准，统计 quant 漏检 / 误检
2) 双方都有效的像素上，比较 net 算出的距离 / 反射率 / SNR

用法:
  py compare_quant.py
  py compare_quant.py cali_data --out tmp/compare_quant
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import net as net_module

TOF_H, TOF_W, TOF_C = 30, 40, 64
BATCH = 16
UPSCALE = 14
CONF_THR = 0.5
N_WORST = 3
N_POINTS = 2
HIST_BINS = 62
MARK_BGR = [(0, 0, 255), (0, 165, 255)]
MARK_MPL = ["#ef4444", "#f59e0b"]


def _setup_mpl() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Segoe UI", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _try_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_raws(src: Path) -> tuple[list[str], np.ndarray]:
    files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".raw")
    if not files:
        raise SystemExit(f"未找到 .raw: {src}")
    n = TOF_H * TOF_W * TOF_C
    hists = []
    names = []
    for p in files:
        raw = np.frombuffer(p.read_bytes(), dtype=np.uint16)
        if raw.size < n:
            print(f"[WARN] 跳过 {p.name}: {raw.size} < {n}")
            continue
        hists.append(raw[-n:].reshape(TOF_H, TOF_W, TOF_C).astype(np.float32, copy=False))
        names.append(p.name)
    return names, np.stack(hists, axis=0)


def infer(model: net_module.Network, hists: np.ndarray, quant: bool) -> dict[str, np.ndarray]:
    net_module.QUANT_INT16 = bool(quant)
    n = hists.shape[0]
    out = {k: np.zeros((n, TOF_H, TOF_W), dtype=np.float32) for k in ("dist", "conf", "refl", "snr")}
    with torch.inference_mode():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            d, c, _p, r, sn = model(torch.from_numpy(hists[s:e]).permute(0, 3, 1, 2))
            out["dist"][s:e] = d[:, 0].cpu().numpy()
            out["conf"][s:e] = c[:, 0].cpu().numpy()
            out["refl"][s:e] = r[:, 0].cpu().numpy()
            out["snr"][s:e] = sn[:, 0].cpu().numpy()
            print(f"  [{'Q' if quant else 'F'}] {e}/{n}")
    return out


def _pct(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "max")}
    ad = np.abs(a)
    return {
        "mean": float(ad.mean()),
        "p50": float(np.percentile(ad, 50)),
        "p95": float(np.percentile(ad, 95)),
        "p99": float(np.percentile(ad, 99)),
        "max": float(ad.max()),
    }


def _rel_pct(quant: np.ndarray, flt: np.ndarray, valid: np.ndarray, eps: float) -> dict[str, float]:
    """共同有效且 |float|>eps 时，|quant/float - 1| 的分位数。"""
    m = valid & np.isfinite(quant) & np.isfinite(flt) & (np.abs(flt) > eps)
    if not np.any(m):
        return _pct(np.array([]))
    return _pct((quant[m] / flt[m] - 1.0) * 100.0)


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.4g}"
    return f"{x:.5g}"


def _print_block(title: str, rows: list[tuple[str, str]]) -> None:
    w = max(len(k) for k, _ in rows)
    print(f"\n=== {title} ===")
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")


def _stat_cell(x: float, unit: str) -> str:
    u = html.escape(unit)
    return f"<td><span class='v'>{html.escape(_fmt(x))}</span><span class='u'>{u}</span></td>"


def _print_stats(title: str, rows: list[tuple[str, str, dict[str, float]]]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'量':<16} {'mean':>12} {'':<3} {'P50':>12} {'':<3} {'P95':>12} {'':<3} {'P99':>12} {'':<3} {'max':>12}")
    for name, unit, s in rows:
        u = f"{unit:<3}"
        print(
            f"  {name:<16} {_fmt(s['mean']):>12} {u} {_fmt(s['p50']):>12} {u} "
            f"{_fmt(s['p95']):>12} {u} {_fmt(s['p99']):>12} {u} {_fmt(s['max']):>12} {u}"
        )


def _up(img: np.ndarray) -> np.ndarray:
    vis = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = vis.shape[:2]
    return cv2.resize(vis, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_NEAREST)


def _xy(px: int, py: int) -> tuple[int, int]:
    oy, ox = int(px), int(TOF_H - 1 - py)
    return ox * UPSCALE + UPSCALE // 2, oy * UPSCALE + UPSCALE // 2


def _mark(img: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    vis = _up(img)
    for i, (py, px) in enumerate(points):
        x, y = _xy(px, py)
        color = MARK_BGR[i % len(MARK_BGR)]
        cv2.circle(vis, (x, y), UPSCALE + 2, color, 2, cv2.LINE_AA)
        cv2.putText(vis, chr(ord("A") + i), (x + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def _color_depth(dist: np.ndarray, valid: np.ndarray) -> np.ndarray:
    d = np.asarray(dist, dtype=np.float32)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        if hi - lo < 0.3:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.15, mid + 0.15
        u8[valid] = np.clip(np.rint(255.0 * (1.0 - (d[valid] - lo) / (hi - lo))), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _valid_xor(fv: np.ndarray, qv: np.ndarray) -> np.ndarray:
    bgr = np.full((TOF_H, TOF_W, 3), 28, dtype=np.uint8)
    bgr[fv & qv] = (60, 60, 60)
    bgr[fv & ~qv] = (220, 80, 40)
    bgr[~fv & qv] = (40, 200, 40)
    return bgr


def _pick(mask: np.ndarray, n: int) -> list[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    step = max(1, ys.size // n)
    idx = list(range(0, ys.size, step))[:n]
    return [(int(ys[i]), int(xs[i])) for i in idx]


def pick_valid_points(miss: np.ndarray, fp: np.ndarray) -> list[tuple[int, int]]:
    pts = _pick(miss, 1) + _pick(fp, 1)
    if len(pts) < N_POINTS:
        for p in _pick(miss | fp, N_POINTS):
            if p not in pts:
                pts.append(p)
            if len(pts) >= N_POINTS:
                break
    return pts[:N_POINTS]


def pick_value_points(d_refl: np.ndarray, d_snr: np.ndarray, both: np.ndarray) -> list[tuple[int, int]]:
    if not both.any():
        return []
    ar = np.abs(d_refl)
    as_ = np.abs(d_snr)
    ar[~both] = -1.0
    as_[~both] = -1.0
    p1 = tuple(int(v) for v in np.unravel_index(int(np.argmax(ar)), ar.shape))
    as_[p1] = -1.0
    p2 = tuple(int(v) for v in np.unravel_index(int(np.argmax(as_)), as_.shape))
    if p2 == p1 or not both[p2]:
        ar[p1] = -1.0
        p2 = tuple(int(v) for v in np.unravel_index(int(np.argmax(ar)), ar.shape))
    return [p1, p2]


def _plot_hist(ax, hist62: np.ndarray, letter: str, py: int, px: int) -> None:
    ax.bar(np.arange(HIST_BINS), hist62[:HIST_BINS], width=0.9, color="#94a3b8", edgecolor="none")
    ax.set_title(f"{letter}  ({py},{px})", fontsize=10, color=MARK_MPL[0 if letter == "A" else 1])
    ax.set_xlabel("bin")
    ax.set_ylabel("count")
    ax.set_xlim(-0.5, HIST_BINS - 0.5)


def _ok(v: bool) -> str:
    return "过" if v else "不过"


def _near(val: float, thr: float) -> bool:
    return val > thr and (val - thr) / max(abs(thr), 1e-9) < 0.2


def _draw_out_table(ax, title: str, f: dict, q: dict, py: int, px: int) -> None:
    """conf 只写 float；距离 / 反射率 / SNR 同时写 float 和 quant。"""
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2)
    fc = float(f["conf"][py, px])
    fr, qr = float(f["refl"][py, px]), float(q["refl"][py, px])
    fs, qs = float(f["snr"][py, px]), float(q["snr"][py, px])
    fd, qd = float(f["dist"][py, px]), float(q["dist"][py, px])
    refl_thr = float(net_module.REFLECT_THRESH)
    snr_thr = float(net_module.SNR_THRESH)
    rows = [
        ["conf", f"{fc:.0f}", _ok(fc > CONF_THR), "—", f"> {CONF_THR:g}"],
        ["距离 cm", _fmt(fd * 100.0), "—", _fmt(qd * 100.0), "—"],
        ["反射率 %", _fmt(fr * 100.0), _ok(fr > refl_thr), _fmt(qr * 100.0), f"> {refl_thr * 100.0:g}"],
        ["SNR 倍", _fmt(fs), _ok(fs > snr_thr), _fmt(qs), f"> {snr_thr:g}"],
    ]
    near = [False, False, _near(fr, refl_thr), _near(fs, snr_thr)]
    tab = ax.table(
        cellText=rows,
        colLabels=["net输出", "float", "过?", "quant", "阈值"],
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.08, 1.6)
    paint = {"过": "#bbf7d0", "不过": "#fecaca", "—": "#e5e7eb"}
    for j in range(5):
        tab[(0, j)].set_facecolor("#1f2937")
        tab[(0, j)].set_text_props(color="white", weight="bold")
    for i, row in enumerate(rows, 1):
        tab[(i, 2)].set_facecolor(paint.get(row[2], "#fff"))
        if near[i - 1]:
            for j in range(5):
                tab[(i, j)].set_facecolor("#fde68a")
            tab[(i, 2)].set_text_props(weight="bold")
            tab[(i, 2)].get_text().set_text("过(临界)")


def save_frame(
    path: Path,
    title: str,
    hist: np.ndarray,
    f: dict,
    q: dict,
    points: list[tuple[int, int]],
    left_img: np.ndarray,
    mid_img: np.ndarray,
    right_img: np.ndarray | None,
    left_t: str,
    mid_t: str,
    right_t: str,
) -> None:
    fig = plt.figure(figsize=(12.6, 9.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 0.95], hspace=0.28, wspace=0.16)
    panels = [(gs[0, 0], left_img, left_t), (gs[0, 1], mid_img, mid_t)]
    if right_img is not None:
        panels.append((gs[0, 2], right_img, right_t))
    for spec, img, t in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(cv2.cvtColor(_mark(img, points), cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for i, (py, px) in enumerate(points):
        _plot_hist(fig.add_subplot(gs[1, i]), hist[py, px], chr(ord("A") + i), py, px)
        _draw_out_table(fig.add_subplot(gs[2, i]), f"{chr(ord('A') + i)}  net 输出", f, q, py, px)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def write_html(
    out_dir: Path,
    src: Path,
    n: int,
    valid_rows: list[tuple[str, str]],
    value_stats: list[tuple[str, str, dict[str, float]]],
    valid_figs: list[Path],
    value_figs: list[Path],
) -> Path:
    def kv_table(rows: list[tuple[str, str]]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in rows)
        return f"<table>{body}</table>"

    def stats_table(rows: list[tuple[str, str, dict[str, float]]]) -> str:
        head = "<tr><th>量</th><th>mean</th><th>P50</th><th>P95</th><th>P99</th><th>max</th></tr>"
        body = "".join(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"{_stat_cell(s['mean'], unit)}{_stat_cell(s['p50'], unit)}"
            f"{_stat_cell(s['p95'], unit)}{_stat_cell(s['p99'], unit)}{_stat_cell(s['max'], unit)}"
            "</tr>"
            for name, unit, s in rows
        )
        return f"<table class='stats'><thead>{head}</thead><tbody>{body}</tbody></table>"

    def figs(paths: list[Path]) -> str:
        return "".join(
            f"<figure><img src='{html.escape(p.name)}'/><figcaption>{html.escape(p.stem)}</figcaption></figure>"
            for p in paths
        )

    text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>compare_quant</title>
<style>
body{{margin:0;background:#f4f6f9;color:#172033;font:14px/1.55 "Microsoft YaHei",sans-serif}}
main{{max-width:1140px;margin:auto;padding:28px 20px 48px}}
h1{{margin:0 0 6px}} .sub{{color:#61708a}}
.panel{{background:#fff;border:1px solid #dfe5ee;border-radius:10px;padding:14px 16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse}} td,th{{padding:8px 10px;border-bottom:1px solid #edf1f6;text-align:right}}
td:first-child,th:first-child{{color:#61708a;text-align:left}}
th{{color:#61708a;font-weight:600}}
table.stats{{table-layout:fixed}}
table.stats th:first-child,table.stats td:first-child{{width:22%}}
table.stats .v{{font-variant-numeric:tabular-nums;font-family:Consolas,"Cascadia Mono",monospace}}
table.stats .u{{display:inline-block;width:2.2em;margin-left:.28em;text-align:left;color:#8a94a6}}
img{{width:100%;border-radius:8px;background:#111}} figure{{margin:14px 0}}
</style></head><body><main>
<h1>量化 vs 非量化</h1>
<p class="sub">{html.escape(str(src))} · {n} 帧 · 全部数值来自 net.Network.forward</p>
<section class="panel"><h2>1. 共同有效点 |Δ|（net dist / reflectance / snr）</h2>{stats_table(value_stats)}</section>
<section class="panel"><h2>2. 有效点（net conf）</h2>{kv_table(valid_rows)}</section>
<section class="panel"><h2>漏检/误检最严重的 3 帧</h2>{figs(valid_figs)}</section>
<section class="panel"><h2>反射率/SNR 差异最严重的 3 帧</h2>{figs(value_figs)}</section>
</main></body></html>
"""
    path = out_dir / "index.html"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 net 量化 / 非量化")
    parser.add_argument("source", nargs="?", default="cali_data", type=Path)
    parser.add_argument("--out", default=Path("tmp/compare_quant"), type=Path)
    args = parser.parse_args()
    _try_utf8_stdout()
    _setup_mpl()

    src = args.source.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    names, hists = load_raws(src)
    n = len(names)
    print(f"[INFO] {src.name}: {n} 帧")

    model = net_module.Network().eval()
    try:
        print("[INFO] float 推理")
        fo = infer(model, hists, False)
        print("[INFO] quant 推理")
        qo = infer(model, hists, True)
    finally:
        net_module.QUANT_INT16 = True

    fv = fo["conf"] > CONF_THR
    qv = qo["conf"] > CONF_THR
    miss, fp, both = fv & ~qv, ~fv & qv, fv & qv
    n_float, n_miss, n_fp, n_both = int(fv.sum()), int(miss.sum()), int(fp.sum()), int(both.sum())
    frame_miss = miss.reshape(n, -1).sum(1)
    frame_fp = fp.reshape(n, -1).sum(1)
    valid_score = frame_miss.astype(np.float64) * 3.0 + frame_fp.astype(np.float64)
    valid_idx = [int(i) for i in np.argsort(-valid_score) if valid_score[i] > 0][:N_WORST]

    d_dist_cm = (qo["dist"] - fo["dist"]) * 100.0
    d_refl_pct = (qo["refl"] - fo["refl"]) * 100.0
    d_snr = qo["snr"] - fo["snr"]
    s_dist = _pct(d_dist_cm[both])
    s_refl = _pct(d_refl_pct[both])
    s_snr = _pct(d_snr[both])
    s_refl_rel = _rel_pct(qo["refl"], fo["refl"], both, 1e-6)
    s_snr_rel = _rel_pct(qo["snr"], fo["snr"], both, 1e-3)

    frame_score = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m = both[i]
        if m.any():
            frame_score[i] = float(np.abs(d_refl_pct[i][m]).max() + np.abs(d_snr[i][m]).max())
    value_idx = [int(i) for i in np.argsort(-frame_score) if frame_score[i] > 0][:N_WORST]

    valid_rows = [
        ("float 有效", f"{n_float} / {fv.size}  ({100.0 * n_float / fv.size:.4f}%)"),
        ("漏检 float有效/quant无效", f"{n_miss}  ({100.0 * n_miss / max(n_float, 1):.4f}% of float有效)"),
        ("误检 float无效/quant有效", f"{n_fp}"),
        ("双方都有效", f"{n_both}"),
    ]
    value_stats = [
        ("距离 |Δ|", "cm", s_dist),
        ("反射率 |Δ|", "%", s_refl),
        ("反射率 |q/f-1|", "%", s_refl_rel),
        ("SNR |Δ|", "", s_snr),
        ("SNR |q/f-1|", "%", s_snr_rel),
    ]
    _print_stats("1. 共同有效 |Δ| (net 输出)", value_stats)
    _print_block("2. 有效点 (net conf)", valid_rows)
    for i in valid_idx:
        print(f"    {names[i]}  漏检={int(frame_miss[i])}  误检={int(frame_fp[i])}")
    for i in value_idx:
        print(f"    {names[i]}  score={_fmt(frame_score[i])}")

    valid_figs: list[Path] = []
    for rank, i in enumerate(valid_idx, 1):
        pts = pick_valid_points(miss[i], fp[i])
        path = out_dir / f"valid_{rank:02d}_{Path(names[i]).stem}.png"
        save_frame(
            path,
            f"有效点 #{rank}  {names[i]}  漏检={int(frame_miss[i])} 误检={int(frame_fp[i])}",
            hists[i],
            {k: v[i] for k, v in fo.items()},
            {k: v[i] for k, v in qo.items()},
            pts,
            _color_depth(fo["dist"][i], fv[i]),
            _color_depth(qo["dist"][i], qv[i]),
            _valid_xor(fv[i], qv[i]),
            "float 距离(conf=1)",
            "quant 距离(conf=1)",
            "蓝=漏检  绿=误检",
        )
        valid_figs.append(path)
        print(f"  [valid #{rank}] {names[i]}  {pts}")

    value_figs: list[Path] = []
    for rank, i in enumerate(value_idx, 1):
        pts = pick_value_points(d_refl_pct[i], d_snr[i], both[i])
        path = out_dir / f"value_{rank:02d}_{Path(names[i]).stem}.png"
        save_frame(
            path,
            f"反射率/SNR #{rank}  {names[i]}",
            hists[i],
            {k: v[i] for k, v in fo.items()},
            {k: v[i] for k, v in qo.items()},
            pts,
            _color_depth(fo["dist"][i], both[i]),
            _color_depth(qo["dist"][i], both[i]),
            None,
            "float 距离",
            "quant 距离",
            "",
        )
        value_figs.append(path)
        print(f"  [value #{rank}] {names[i]}  {pts}")

    html_path = write_html(out_dir, src, n, valid_rows, value_stats, valid_figs, value_figs)
    print(f"\n[OK] {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
