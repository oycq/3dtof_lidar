#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TOF_H = 30
TOF_W = 40
TOF_C = 64
TOTAL_PIXELS = TOF_H * TOF_W
SHOW_W = 390
SHOW_H = 520
HEADER_H = 36

# 固定配置（按需求不走命令行）
CONF_THR = 0.5
REL_ERR_THR = 0.07
EPS = 1e-6
DEPTH_NEAR_M = 1.0
DEPTH_FAR_M = 30.0
DEPTH_GAMMA = 1.6


@dataclass
class SceneStats:
    scene_name: str
    input_path: Path
    output_path: Path
    wrong_ratio: float
    wrong_count: int
    conf_count: int
    total_valid: int
    gt_mean: float
    pred_mean: float
    abs_err_mean: float
    gt_map: np.ndarray
    pred_map: np.ndarray
    conf_mask: np.ndarray
    wrong_mask: np.ndarray


def find_pairs(train_dir: Path) -> list[tuple[Path, Path]]:
    inputs = sorted(train_dir.glob("input_*.npy"))
    pairs: list[tuple[Path, Path]] = []
    for ip in inputs:
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


def load_network(nn_dir: Path, ckpt_path: Path):
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    net.load_state_dict(sd, strict=True)
    return net, device


def natural_key(s: str) -> list[object]:
    parts = re.split(r"(\d+)", s)
    out: list[object] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def scene_name_from_input(ip: Path) -> str:
    name = ip.stem
    if name.startswith("input_"):
        return name[len("input_") :]
    return name


def print_progress(prefix: str, idx: int, total: int) -> None:
    total_safe = max(int(total), 1)
    done = int(idx)
    ratio = float(done) / float(total_safe)
    width = 30
    fill = int(round(ratio * width))
    bar = "#" * fill + "-" * (width - fill)
    print(f"\r{prefix} [{bar}] {done}/{total_safe} ({ratio * 100.0:5.1f}%)", end="", flush=True)
    if done >= total_safe:
        print("")


def _rotate_for_display(img: np.ndarray):
    import cv2  # type: ignore

    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)


def _draw_text(img: np.ndarray, s: str, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    import cv2  # type: ignore

    out = img.copy()
    cv2.putText(out, s, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def _color_depth(depth: np.ndarray) -> np.ndarray:
    # 配色和 check.py 保持一致：inv-depth + turbo + gamma
    import cv2  # type: ignore

    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    if np.any(valid):
        dc = np.clip(d[valid], DEPTH_NEAR_M, DEPTH_FAR_M)
        inv = 1.0 / np.clip(dc, EPS, np.inf)
        inv_n = 1.0 / float(DEPTH_NEAR_M)
        inv_f = 1.0 / float(DEPTH_FAR_M)
        n = (inv - inv_f) / max(inv_n - inv_f, EPS)
        n = np.power(np.clip(n, 0.0, 1.0), 1.0 / float(DEPTH_GAMMA))
        u8[valid] = np.clip(np.rint(n * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))
    bgr[~valid] = (0, 0, 0)
    return bgr


def _wrong_mask_bgr(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    out[mask] = (0, 0, 255)
    return out


def save_scene_image(item: SceneStats, rank_idx: int, out_dir: Path) -> None:
    import cv2  # type: ignore

    gt_bgr = _color_depth(item.gt_map)
    pred_raw_bgr = _color_depth(item.pred_map)
    pred_raw_bgr[~item.conf_mask] = (0, 0, 0)  # conf<50% 不参与绘制
    pred_mark_bgr = pred_raw_bgr.copy()
    pred_mark_bgr[item.wrong_mask] = (0, 0, 255)  # 错误像素标红

    pred_raw_show = cv2.resize(_rotate_for_display(pred_raw_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    gt_show = cv2.resize(_rotate_for_display(gt_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    pred_mark_show = cv2.resize(_rotate_for_display(pred_mark_bgr), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

    pred_raw_show = _draw_text(pred_raw_show, "PRED_RAW")
    gt_show = _draw_text(gt_show, "GT")
    pred_mark_show = _draw_text(pred_mark_show, "PRED_RED(wrong, conf>=50%)", (0, 0, 255))

    view = np.hstack([pred_raw_show, gt_show, pred_mark_show])
    header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
    header_text = (
        f"rank {rank_idx + 1}  scene {item.scene_name}  "
        f"wrong_rate {item.wrong_ratio:.6f} ({item.wrong_count}/{TOTAL_PIXELS})  "
        f"gt {item.gt_mean:.3f}m  pred {item.pred_mean:.3f}m  abs_err {item.abs_err_mean:.3f}m"
    )
    header = _draw_text(header, header_text)
    canvas = np.vstack([header, view])

    safe_scene = re.sub(r"[^\w\-\.]+", "_", item.scene_name)
    out_path = out_dir / f"{rank_idx + 1:03d}_{safe_scene}.png"
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"failed to write image: {out_path}")


def main() -> int:
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    here = Path(__file__).resolve().parent
    nn_dir = here.parent
    train_dir = nn_dir / "train_data"
    ckpt_path = nn_dir / "model_last.pt"
    out_dir = here / "rank"

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    pairs = find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    net, device = load_network(nn_dir=nn_dir, ckpt_path=ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧结果，避免上次遗留图片干扰排序观察。
    for p in out_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass
    old_csv = out_dir / "ranking.csv"
    if old_csv.exists():
        try:
            old_csv.unlink()
        except OSError:
            pass

    rows: list[SceneStats] = []
    conf_thr = float(CONF_THR)
    rel_err_thr = float(REL_ERR_THR)

    with torch.no_grad():
        total_pairs = len(pairs)
        for idx, (ip, op) in enumerate(pairs, start=1):
            x, gt = load_pair(ip, op)
            inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            out = net.forward_train(inp)
            pred = out["dist"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
            conf = out["conf"][0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

            valid = np.isfinite(gt) & np.isfinite(pred) & np.isfinite(conf) & (gt > 0.0)
            total_valid = int(np.count_nonzero(valid))
            conf_pos = valid & (conf >= conf_thr)  # conf<50% 不参与统计
            abs_err_map = np.abs(pred - gt).astype(np.float32, copy=False)
            rel_err_map = np.zeros_like(abs_err_map, dtype=np.float32)
            rel_err_map[conf_pos] = abs_err_map[conf_pos] / np.clip(np.abs(gt[conf_pos]), EPS, np.inf)
            wrong_mask = conf_pos & (rel_err_map > rel_err_thr)
            conf_count = int(np.count_nonzero(conf_pos))
            wrong_count = int(np.count_nonzero(wrong_mask))
            wrong_ratio = float(wrong_count) / float(TOTAL_PIXELS)

            scene_name = scene_name_from_input(ip)
            gt_v = gt[conf_pos]
            pred_v = pred[conf_pos]
            err_v = abs_err_map[conf_pos]
            if total_valid == 0 and conf_count == 0:
                continue
            rows.append(
                SceneStats(
                    scene_name=scene_name,
                    input_path=ip,
                    output_path=op,
                    wrong_ratio=wrong_ratio,
                    wrong_count=wrong_count,
                    conf_count=conf_count,
                    total_valid=total_valid,
                    gt_mean=float(np.mean(gt_v)) if gt_v.size else float("nan"),
                    pred_mean=float(np.mean(pred_v)) if pred_v.size else float("nan"),
                    abs_err_mean=float(np.mean(err_v)) if err_v.size else float("nan"),
                    gt_map=gt,
                    pred_map=pred,
                    conf_mask=conf_pos,
                    wrong_mask=wrong_mask,
                )
            )
            print_progress("统计进度", idx, total_pairs)

    if not rows:
        raise RuntimeError("no valid scene found after filtering")

    rows.sort(
        key=lambda r: (
            -r.wrong_ratio,
            -r.wrong_count,
            natural_key(r.scene_name),
        )
    )
    # 校验：rank 越往后，错误率不升高（允许持平）。
    for i in range(1, len(rows)):
        if rows[i].wrong_ratio > rows[i - 1].wrong_ratio:
            raise RuntimeError(
                "rank order invalid: later rank has higher wrong_ratio "
                f"(rank={i+1}, prev={rows[i-1].wrong_ratio:.8f}, curr={rows[i].wrong_ratio:.8f})"
            )

    total_ranked = len(rows)
    for i, item in enumerate(rows, start=1):
        save_scene_image(item=item, rank_idx=i - 1, out_dir=out_dir)
        print_progress("出图进度", i, total_ranked)

    csv_path = out_dir / "ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "scene_name",
                "wrong_ratio_wrong_over_1200",
                "wrong_count",
                "conf_count",
                "valid_count",
                "gt_mean_m",
                "pred_mean_m",
                "abs_err_mean_m",
                "input_file",
                "output_file",
            ]
        )
        for i, item in enumerate(rows, start=1):
            writer.writerow(
                [
                    i,
                    item.scene_name,
                    f"{item.wrong_ratio:.6f}",
                    item.wrong_count,
                    item.conf_count,
                    item.total_valid,
                    f"{item.gt_mean:.6f}" if math.isfinite(item.gt_mean) else "nan",
                    f"{item.pred_mean:.6f}" if math.isfinite(item.pred_mean) else "nan",
                    f"{item.abs_err_mean:.6f}" if math.isfinite(item.abs_err_mean) else "nan",
                    item.input_path.name,
                    item.output_path.name,
                ]
            )

    print(f"[done] pairs={len(pairs)}")
    print(f"[done] ranked scenes={len(rows)}")
    print(f"[done] rank dir: {out_dir}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] conf threshold: >= {conf_thr:.2f}")
    print(f"[done] wrong criterion: |pred-gt|/gt > {rel_err_thr * 100.0:.2f}%")
    print(f"[done] wrong ratio definition: wrong_count/{TOTAL_PIXELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

