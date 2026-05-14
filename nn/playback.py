#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/playback.py

回放 nn/tmp/ 下的 *.raw 文件，按文件名（时间戳）排序，最新在前。

交互：
- 启动时显示最新的一张（idx=0）
- 左方向键：上一张（更老的，idx+1）
- 右方向键：下一张（更新的，idx-1）
- 鼠标悬停：显示 dist/snr/conf/peak/reflectance
- ESC / Q 退出
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 复用 realtime.py 的解析 / 推理 / 渲染 / 常量
from realtime import (
    HEADER_H,
    ROTATE_90,
    TOF_C,
    TOF_H,
    TOF_W,
    _get_show_size,
    _render_view,
    _run_infer,
    _with_text,
    tof_histograms_from_u16,
)


TMP_DIR = Path(__file__).resolve().parent / "tmp"


def _scan_raw_files(tmp_dir: Path) -> list[Path]:
    """扫描 tmp 下的 .raw 文件，按文件名倒序（最新在前）。"""
    if not tmp_dir.exists():
        return []
    files = [p for p in tmp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".raw"]
    files.sort(key=lambda p: p.name, reverse=True)
    return files


def _load_hist_from_raw(path: Path) -> np.ndarray | None:
    """读取 .raw 并解析为 (30,40,64) 直方图，失败返回 None。"""
    try:
        raw_bytes = path.read_bytes()
    except Exception as exc:
        print(f"[ERR] 读取失败 {path.name}: {exc}")
        return None
    raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    hists = tof_histograms_from_u16(raw_u16)
    if hists.shape != (TOF_H, TOF_W, TOF_C):
        print(f"[ERR] shape 不匹配 {path.name}: {hists.shape}")
        return None
    return hists


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from exc

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("missing dependency torch") from exc

    cv2.setUseOptimized(True)

    files = _scan_raw_files(TMP_DIR)
    if not files:
        print(f"[ERR] {TMP_DIR} 下没有 .raw 文件")
        return 1
    print(f"[INFO] 找到 {len(files)} 个 .raw 文件，最新: {files[0].name}")

    nn_dir = Path(__file__).resolve().parent
    root = nn_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))

    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    rotate_90 = bool(ROTATE_90)
    show_w, show_h = _get_show_size(rotate_90)

    win = "NN_PLAYBACK"
    hist_win = "HIST"
    strip_win = "BINS"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(hist_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(strip_win, cv2.WINDOW_AUTOSIZE)

    mouse: dict = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) == int(cv2.EVENT_MOUSEMOVE):
            mouse["x"] = int(x)
            mouse["y"] = int(y)

    cv2.setMouseCallback(win, on_mouse)

    n = len(files)
    idx = 0

    # 缓存: idx -> (hist, dist, snr, conf, peak, refl)
    cache: dict[int, tuple] = {}

    def get_frame(i: int):
        if i in cache:
            return cache[i]
        hists = _load_hist_from_raw(files[i])
        if hists is None:
            return None
        d, s, c, p, r = _run_infer(net, device, hists.astype(np.float32))
        cache[i] = (hists.astype(np.float32), d, s, c, p, r)
        return cache[i]

    last_idx = -1
    last_mouse_xy = (-1, -1)
    view_cache = None
    hist_cache = None
    strip_cache = None

    print("[INFO] LEFT=上一张(更老)  RIGHT=下一张(更新)  ESC/Q=退出")

    while True:
        entry = get_frame(idx)
        if entry is None:
            print(f"[WARN] 跳过无效帧: {files[idx].name}")
            if idx < n - 1:
                idx += 1
                continue
            return 1
        hists_f, dist, snr, conf, peak, refl = entry

        mouse_xy = (int(mouse["x"]), int(mouse["y"]))
        need_redraw = (idx != last_idx) or (mouse_xy != last_mouse_xy) or (view_cache is None)

        if need_redraw:
            view, hist_img, strip_img, px, py = _render_view(
                dist, conf, peak, refl, hists_f,
                mouse_xy[0], mouse_xy[1], show_w, show_h, rotate_90,
            )
            hover1 = (
                f"[{idx + 1}/{n}] {files[idx].name}  "
                f"dist {float(dist[py, px]):.3f}m  snr {float(snr[py, px]):.3f}  "
                f"conf {float(conf[py, px]):.0f}  peak {float(peak[py, px]):.3f}"
            )
            hover2 = f"reflectance {float(refl[py, px]) * 100.0:.3f}%   (LEFT=older  RIGHT=newer)"
            _with_text(view[:HEADER_H], hover1, y=22)
            _with_text(view[:HEADER_H], hover2, y=46)

            view_cache = view
            hist_cache = hist_img
            strip_cache = strip_img
            last_idx = idx
            last_mouse_xy = mouse_xy

        cv2.imshow(win, view_cache)
        cv2.imshow(hist_win, hist_cache)
        cv2.imshow(strip_win, strip_cache)

        key = cv2.waitKeyEx(20)
        if key == -1:
            continue
        if key in (27, ord("q"), ord("Q")):
            break
        # 左键: 上一张(更老) -> idx + 1
        if key in (2424832, ord("a"), ord("A")):
            if idx < n - 1:
                idx += 1
        # 右键: 下一张(更新) -> idx - 1
        elif key in (2555904, ord("d"), ord("D")):
            if idx > 0:
                idx -= 1

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
