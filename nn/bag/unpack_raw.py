#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unpack_raw.py

从 BAG/MCAP 读取 sensor/vp_tof_info 的 VpTofInfo 原始 64-bin 直方图，
用 nn/net.py 的规则网络计算 dist/conf/peak/reflectance/snr 五通道，
UI 完全与 realtime.py 一致（四宫格 + 直方图窗口）。

依赖：pip install mcap numpy opencv-python torch
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from mcap.exceptions import EndOfFile
from mcap.reader import NonSeekingReader, make_reader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from net import (
    IS_6321, DIST_SCALE_M, TAIL_BASE, PULSES,
    REFLECT_K, DIST_BIAS, REFLECT_THRESH, SNR_THRESH,
    ARGMAX_CLIP_MIN, ARGMAX_CLIP_MAX, NOISE_BIAS,
)

# ======================== ToF 常量 ========================
TOF_H = 30
TOF_W = 40
BIN_NUM = 64
PIXELS = TOF_H * TOF_W

# ======================== VpTofInfo 结构 ========================
HEADER_FMT = "<B3xIQB1xHIIfff"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 40
RAW_U16_COUNT = 2 * PIXELS * BIN_NUM
RAW_BYTES = RAW_U16_COUNT * 2
RESERVED_BYTES = 8 * 2
PAYLOAD_SIZE = HEADER_SIZE + RAW_BYTES + RESERVED_BYTES

# ======================== 配置 ========================
BAG_NAME = "1.bag"
RAW_TOPIC = "sensor/vp_tof_info"

# ======================== 显示参数（与 realtime.py 对齐） ========================
SHOW_W = 520
SHOW_H = 390
HEADER_H = 56
EPS = 1e-6
PLAY_HZ = 10.0

# ======================== 直方图窗口参数（与 realtime.py 对齐） ========================
HIST_W = 640
HIST_H = 280


# ======================== Network ========================
class NetworkRaw(nn.Module):
    def __init__(self):
        super().__init__()
        hist_bias = torch.tensor([80.0] + [0.0] * 61, dtype=torch.float32).view(1, 62, 1, 1)
        self.register_buffer("hist_bias", hist_bias)

    def forward(self, x):
        hist, raw_bin_63, raw_bin_64 = torch.split(x, [62, 1, 1], dim=1)
        hist = torch.relu(hist - self.hist_bias)

        peak_idx = torch.argmax(hist, dim=1, keepdim=True)
        peak_idx = torch.clamp(peak_idx, min=ARGMAX_CLIP_MIN, max=ARGMAX_CLIP_MAX)
        a = torch.gather(hist, dim=1, index=peak_idx - 1)
        b = torch.gather(hist, dim=1, index=peak_idx)
        c = torch.gather(hist, dim=1, index=peak_idx + 1)
        centroid = (c - a) / (a + b + c) + peak_idx
        dist = centroid * DIST_SCALE_M + DIST_BIAS

        if IS_6321:
            sat_value = raw_bin_63 * TAIL_BASE + raw_bin_64
        else:
            sat_value = raw_bin_64 * TAIL_BASE + raw_bin_63
            sat_value = torch.where(sat_value > 0, sat_value, torch.full_like(sat_value, PULSES))
        k = PULSES / sat_value

        mean_val = torch.mean(hist, dim=1, keepdim=True) * k
        peak_val = torch.max(hist, dim=1, keepdim=True).values * k
        signal = peak_val - mean_val
        noise = torch.sqrt(mean_val) + NOISE_BIAS
        snr = signal / noise

        reflectance = dist * dist * signal / REFLECT_K
        conf = ((snr > SNR_THRESH) & (reflectance > REFLECT_THRESH)).to(torch.float32)

        dist = torch.where(conf == 0, torch.zeros_like(dist), dist)

        return dist, conf, peak_val, reflectance, snr


# ======================== 数据结构 ========================
@dataclass
class TofInfoHeader:
    is_valid: int
    frame_id: int
    timestamp_us: int
    work_mode: int
    bin_mode: int
    light_count: int
    expo_time: int
    pulse_width: float
    rx_temp: float
    tx_temp: float


@dataclass
class RawFrameData:
    src_file: str
    seq_in_file: int
    header: TofInfoHeader
    hist: np.ndarray  # (30,40,64) uint16


@dataclass
class ComputedFrameData:
    raw: RawFrameData
    dist: np.ndarray       # (30,40) float32 m
    conf: np.ndarray       # (30,40) float32 0/1
    peak: np.ndarray       # (30,40) float32
    reflect: np.ndarray    # (30,40) float32
    snr: np.ndarray        # (30,40) float32


# ======================== BAG 解析 ========================
def parse_vp_tof_info(payload: bytes):
    if len(payload) < PAYLOAD_SIZE:
        return None
    vals = struct.unpack_from(HEADER_FMT, payload, 0)
    header = TofInfoHeader(
        is_valid=vals[0], frame_id=vals[1], timestamp_us=vals[2],
        work_mode=vals[3], bin_mode=vals[4], light_count=vals[5],
        expo_time=vals[6], pulse_width=vals[7], rx_temp=vals[8], tx_temp=vals[9],
    )
    raw_all = np.frombuffer(payload[HEADER_SIZE:HEADER_SIZE + RAW_BYTES], dtype="<u2")
    hist = raw_all[:PIXELS * BIN_NUM].reshape(TOF_H, TOF_W, BIN_NUM).copy()
    return header, hist


def load_raw_frames(bag_path: Path):
    frames: list[RawFrameData] = []
    print(f"[INFO] 扫描: {bag_path}")
    cnt = 0

    def consume(msg_iter):
        nonlocal cnt
        for _, channel, message in msg_iter:
            if (channel.topic or "") != RAW_TOPIC:
                continue
            cnt += 1
            parsed = parse_vp_tof_info(message.data)
            if parsed:
                header, hist = parsed
                frames.append(RawFrameData(
                    src_file=bag_path.name, seq_in_file=cnt - 1,
                    header=header, hist=hist,
                ))

    with bag_path.open("rb") as f:
        try:
            consume(make_reader(f).iter_messages())
        except EndOfFile:
            print(f"[WARN] {bag_path.name}: 文件尾截断，顺序读取")
            f.seek(0)
            try:
                consume(NonSeekingReader(f).iter_messages(log_time_order=False))
            except EndOfFile:
                pass
    print(f"[OK] {bag_path.name}: topic帧={cnt}, 有效帧={len(frames)}")
    return frames


# ======================== 批量推理 ========================
def run_network(frames: list[RawFrameData], batch_size: int = 128):
    net = NetworkRaw()
    net.eval()
    n = len(frames)
    print(f"[INFO] 批量推理 {n} 帧 (batch={batch_size})...")

    all_hist = np.stack(
        [f.hist.astype(np.float32).transpose(2, 0, 1) for f in frames], axis=0
    )
    all_tensor = torch.from_numpy(all_hist)

    d_l, c_l, p_l, r_l, s_l = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            d, c, pk, r, sn = net(all_tensor[s:e])
            d_l.append(d.squeeze(1).numpy())
            c_l.append(c.squeeze(1).numpy())
            p_l.append(pk.squeeze(1).numpy())
            r_l.append(r.squeeze(1).numpy())
            s_l.append(sn.squeeze(1).numpy())
            if e % 1000 < batch_size or e == n:
                print(f"  [{e}/{n}]")

    print("[OK] 推理完成")
    return (np.concatenate(d_l), np.concatenate(c_l),
            np.concatenate(p_l), np.concatenate(r_l), np.concatenate(s_l))


# ======================== 着色函数（与 realtime.py 对齐） ========================
def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    d_valid = np.maximum(d[valid], EPS)
    y = 1.8 / d_valid
    u8[valid] = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr



def _with_text(img_bgr: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    cv2.putText(img_bgr, text, (10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int) -> tuple[int, int]:
    dx = int(np.clip((int(px) + 0.5) * max(show_w, 1) / TOF_W, 0, max(show_w, 1) - 1))
    dy = int(np.clip((int(py) + 0.5) * max(show_h, 1) / TOF_H, 0, max(show_h, 1) - 1))
    return dx, dy


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int) -> tuple[int, int]:
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
    py = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    return px, py


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    xx = int(np.clip(x, 0, img_bgr.shape[1] - 1))
    yy = int(np.clip(y, 0, img_bgr.shape[0] - 1))
    cv2.circle(img_bgr, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img_bgr, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


# ======================== 直方图绘制（与 realtime.py _render_histogram_bgr 对齐） ========================
def _render_histogram_bgr(
    bins: np.ndarray,
    w: int = HIST_W,
    h: int = HIST_H,
) -> np.ndarray:
    b = np.asarray(bins, dtype=np.float32).reshape(-1)[:BIN_NUM]
    img = np.zeros((max(int(h), 1), max(int(w), 1), 3), dtype=np.uint8)
    if b.size <= 0:
        return img
    b_draw = b[:62]
    tail_63 = float(b[62]) if b.size > 62 else 0.0
    tail_64 = float(b[63]) if b.size > 63 else 0.0
    sat_value = tail_63 * TAIL_BASE + tail_64
    if sat_value == 0.0:
        sat_value = PULSES
    vmax_raw = float(np.max(b_draw)) if b_draw.size > 0 else 0.0
    vmax_eq_sat = vmax_raw * PULSES / sat_value

    x0, y0 = 14, 128
    x1, y1 = img.shape[1] - 10, img.shape[0] - 18
    vmax = 1.0 if (not np.isfinite(vmax_raw) or vmax_raw <= 0.0) else vmax_raw
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
    bar_w = max(int((x1 - x0) / max(b_draw.size, 1)), 1)
    for i, v in enumerate(b_draw):
        vv = float(v) if np.isfinite(v) and float(v) > 0.0 else 0.0
        hh = int(np.clip(vv / vmax, 0.0, 1.0) * (y1 - y0 - 1))
        xl = x0 + i * bar_w
        xr = min(xl + bar_w, x1)
        if xr <= xl:
            continue
        yt = y1 - hh
        cv2.rectangle(img, (xl, yt), (xr, y1), (255, 220, 0), -1)
        cv2.rectangle(img, (xl, yt), (xr, y1), (30, 30, 30), 1)
    img = _with_text(img, "RAW_HIST (only 0-61 bins)", y=24)
    img = _with_text(img, f"max={vmax_raw:.3f}", y=48)
    img = _with_text(img, f"sat_value={sat_value:.3f}", y=72)
    img = _with_text(img, f"max_eq_sat={vmax_eq_sat:.3f}", y=96)
    return img


# ======================== 交互式查看器 ========================
def run_viewer(frames: list[ComputedFrameData]) -> None:
    if not frames:
        print("[WARN] 没有帧")
        return

    win = "ONBOARD_RAW"
    hist_win = "HISTOGRAM"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(hist_win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", win, 0, len(frames) - 1, lambda _: None)

    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback(win, on_mouse)

    playing = False
    play_interval_ms = int(round(1000.0 / PLAY_HZ))
    next_play_ms = 0
    idx = 0

    while True:
        tb_pos = cv2.getTrackbarPos("frame", win)
        if tb_pos != idx and not playing:
            idx = max(0, min(tb_pos, len(frames) - 1))

        now_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if playing and now_ms >= next_play_ms:
            if idx < len(frames) - 1:
                idx += 1
            else:
                playing = False
            cv2.setTrackbarPos("frame", win, idx)
            next_play_ms = now_ms + play_interval_ms

        frame = frames[idx]
        dist = frame.dist
        conf = frame.conf
        peak = frame.peak
        reflect = frame.reflect
        snr = frame.snr

        # ---- 着色（与 realtime.py 完全一致） ----
        dist_for_disp = np.where(conf > 0.5, dist, 0.0)
        dist_big = cv2.resize(_colorize_depth(dist_for_disp), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

        peak_norm = peak / max(float(np.max(peak)), EPS)
        peak_u8 = np.clip(np.rint(np.clip(peak_norm, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        peak_bgr = cv2.cvtColor(
            cv2.resize(peak_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )

        refl_u8 = np.clip(np.rint(np.clip(reflect, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
        refl_bgr = cv2.cvtColor(
            cv2.resize(refl_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )

        empty_bgr = np.zeros((SHOW_H, SHOW_W, 3), dtype=np.uint8)

        # ---- 鼠标 hover -> 像素坐标 ----
        mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 2 - 1))
        my = int(np.clip(int(mouse.get("y", 0)) - int(HEADER_H), 0, SHOW_H * 2 - 1))
        tile_x0 = 0 if mx < SHOW_W else SHOW_W
        tile_y0 = 0 if my < SHOW_H else SHOW_H
        px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, SHOW_W, SHOW_H)
        dx, dy = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)

        # ---- 画 marker ----
        for img in [dist_big, peak_bgr, refl_bgr]:
            _draw_marker(img, dx, dy)

        # ---- 面板标签（与 realtime.py 完全一致） ----
        _with_text(dist_big, "DISTANCE")
        _with_text(peak_bgr, "PEAK")
        _with_text(refl_bgr, "REFLECTANCE")

        # ---- hover 信息（与 realtime.py 格式一致） ----
        hover1 = (
            f"idx={idx}  frame_id={frame.raw.header.frame_id}  "
            f"dist {float(dist[py, px]):.3f}m  snr {float(snr[py, px]):.3f}  "
            f"conf {float(conf[py, px]):.0f}  peak {float(peak[py, px]):.3f}"
        )
        hover2 = f"reflectance {float(reflect[py, px]) * 100.0:.3f}%"

        # ---- 组装画面（与 realtime.py 完全一致的布局） ----
        view = np.vstack(
            [
                np.zeros((HEADER_H, SHOW_W * 2, 3), dtype=np.uint8),
                np.hstack([dist_big, peak_bgr]),
                np.hstack([refl_bgr, empty_bgr]),
            ]
        )
        _with_text(view[:HEADER_H], hover1, y=22)
        _with_text(view[:HEADER_H], hover2, y=46)

        cv2.imshow(win, view)

        # ---- 直方图窗口（与 realtime.py 完全一致） ----
        hist_img = _render_histogram_bgr(frame.raw.hist[py, px])
        cv2.imshow(hist_win, hist_img)

        key = cv2.waitKeyEx(20)
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            playing = not playing
            next_play_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if key in (2424832, ord("a"), ord("A")) and idx > 0:
            idx -= 1
            cv2.setTrackbarPos("frame", win, idx)
        if key in (2555904, ord("d"), ord("D")) and idx < len(frames) - 1:
            idx += 1
            cv2.setTrackbarPos("frame", win, idx)


# ======================== main ========================
def main() -> int:
    bag_path = (Path.cwd() / BAG_NAME).resolve()
    if not bag_path.exists():
        print(f"[WARN] 文件不存在: {bag_path}")
        return 1

    raw_frames = load_raw_frames(bag_path)
    if not raw_frames:
        return 1

    dist_np, conf_np, peak_np, refl_np, snr_np = run_network(raw_frames, batch_size=128)

    computed: list[ComputedFrameData] = []
    for i, rf in enumerate(raw_frames):
        computed.append(ComputedFrameData(
            raw=rf, dist=dist_np[i], conf=conf_np[i],
            peak=peak_np[i], reflect=refl_np[i], snr=snr_np[i],
        ))

    print(f"[shape] {BIN_NUM}x{TOF_H}x{TOF_W}, dtype=uint16 -> float32")
    print("[chan] dist/conf/peak/reflectance/snr")
    print("[mask] dist: conf<=0.5 -> black")
    print(f"[INFO] 共 {len(computed)} 帧")
    print("  操作: A/D ← → 切帧, 空格 播放/暂停, ESC 退出")
    run_viewer(computed)
    cv2.destroyAllWindows()
    print("[DONE] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
