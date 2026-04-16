#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/realtime.py

实时读取 tof.raw（内置 ToFRealtimeServer 采集线程），
运行模型并实时显示 4 张图：
- DIST: 预测距离（伪彩）
- SNR: 信噪比（灰度）
- PEAK: 峰值（灰度）
- REFLECT: 反射率（灰度）
- HIST: 鼠标悬停点的输入直方图（实时刷新）

交互：
- 鼠标悬停：显示 dist/snr/conf/peak/reflectance
- 空格：开始/停止录制 mp4；按 0：保存当前帧 tof.raw 到 r/tmp
- ESC 退出
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import threading
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

TOF_H = 30
TOF_W = 40
TOF_C = 64

# 输出分类配置（与 train.py / net.py 对齐）
NUM_BINS = 64
VALID_BINS = 63
INVALID_BIN = 63
MAX_VALID_M = 35.0
LOG_BASE = 1.06

# 单图显示长短边；是否旋转时会自动交换宽高
SHOW_LONG = 520
SHOW_SHORT = 390
ROTATE_90 = False
HEADER_H = 56
HIST_BINS = 62
HIST_W = 640
HIST_H = 280
TARGET_FPS = 25.0
FPS_STAT_INTERVAL_S = 0.5
REC_FPS = 20.0

EPS = 1e-6
TAIL_BASE = 1024.0
PULSES = 50000.0
DISP_GAMMA = 1.2
SUM_GATE_MAX = 20000.0
SUM_GATE_SNR_DIV = 3.0
PEAK_GATE_MIN = 30.0
SNR_GATE_LE3M = 5.5      # <=3m
SNR_GATE_3TO5M = 5.0     # (3,5]m
SNR_GATE_5TO8M = 4.5     # (5,8]m
SNR_GATE_GT8M = 4.0      # >8m
SNR_SHOW_MAX = 10.0
TOF_RAW_HEADER_BYTES = 5120
RECORD_DIR = Path("./tmp")
LOCAL_CACHE_DIR = Path("./tmp")
LOCAL_RAW_PATH = LOCAL_CACHE_DIR / "tof.raw"
ADB_PULL_TIMEOUT_S = 0.9

# BAG/MCAP 模式常量 (--bag)
_VPI_HEADER_FMT = "<B3xIQB1xHIIfff"
_VPI_HEADER_SIZE = struct.calcsize(_VPI_HEADER_FMT)  # 40
_BAG_PIXELS = TOF_H * TOF_W
_BAG_RAW_U16_COUNT = 2 * _BAG_PIXELS * TOF_C
_BAG_RAW_BYTES = _BAG_RAW_U16_COUNT * 2
_BAG_RESERVED_BYTES = 8 * 2
_BAG_PAYLOAD_SIZE = _VPI_HEADER_SIZE + _BAG_RAW_BYTES + _BAG_RESERVED_BYTES
_BAG_RAW_TOPIC = "sensor/vp_tof_info"
_BAG_PLAY_HZ = 10.0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ToFFrame:
    ts: float
    raw_bytes: bytes


class ToFRealtimeServer:
    """轻量同进程 ToF 采集服务（内置版，避免依赖外部模块）。"""

    def __init__(
        self,
        *,
        queue_maxlen: int = 5,
        min_peak_count: float = 100.0,
        target_fps: float = 10.0,
        raw_expected_bytes: int | None = None,
        read_retry: int = 3,
    ) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._q: Deque[ToFFrame] = deque(maxlen=int(max(queue_maxlen, 1)))
        self._target_dt = 1.0 / float(max(target_fps, 1.0))
        self._read_retry = int(max(read_retry, 0))
        self._min_peak_count = float(max(min_peak_count, 0.0))
        if raw_expected_bytes is None:
            self._raw_expected_bytes = int(TOF_RAW_HEADER_BYTES + TOF_H * TOF_W * TOF_C * 2)
        else:
            self._raw_expected_bytes = int(raw_expected_bytes)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ToFRealtimeServerInline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=0.8)
        self._thread = None

    def get_latest(self) -> Optional[ToFFrame]:
        with self._lock:
            return self._q[-1] if self._q else None

    @staticmethod
    def _adb_trigger_generate_raw() -> bool:
        cmd = "if [ -e /tmp/sv_tof ]; then rm /tmp/sv_tof && rm /tmp/tof.raw; fi && touch /tmp/sv_tof"
        try:
            r = subprocess.run(
                ["adb", "shell", cmd],
                timeout=0.6,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return int(r.returncode) == 0
        except Exception:
            return False

    @staticmethod
    def _adb_pull_raw_bytes(*, expected_bytes: int, retry: int) -> bytes | None:
        expected = int(expected_bytes)
        retr = int(max(retry, 0))
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for k in range(retr + 1):
            try:
                if LOCAL_RAW_PATH.exists():
                    LOCAL_RAW_PATH.unlink(missing_ok=True)
                r = subprocess.run(
                    ["adb", "pull", "/tmp/tof.raw", str(LOCAL_RAW_PATH)],
                    timeout=float(ADB_PULL_TIMEOUT_S),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if int(r.returncode) != 0:
                    if k < retr:
                        time.sleep(0.01)
                    continue
                if (not LOCAL_RAW_PATH.exists()) or int(LOCAL_RAW_PATH.stat().st_size) < expected:
                    if k < retr:
                        time.sleep(0.01)
                    continue
                out = LOCAL_RAW_PATH.read_bytes()
                if len(out) >= expected:
                    return bytes(out[:expected])
            except Exception:
                pass
            if k < retr:
                time.sleep(0.01)
        return None

    def _run(self) -> None:
        fail_sleep = 0.15
        while not self._stop.is_set():
            ok = self._adb_trigger_generate_raw()
            if not ok:
                time.sleep(fail_sleep)
                continue
            time.sleep(0.03)
            t0 = time.perf_counter()
            raw_bytes = self._adb_pull_raw_bytes(expected_bytes=self._raw_expected_bytes, retry=self._read_retry)
            if not raw_bytes:
                time.sleep(fail_sleep)
                continue
            raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            hists = tof_histograms_from_u16(raw_u16)
            if hists.shape != (TOF_H, TOF_W, TOF_C):
                time.sleep(fail_sleep)
                continue
            if self._min_peak_count > 0.0:
                peak = float(np.max(hists[:, :, :HIST_BINS]))
                if peak < self._min_peak_count:
                    time.sleep(fail_sleep)
                    continue
            frame = ToFFrame(ts=time.time(), raw_bytes=raw_bytes)
            with self._lock:
                self._q.append(frame)
            dt = time.perf_counter() - t0
            sleep = self._target_dt - dt
            if sleep > 0:
                time.sleep(min(sleep, 0.2))


def tof_histograms_from_u16(raw_u16: np.ndarray) -> np.ndarray:
    """从包含头部的 tof.raw(uint16) 解析 (30,40,64) 直方图。"""
    header_words = int(TOF_RAW_HEADER_BYTES // 2)
    if raw_u16.size <= header_words:
        return np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    data = raw_u16[header_words:]
    expected = TOF_H * TOF_W * TOF_C
    if data.size < expected:
        return np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    return data[:expected].reshape((TOF_H, TOF_W, TOF_C)).astype(np.uint16, copy=False)


def _get_show_size(rotate_90: bool) -> Tuple[int, int]:
    """返回单图显示尺寸；旋转后同步切换宽高比例。"""
    if rotate_90:
        return SHOW_SHORT, SHOW_LONG
    return SHOW_LONG, SHOW_SHORT


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int, rotate_90: bool) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    if rotate_90:
        # 与项目里旧版四宫格一致：rot90CW + flipH，等价于转置。
        py = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
        px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    else:
        px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
        py = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int, rotate_90: bool) -> Tuple[int, int]:
    """ToF 像素坐标 -> 显示坐标。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    if rotate_90:
        dx = int(np.clip((py_i + 0.5) * sw / TOF_H, 0, sw - 1))
        dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    else:
        dx = int(np.clip((px_i + 0.5) * sw / TOF_W, 0, sw - 1))
        dy = int(np.clip((py_i + 0.5) * sh / TOF_H, 0, sh - 1))
    return dx, dy


def _orient_for_display(img: np.ndarray, rotate_90: bool) -> np.ndarray:
    """按显示选项变换方向；旋转模式与旧版四宫格保持一致。"""
    if not rotate_90:
        return img

    import cv2  # type: ignore

    return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    """在图上画一个小圆点（黑边白心），用于标记 hover 像素。"""
    import cv2  # type: ignore

    xx = int(np.clip(x, 0, img_bgr.shape[1] - 1))
    yy = int(np.clip(y, 0, img_bgr.shape[0] - 1))
    cv2.circle(img_bgr, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img_bgr, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def _with_text(img_bgr: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    import cv2  # type: ignore

    cv2.putText(img_bgr, text, (10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def _render_histogram_bgr(
    bins: np.ndarray,
    w: int = HIST_W,
    h: int = HIST_H,
) -> np.ndarray:
    import cv2  # type: ignore

    b = np.asarray(bins, dtype=np.float32).reshape(-1)[:TOF_C]
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


def _colorize_gray01(x: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=2)


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """(H,W,64) -> (H,W) uint8，最大 bin 亮度图。"""
    h = np.asarray(hists, dtype=np.float32)
    if h.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad hists shape: {h.shape}")

    max_bin = np.max(h[:, :, :HIST_BINS], axis=2)
    sat_value = h[:, :, 62] * TAIL_BASE + h[:, :, 63]
    sat_value = np.where(sat_value > 0.0, sat_value, float(PULSES))

    y = max_bin * float(PULSES) / sat_value / 20000.0
    y = np.power(np.clip(y, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def _compute_snr_from_input(hists: np.ndarray) -> np.ndarray:
    """按输入前 62 个 bin 计算 SNR 图：snr=(max-mean)/std。"""
    h = np.asarray(hists, dtype=np.float32)
    if h.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad hists shape: {h.shape}")

    src = h[:, :, :HIST_BINS]
    vmax = np.max(src, axis=2)
    vsum = np.sum(src, axis=2, dtype=np.float32)
    mean = np.mean(src, axis=2, dtype=np.float32)
    std = np.std(src, axis=2, dtype=np.float32)
    snr = (vmax - mean) / np.maximum(std, 1e-6)

    div = float(max(SUM_GATE_SNR_DIV, 1.0))
    snr = np.where(vsum > float(SUM_GATE_MAX), snr / div, snr)
    snr = np.where(vmax < float(PEAK_GATE_MIN), 0.0, snr)
    return snr.astype(np.float32, copy=False)


def _colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """(H,W) depth(m) -> BGR (JET), 直接按 y=1.8/x 做伪彩映射。"""
    import cv2  # type: ignore

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


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """SNR 标量图 -> 灰度 BGR（gamma=DISP_GAMMA），invalid 为黑。"""
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)

    gamma = float(DISP_GAMMA)
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m] / float(SNR_SHOW_MAX), 0.0, 1.0), 1.0 / gamma)

    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _make_range_snr_mask(depth_m: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """按 pred 距离段应用 SNR 阈值：

    - <=3m:   snr > 5.5
    - 3~5m:   snr > 5
    - 5~8m:   snr > 4.5
    - >8m:    snr > 4
    """
    d = np.asarray(depth_m, dtype=np.float32)
    s = np.asarray(snr, dtype=np.float32)
    valid = np.isfinite(d) & np.isfinite(s) & (d > 0.0)
    thr = np.full(d.shape, float(SNR_GATE_GT8M), dtype=np.float32)
    thr = np.where(d <= 8.0, float(SNR_GATE_5TO8M), thr)
    thr = np.where(d <= 5.0, float(SNR_GATE_3TO5M), thr)
    thr = np.where(d <= 3.0, float(SNR_GATE_LE3M), thr)
    return valid & (s > thr)


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hist (H,W,64) -> pred_depth, snr, conf, peak, reflectance."""
    import torch

    h = np.asarray(hists, dtype=np.float32)
    with torch.inference_mode():
        inp = torch.from_numpy(h).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        dist_t, conf_t, peak_t, reflectance_t, snr_t = net(inp)
        pred_depth = dist_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        snr = snr_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        conf = conf_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        peak = peak_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        reflectance = reflectance_t[0, 0].cpu().numpy().astype(np.float32, copy=False)

    invalid = (~np.isfinite(pred_depth)) | (pred_depth <= 0.0)
    if np.any(invalid):
        pred_depth = pred_depth.copy()
        snr = snr.copy()
        conf = conf.copy()
        peak = peak.copy()
        reflectance = reflectance.copy()
        pred_depth[invalid] = 0.0
        snr[invalid] = 0.0
        conf[invalid] = 0.0
        peak[invalid] = 0.0
        reflectance[invalid] = 0.0

    return pred_depth, snr, conf, peak, reflectance


# ======================== 共用渲染 ========================

def _render_view(
    dist: np.ndarray,
    conf: np.ndarray,
    peak: np.ndarray,
    reflectance: np.ndarray,
    hists: np.ndarray,
    mouse_x: int,
    mouse_y: int,
    show_w: int,
    show_h: int,
    rotate_90: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """渲染四宫格 (DISTANCE|PEAK / REFLECTANCE|空) + 直方图。

    Returns: (view_bgr, hist_bgr, tof_px, tof_py)
        view_bgr 的前 HEADER_H 行为空白，留给调用方写 hover 文本。
    """
    import cv2

    dist_for_disp = np.where(conf > 0.5, dist, 0.0)
    dist_big = cv2.resize(
        _orient_for_display(_colorize_depth(dist_for_disp), rotate_90),
        (show_w, show_h), interpolation=cv2.INTER_NEAREST,
    )

    peak_u8 = np.clip(
        np.power(peak / max(float(peak.mean()), EPS) * (50.0 / 255.0), 1.0 / 1.5) * 255.0,
        0, 255,
    ).astype(np.uint8)
    peak_bgr = cv2.cvtColor(
        cv2.resize(_orient_for_display(peak_u8, rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    refl_u8 = np.clip(np.rint(np.clip(reflectance, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    refl_bgr = cv2.cvtColor(
        cv2.resize(_orient_for_display(refl_u8, rotate_90), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    empty_bgr = np.zeros((show_h, show_w, 3), dtype=np.uint8)

    mx = int(np.clip(mouse_x, 0, show_w * 2 - 1))
    my = int(np.clip(mouse_y - HEADER_H, 0, show_h * 2 - 1))
    tile_x0 = 0 if mx < show_w else show_w
    tile_y0 = 0 if my < show_h else show_h
    px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, show_w, show_h, rotate_90)
    dx, dy = _pixel_to_disp_xy(px, py, show_w, show_h, rotate_90)

    for img in [dist_big, peak_bgr, refl_bgr]:
        _draw_marker(img, dx, dy)

    _with_text(dist_big, "DISTANCE")
    _with_text(peak_bgr, "PEAK")
    _with_text(refl_bgr, "REFLECTANCE")

    view = np.vstack([
        np.zeros((HEADER_H, show_w * 2, 3), dtype=np.uint8),
        np.hstack([dist_big, peak_bgr]),
        np.hstack([refl_bgr, empty_bgr]),
    ])

    hist_img = _render_histogram_bgr(hists[py, px, :])
    return view, hist_img, px, py


# ======================== BAG/MCAP 解析 ========================

def _parse_vp_tof_info(payload: bytes):
    """解析 VpTofInfo 消息，返回 (TofInfoHeader, hist(30,40,64)) 或 None。"""
    if len(payload) < _BAG_PAYLOAD_SIZE:
        return None
    vals = struct.unpack_from(_VPI_HEADER_FMT, payload, 0)
    header = TofInfoHeader(
        is_valid=vals[0], frame_id=vals[1], timestamp_us=vals[2],
        work_mode=vals[3], bin_mode=vals[4], light_count=vals[5],
        expo_time=vals[6], pulse_width=vals[7], rx_temp=vals[8], tx_temp=vals[9],
    )
    raw_all = np.frombuffer(payload[_VPI_HEADER_SIZE:_VPI_HEADER_SIZE + _BAG_RAW_BYTES], dtype="<u2")
    hist = raw_all[:_BAG_PIXELS * TOF_C].reshape(TOF_H, TOF_W, TOF_C).copy()
    return header, hist


def _load_bag_frames(bag_path: Path) -> list:
    """从 BAG/MCAP 文件加载所有 VpTofInfo 帧，返回 [(TofInfoHeader, hist), ...]。"""
    from mcap.exceptions import EndOfFile
    from mcap.reader import NonSeekingReader, make_reader

    frames: list = []
    print(f"[INFO] 扫描: {bag_path}")
    cnt = 0

    def consume(msg_iter):
        nonlocal cnt
        for _, channel, message in msg_iter:
            if (channel.topic or "") != _BAG_RAW_TOPIC:
                continue
            cnt += 1
            parsed = _parse_vp_tof_info(message.data)
            if parsed:
                frames.append(parsed)

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


# ======================== BAG 模式 ========================

def _run_bag_mode(bag_path_str: str) -> int:
    """从 BAG/MCAP 文件读取帧，批量推理后交互式浏览。"""
    import cv2
    import torch

    cv2.setUseOptimized(True)
    rotate_90 = bool(ROTATE_90)
    show_w, show_h = _get_show_size(rotate_90)

    bag_path = Path(bag_path_str).resolve()
    if not bag_path.exists():
        print(f"[ERR] 文件不存在: {bag_path}")
        return 1

    frames = _load_bag_frames(bag_path)
    if not frames:
        print("[ERR] 没有有效帧")
        return 1

    nn_dir = Path(__file__).resolve().parent
    root = nn_dir.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    n = len(frames)
    print(f"[INFO] 批量推理 {n} 帧...")
    all_dist = np.zeros((n, TOF_H, TOF_W), dtype=np.float32)
    all_snr = np.zeros_like(all_dist)
    all_conf = np.zeros_like(all_dist)
    all_peak = np.zeros_like(all_dist)
    all_refl = np.zeros_like(all_dist)

    for i, (_, hist) in enumerate(frames):
        d, s, c, p, r = _run_infer(net, device, hist.astype(np.float32))
        all_dist[i], all_snr[i], all_conf[i], all_peak[i], all_refl[i] = d, s, c, p, r
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  [{i + 1}/{n}]")
    print("[OK] 推理完成")

    win = "NN_REALTIME"
    hist_win = "HIST"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(hist_win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", win, 0, n - 1, lambda _: None)

    mouse: dict = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) == int(cv2.EVENT_MOUSEMOVE):
            mouse["x"], mouse["y"] = int(x), int(y)

    cv2.setMouseCallback(win, on_mouse)

    playing = False
    play_interval_ms = int(round(1000.0 / _BAG_PLAY_HZ))
    next_play_ms = 0
    idx = 0
    print(f"[INFO] 共 {n} 帧, A/D ← → 切帧, 空格 播放/暂停, ESC 退出")

    while True:
        tb_pos = cv2.getTrackbarPos("frame", win)
        if tb_pos != idx and not playing:
            idx = max(0, min(tb_pos, n - 1))

        now_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if playing and now_ms >= next_play_ms:
            if idx < n - 1:
                idx += 1
            else:
                playing = False
            cv2.setTrackbarPos("frame", win, idx)
            next_play_ms = now_ms + play_interval_ms

        header, hist = frames[idx]
        view, hist_img, px, py = _render_view(
            all_dist[idx], all_conf[idx], all_peak[idx], all_refl[idx],
            hist.astype(np.float32), mouse["x"], mouse["y"],
            show_w, show_h, rotate_90,
        )

        hover1 = (
            f"idx={idx}/{n - 1}  frame_id={header.frame_id}  "
            f"dist {float(all_dist[idx][py, px]):.3f}m  snr {float(all_snr[idx][py, px]):.3f}  "
            f"conf {float(all_conf[idx][py, px]):.0f}  peak {float(all_peak[idx][py, px]):.3f}"
        )
        hover2 = f"reflectance {float(all_refl[idx][py, px]) * 100.0:.3f}%"
        _with_text(view[:HEADER_H], hover1, y=22)
        _with_text(view[:HEADER_H], hover2, y=46)

        cv2.imshow(win, view)
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
        if key in (2555904, ord("d"), ord("D")) and idx < n - 1:
            idx += 1
            cv2.setTrackbarPos("frame", win, idx)

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="realtime.py — ADB 实时 / BAG 回放")
    parser.add_argument("bag", nargs="?", default=None,
                        help="BAG/MCAP 文件路径；不指定则默认 ADB 实时模式")
    args = parser.parse_args()

    if args.bag:
        return _run_bag_mode(args.bag)

    # ---- ADB 实时模式 ----
    rotate_90 = bool(ROTATE_90)
    show_w, show_h = _get_show_size(rotate_90)

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e
    cv2.setUseOptimized(True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    cv2.namedWindow("NN_REALTIME", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("HIST", cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback("NN_REALTIME", on_mouse)

    tof_srv = ToFRealtimeServer(queue_maxlen=5, min_peak_count=100.0, target_fps=float(TARGET_FPS))
    tof_srv.start()

    last_ts = 0.0
    cached_in: np.ndarray | None = None
    cached_pred_depth: np.ndarray | None = None
    cached_snr: np.ndarray | None = None
    cached_peak: np.ndarray | None = None
    cached_reflectance: np.ndarray | None = None
    cached_conf: np.ndarray | None = None
    last_mouse_xy = (-1, -1)
    view_cache: np.ndarray | None = None
    hist_cache: np.ndarray | None = None

    io_fps = 0.0
    infer_fps = 0.0
    ui_fps = 0.0
    io_cnt = 0
    infer_cnt = 0
    ui_cnt = 0
    fps_tick = time.perf_counter()
    rec_writer: object | None = None
    rec_path = ""
    rec_err = ""
    last_rec_on = False
    latest_raw_bytes: bytes | None = None

    try:
        while True:
            now = time.perf_counter()
            got_new_frame = False
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                io_cnt += 1
                raw_u16 = np.frombuffer(frame.raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16)
                if hists.shape == (TOF_H, TOF_W, TOF_C):
                    pred_depth, snr, conf, peak, reflectance = _run_infer(net, device, hists)
                    infer_cnt += 1
                    cached_in = hists
                    cached_pred_depth = pred_depth
                    cached_snr = snr
                    cached_peak = peak
                    cached_reflectance = reflectance
                    cached_conf = conf
                    latest_raw_bytes = bytes(frame.raw_bytes)
                    last_ts = float(frame.ts)
                    got_new_frame = True

            if (
                cached_in is None
                or cached_pred_depth is None
                or cached_snr is None
                or cached_peak is None
                or cached_reflectance is None
                or cached_conf is None
            ):
                k = int(cv2.waitKey(5) & 0xFF)
                if k == 27:
                    break
                continue

            mouse_xy = (int(mouse.get("x", 0)), int(mouse.get("y", 0)))
            rec_on = rec_writer is not None
            need_redraw = (
                got_new_frame
                or (mouse_xy != last_mouse_xy)
                or (view_cache is None)
                or (hist_cache is None)
                or (rec_on != last_rec_on)
            )

            if need_redraw:
                view, hist_new, px, py = _render_view(
                    cached_pred_depth, cached_conf, cached_peak, cached_reflectance,
                    cached_in, mouse_xy[0], mouse_xy[1], show_w, show_h, rotate_90,
                )

                dt_fps = now - fps_tick
                if dt_fps >= float(FPS_STAT_INTERVAL_S):
                    inv_dt = 1.0 / max(dt_fps, 1e-6)
                    io_fps = float(io_cnt) * inv_dt
                    infer_fps = float(infer_cnt) * inv_dt
                    ui_fps = float(ui_cnt) * inv_dt
                    io_cnt = 0
                    infer_cnt = 0
                    ui_cnt = 0
                    fps_tick = now

                hover1 = (
                    f"io_fps {io_fps:.1f}  infer_fps {infer_fps:.1f}  ui_fps {ui_fps:.1f}  "
                    f"dist {float(cached_pred_depth[py, px]):.3f}m  snr {float(cached_snr[py, px]):.3f}  conf {float(cached_conf[py, px]):.0f}  peak {float(cached_peak[py, px]):.3f}"
                )
                hover2 = f"reflectance {float(cached_reflectance[py, px]) * 100.0:.3f}%"
                _with_text(view[:HEADER_H], hover1, y=22)
                _with_text(view[:HEADER_H], hover2, y=46)
                if rec_writer is not None:
                    cv2.circle(view, (show_w * 2 - 24, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(view, "REC", (show_w * 2 - 72, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2, cv2.LINE_AA)
                if rec_err:
                    cv2.putText(
                        view,
                        f"rec err: {rec_err}",
                        (10, HEADER_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                view_cache = view
                hist_cache = hist_new
                last_mouse_xy = mouse_xy
                last_rec_on = rec_on

            ui_cnt += 1
            cv2.imshow("NN_REALTIME", view_cache)
            cv2.imshow("HIST", hist_cache)
            if rec_writer is not None and view_cache is not None:
                rec_writer.write(view_cache)

            k = int(cv2.waitKey(1) & 0xFF)
            if k == 32:  # Space: toggle recording
                if rec_writer is None:
                    rec_err = ""
                    try:
                        RECORD_DIR.mkdir(parents=True, exist_ok=True)
                        rec_path = str(RECORD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(rec_path, fourcc, max(float(REC_FPS), 1.0), (view_cache.shape[1], view_cache.shape[0]))
                        if not writer.isOpened():
                            writer.release()
                            raise RuntimeError("VideoWriter open failed")
                        rec_writer = writer
                        print(f"[rec] start {rec_path}")
                    except Exception as e:
                        rec_writer = None
                        rec_err = str(e)
                else:
                    rec_writer.release()
                    rec_writer = None
                    rec_err = ""
                    print(f"[rec] stop  {rec_path}")
            if k == 48:  # '0': save current raw bytes
                try:
                    if latest_raw_bytes is None:
                        print("[raw] no frame yet")
                    else:
                        RECORD_DIR.mkdir(parents=True, exist_ok=True)
                        raw_path = str(RECORD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw")
                        Path(raw_path).write_bytes(latest_raw_bytes)
                        print(f"[raw] saved {raw_path}")
                except Exception as e:
                    print(f"[raw] save failed: {e}")
            if k == 27:
                break
    finally:
        if rec_writer is not None:
            rec_writer.release()
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

