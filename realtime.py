#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/realtime.py

实时读取 tof.raw（内置 ToFRealtimeServer 采集线程），
运行模型并在【单个仪表盘窗口】中显示：
- 主视图: 预测距离(伪彩) / 峰值 / 反射率
- BINS:   鼠标悬停像素的 bins 表格
- HIST:   鼠标悬停像素的输入直方图（实时刷新）
- HIST 上方: 悬停像素的 dist/snr/peak/reflectance
- 距离色条: 近红远蓝；右下角文字输入可调最近/最远距离与最小/最大亮度；悬停色条可查询距离

交互：
- 鼠标悬停：显示 dist/snr/peak/reflectance；悬停色条查询该颜色对应距离
- 右下角输入框：最近距离(m) / 最远距离(m) / 最小亮度 / 最大亮度（点击后键盘输入，Enter 确认）
- 空格：开始/停止录制 mp4；按 0：保存当前帧 tof.raw 到 r/tmp
- ESC 退出（输入框聚焦时 ESC 取消编辑）

命令行：
- `py realtime.py`              ADB 实时模式
- `py realtime.py xxx.mcap`     BAG 回放
- `py realtime.py cali_data`    浏览目录内所有 .raw（进度条可拖动，A/D 切帧，空格播放）
- `py realtime.py 日志目录`     自动查找 dtof_depth_bag 下的 BAG 并合并回放
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import threading
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
DEPTH_COLOR_NEAR_M = 0.0   # 红色对应距离(米)
DEPTH_COLOR_FAR_M = 25.0   # 蓝色对应距离(米)
DEPTH_COLOR_TB_MAX_DM = 350  # trackbar 上限：0.1m 单位 → 35.0m
CBAR_W = 84                  # 距离色条宽度
PEAK_DISP_LO = 0.0           # PEAK 灰度显示下限默认
PEAK_DISP_HI = 5000.0        # PEAK 灰度显示上限默认
PEAK_DISP_ABS_MAX = 1_000_000.0  # 亮度输入允许的最大值（不再卡死在 5000）
LOG_BASE = 1.06

# 单图显示长短边；是否旋转时会自动交换宽高
SHOW_LONG = 520
SHOW_SHORT = 390
# 显示方向开关：先按原始方向，再决定是否顺时针旋转 90°，最后是否水平镜像
ROTATE_90 = 1
MIRROR = 0
HIST_BINS = 62
HIST_W = 640
HIST_H = 280
STRIP_BINS = 62   # 表格显示 bins 0..61
STRIP_W = 240     # 图像宽度（函数内动态计算高度，此值仅作默认参数）
STRIP_H = 0       # 高度由行数自动决定，此常量不再用于绘制
# 显示用 peak：不再用网络 peak，而是 hist_k 在 bin [LO, HI] 的最大值
PEAK_BIN_LO = 5
PEAK_BIN_HI = 60  # inclusive
TARGET_FPS = 25.0
FPS_STAT_INTERVAL_S = 0.5
REC_FPS = 20.0

EPS = 1e-6
TAIL_BASE = 1024.0
PULSES = 50000.0
DIST_SCALE_M = 0.6   # 与 net.py 一致，用于反查有效测距 bin
DIST_BIAS = -2.14
MIN_DIST_M = 0.4
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
_BAG_PIXELS = TOF_H * TOF_W
_BAG_RAW_U16_COUNT = _BAG_PIXELS * TOF_C
_BAG_RAW_BYTES = _BAG_RAW_U16_COUNT * 2
_BAG_RESERVED_BYTES = 8 * 2

# ---- 旧协议 V1: header(40) -> tof_raw -> reserved(16) ----
# 字段顺序: is_valid, frame_id, timestamp, work_mode, bin_mode, light_count,
#           expo_time, pulse_width, rx_temp, tx_temp
_VPI_HEADER_FMT = "<B3xIQB1xHIIfff"
_VPI_HEADER_SIZE = struct.calcsize(_VPI_HEADER_FMT)  # 40
_BAG_PAYLOAD_SIZE = _VPI_HEADER_SIZE + _BAG_RAW_BYTES + _BAG_RESERVED_BYTES  # 153656

# ---- 新协议 V2: header(40) -> reserved(16) -> tof_metadata -> tof_raw ----
# 字段顺序: timestamp, is_valid, work_mode, bin_mode, frame_id, light_count,
#           expo_time, pulse_width, rx_temp, tx_temp, vspad(新增)
# tof_metadata[TOF_W * TOF_BIN_MAX] = 40*64 = 2560 u16 = 5120B，插在 raw 之前
_VPI_HEADER_FMT_V2 = "<QBBHIIIffff"
_VPI_HEADER_SIZE_V2 = struct.calcsize(_VPI_HEADER_FMT_V2)  # 40
_BAG_METADATA_BYTES = (TOF_W * TOF_C) * 2  # 5120
_BAG_PAYLOAD_SIZE_V2 = (
    _VPI_HEADER_SIZE_V2 + _BAG_RESERVED_BYTES + _BAG_METADATA_BYTES + _BAG_RAW_BYTES
)  # 158776

_BAG_RAW_TOPIC = "sensor/vp_tof_info"
_BAG_PLAY_HZ = 10.0
_BAG_LOAD_WORKERS = max(1, min(os.cpu_count() or 4, 16))
_BAG_INFER_BATCH = 64

# BAG 深度 topic (alg/dtof_depth) 常量
_BAG_DEPTH_TOPIC = "alg/dtof_depth"
_DEPTH_HDR_SIZE = 16  # uint64 + uint32 + uint8 + reserved[3]
_DEPTH_DIST_BYTES = _BAG_PIXELS * 2
_DEPTH_CONF_BYTES = _BAG_PIXELS
_DEPTH_PEAK_BYTES = _BAG_PIXELS * 2
_DEPTH_PAYLOAD_SIZE = _DEPTH_HDR_SIZE + _DEPTH_DIST_BYTES + _DEPTH_CONF_BYTES + _DEPTH_PEAK_BYTES


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
    vspad: float = 0.0  # V2 新增；V1 包无此字段，默认 0.0


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
                    return bytes(out)
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
                peak = float(np.max(hists[:, :, PEAK_BIN_LO : PEAK_BIN_HI + 1]))
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
    """从 tof.raw(uint16) 解析 (30,40,64) 直方图。

    直接取末尾的 30*40*64 个值，无需关心前面头部到底有多少字节，
    兼容不同头部长度的数据源。
    """
    expected = TOF_H * TOF_W * TOF_C
    if raw_u16.size < expected:
        return np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    return raw_u16[-expected:].reshape((TOF_H, TOF_W, TOF_C)).astype(np.uint16, copy=False)


def _get_show_size(rotate_90: bool) -> Tuple[int, int]:
    """返回单图显示尺寸；旋转后同步切换宽高比例。"""
    if rotate_90:
        return SHOW_SHORT, SHOW_LONG
    return SHOW_LONG, SHOW_SHORT


def _orient_for_display(img: np.ndarray, rotate_90: bool, mirror: bool) -> np.ndarray:
    """按开关变换显示方向：先(可选)顺时针旋转 90°，再(可选)水平镜像。"""
    import cv2  # type: ignore

    if rotate_90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mirror:
        img = cv2.flip(img, 1)
    return img


def _disp_xy_to_pixel(dx: int, dy: int, show_w: int, show_h: int, rotate_90: bool, mirror: bool) -> Tuple[int, int]:
    """显示坐标 -> ToF 像素坐标（按显示开关做反变换）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    if mirror:
        dx = sw - 1 - int(dx)
    if rotate_90:
        # 顺时针旋转后：display x ↔ TOF_H 维(行的反向), display y ↔ TOF_W 维(列)
        # 必须先 floor 列号再做反向减法，否则与逆变换/INTER_NEAREST 显示差 1
        col = int(np.clip(dx * TOF_H / sw, 0, TOF_H - 1))
        py = int(np.clip((TOF_H - 1) - col, 0, TOF_H - 1))
        px = int(np.clip(dy * TOF_W / sh, 0, TOF_W - 1))
    else:
        px = int(np.clip(dx * TOF_W / sw, 0, TOF_W - 1))
        py = int(np.clip(dy * TOF_H / sh, 0, TOF_H - 1))
    return px, py


def _pixel_to_disp_xy(px: int, py: int, show_w: int, show_h: int, rotate_90: bool, mirror: bool) -> Tuple[int, int]:
    """ToF 像素坐标 -> 显示坐标（按显示开关做正变换）。"""
    sw = max(int(show_w), 1)
    sh = max(int(show_h), 1)
    px_i = int(np.clip(px, 0, TOF_W - 1))
    py_i = int(np.clip(py, 0, TOF_H - 1))
    if rotate_90:
        rx = (TOF_H - 1 - py_i) + 0.5  # 旋转后图像中的 x 坐标(基于 TOF_H 维)
        dx = int(np.clip(rx * sw / TOF_H, 0, sw - 1))
        dy = int(np.clip((px_i + 0.5) * sh / TOF_W, 0, sh - 1))
    else:
        dx = int(np.clip((px_i + 0.5) * sw / TOF_W, 0, sw - 1))
        dy = int(np.clip((py_i + 0.5) * sh / TOF_H, 0, sh - 1))
    if mirror:
        dx = sw - 1 - dx
    return dx, dy


def _draw_marker(img_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    """在图上画一个小圆点（黑边白心），用于标记 hover 像素。"""
    import cv2  # type: ignore

    xx = int(np.clip(x, 0, img_bgr.shape[1] - 1))
    yy = int(np.clip(y, 0, img_bgr.shape[0] - 1))
    cv2.circle(img_bgr, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img_bgr, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def _with_text(img_bgr: np.ndarray, text: str, y: int = 24, *, align: str = "left") -> np.ndarray:
    import cv2  # type: ignore

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.48, 1
    if align == "right":
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(int(img_bgr.shape[1] - tw - 10), 0)
    else:
        x = 10
    cv2.putText(img_bgr, text, (x, int(y)), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img_bgr


def _render_histogram_bgr(
    bins: np.ndarray,
    effective_bin: int = -1,
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
    # peak: 仅统计 bin [PEAK_BIN_LO, PEAK_BIN_HI] 的最大值（只用于数值显示，不标色）
    peak_slice = b_draw[PEAK_BIN_LO : PEAK_BIN_HI + 1]
    vmax_raw = float(np.max(peak_slice)) if peak_slice.size > 0 else 0.0
    peak_bin = int(PEAK_BIN_LO + int(np.argmax(peak_slice))) if peak_slice.size > 0 else -1
    value = vmax_raw * float(PULSES) / sat_value

    x0, y0 = 14, 80
    x1, y1 = img.shape[1] - 10, img.shape[0] - 18
    # 直方图柱高仍按全量程归一，便于看近场/尾部；peak 数值单独按区间算
    vmax_draw = float(np.max(b_draw)) if b_draw.size > 0 else 0.0
    vmax = 1.0 if (not np.isfinite(vmax_draw) or vmax_draw <= 0.0) else vmax_draw
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
        bar_color = (0, 140, 255) if i == int(effective_bin) else (255, 220, 0)
        cv2.rectangle(img, (xl, yt), (xr, y1), bar_color, -1)
        cv2.rectangle(img, (xl, yt), (xr, y1), (30, 30, 30), 1)
    active_text = str(int(effective_bin)) if effective_bin >= 0 else "INVALID"
    img = _with_text(img, f"RAW_HIST (0-61)  effective_bin={active_text}", y=24)
    img = _with_text(
        img,
        f"peak(bin{PEAK_BIN_LO}-{PEAK_BIN_HI})={value:.3f}  @bin={peak_bin}",
        y=48,
    )
    return img


def _render_bins_strip_bgr(
    bins: np.ndarray,
    px: int,
    py: int,
    effective_bin: int = -1,
    w: int = STRIP_W,
    h: int = STRIP_H,
) -> np.ndarray:
    """纯文字表格：62 行 × 2 列（bin#  eq_val），针对鼠标悬停像素。

    eq_val = bin_raw * PULSES / sat_value
    sat_value = b[62] * TAIL_BASE + b[63]（为 0 时取 PULSES）
    网络真实选中的测距 bin（effective）用橙色高亮；无有效 conf 则不标橙。
    """
    import cv2  # type: ignore

    b = np.asarray(bins, dtype=np.float32).reshape(-1)
    n_show = min(STRIP_BINS, int(b.size))  # 62

    b_draw = b[:n_show].copy()

    # 计算 sat_value（同主直方图逻辑）
    tail_63 = float(b[62]) if b.size > 62 else 0.0
    tail_64 = float(b[63]) if b.size > 63 else 0.0
    sat_value = tail_63 * TAIL_BASE + tail_64
    if sat_value == 0.0:
        sat_value = float(PULSES)

    # eq_val = raw * PULSES / sat_value
    eq = b_draw * float(PULSES) / sat_value

    # peak 数值：仅在 bin [PEAK_BIN_LO, PEAK_BIN_HI] 内取最大值（不标色）
    lo = max(0, min(PEAK_BIN_LO, n_show - 1))
    hi = max(lo, min(PEAK_BIN_HI, n_show - 1))
    peak_slice = b_draw[lo : hi + 1]
    argmax_idx = int(lo + int(np.argmax(peak_slice))) if peak_slice.size > 0 else -1
    peak_eq = float(eq[argmax_idx]) if argmax_idx >= 0 else 0.0

    # 每行行高
    row_h = 16
    header_h = 48
    img_h = header_h + n_show * row_h + 4
    img_w = max(int(w), 260)
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # 标题
    cv2.putText(
        img,
        f"pixel=({int(px)},{int(py)})  effective={effective_bin if effective_bin >= 0 else 'INVALID'}",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"sat={sat_value:.1f}  peak_eq({PEAK_BIN_LO}-{PEAK_BIN_HI})={peak_eq:.1f} @{argmax_idx}",
        (8, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, "bin    eq_val (raw*PULSES/sat)",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA,
    )
    cv2.line(img, (0, header_h - 1), (img_w, header_h - 1), (60, 60, 60), 1)

    for i in range(n_show):
        y_text = header_h + i * row_h + row_h - 4
        color = (0, 165, 255) if i == int(effective_bin) else (220, 220, 220)
        if i % 2 == 0:
            cv2.rectangle(
                img,
                (0, header_h + i * row_h),
                (img_w, header_h + (i + 1) * row_h - 1),
                (20, 20, 20), -1,
            )
        cv2.putText(
            img,
            f"{i:3d}    {eq[i]:10.2f}",
            (8, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )

    return img


def _dist_per_bin_pixel(hist62: np.ndarray) -> np.ndarray:
    """与 net.py _dist_per_bin 一致：每 bin 的三邻域重心距离 (62,)。"""
    x = np.asarray(hist62, dtype=np.float32).reshape(-1)
    if x.size < 62:
        x = np.pad(x, (0, 62 - int(x.size)), mode="constant")
    x = x[:62]
    anchors = np.clip(np.arange(62, dtype=np.int32), 1, 60)
    left = x[anchors - 1]
    center = x[anchors]
    right = x[anchors + 1]
    centroid = (right - left) / (left + center + right + 1.0) + anchors.astype(np.float32)
    dist = centroid * float(DIST_SCALE_M) + float(DIST_BIAS)
    return np.maximum(dist, float(MIN_DIST_M)).astype(np.float32, copy=False)


def _effective_bin_from_output(
    bins: np.ndarray,
    dist_m: float,
    conf: float,
) -> int:
    """反查网络 one_hot 选中的测距 bin。

    有有效 conf 时，在 62 路 dist_per_bin 中找与输出 dist 最接近的 bin；
    conf 无效则返回 -1（直方图/表格不标橙色）。
    """
    if conf <= 0.5 or (not np.isfinite(dist_m)) or float(dist_m) <= 0.0:
        return -1

    b = np.asarray(bins, dtype=np.float32).reshape(-1)
    if b.size < HIST_BINS:
        return -1

    d_per = _dist_per_bin_pixel(b[:HIST_BINS])
    if not np.all(np.isfinite(d_per)):
        return -1
    return int(np.argmin(np.abs(d_per - float(dist_m))))


def _colorize_gray01(x: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=2)


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    """(H,W,64) -> (H,W) uint8，最大 bin 亮度图（仅 bin 6-60）。"""
    h = np.asarray(hists, dtype=np.float32)
    if h.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad hists shape: {h.shape}")

    max_bin = np.max(h[:, :, PEAK_BIN_LO : PEAK_BIN_HI + 1], axis=2)
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


def _normalize_depth_color_range(near_m: float, far_m: float) -> tuple[float, float]:
    """规范化红/蓝端点距离；两端至少相差 0.1m。"""
    near = float(near_m) if np.isfinite(near_m) else float(DEPTH_COLOR_NEAR_M)
    far = float(far_m) if np.isfinite(far_m) else float(DEPTH_COLOR_FAR_M)
    near = float(np.clip(near, 0.0, float(DEPTH_COLOR_TB_MAX_DM) / 10.0))
    far = float(np.clip(far, 0.0, float(DEPTH_COLOR_TB_MAX_DM) / 10.0))
    if abs(far - near) < 0.1:
        far = min(near + 0.1, float(DEPTH_COLOR_TB_MAX_DM) / 10.0)
        if abs(far - near) < 0.1:
            near = max(far - 0.1, 0.0)
    return near, far


def _default_disp_ctrl() -> dict:
    """显示控制状态：距离色域 + PEAK 灰度范围 + 输入框编辑。"""
    return {
        "near_m": float(DEPTH_COLOR_NEAR_M),
        "far_m": float(DEPTH_COLOR_FAR_M),
        "peak_lo": float(PEAK_DISP_LO),
        "peak_hi": float(PEAK_DISP_HI),
        "focus": None,   # near_m / far_m / peak_lo / peak_hi
        "edit": "",
        "drag": None,    # seek 拖动
    }


def _normalize_peak_disp_range(lo: float, hi: float) -> tuple[float, float]:
    """规范化 PEAK 显示范围；两端至少相差 1，亮度上限放宽到 PEAK_DISP_ABS_MAX。"""
    lo_v = float(lo) if np.isfinite(lo) else float(PEAK_DISP_LO)
    hi_v = float(hi) if np.isfinite(hi) else float(PEAK_DISP_HI)
    lo_v = float(np.clip(lo_v, 0.0, float(PEAK_DISP_ABS_MAX)))
    hi_v = float(np.clip(hi_v, 0.0, float(PEAK_DISP_ABS_MAX)))
    if abs(hi_v - lo_v) < 1.0:
        hi_v = min(lo_v + 1.0, float(PEAK_DISP_ABS_MAX))
        if abs(hi_v - lo_v) < 1.0:
            lo_v = max(hi_v - 1.0, 0.0)
    return lo_v, hi_v


def _sync_disp_ctrl(ctrl: dict) -> None:
    near, far = _normalize_depth_color_range(ctrl.get("near_m", DEPTH_COLOR_NEAR_M), ctrl.get("far_m", DEPTH_COLOR_FAR_M))
    lo, hi = _normalize_peak_disp_range(ctrl.get("peak_lo", PEAK_DISP_LO), ctrl.get("peak_hi", PEAK_DISP_HI))
    ctrl["near_m"] = near
    ctrl["far_m"] = far
    ctrl["peak_lo"] = lo
    ctrl["peak_hi"] = hi


def _peak_to_u8(
    peak: np.ndarray,
    lo: float = PEAK_DISP_LO,
    hi: float = PEAK_DISP_HI,
) -> np.ndarray:
    """peak 标量图 -> uint8 灰度，[lo, hi] 线性映射到 [0, 255]。"""
    lo, hi = _normalize_peak_disp_range(lo, hi)
    p = np.asarray(peak, dtype=np.float32)
    t = (p - lo) / (hi - lo)
    return np.clip(np.rint(t * 255.0), 0, 255).astype(np.uint8)


def _colorize_depth(
    depth_m: np.ndarray,
    near_m: float = DEPTH_COLOR_NEAR_M,
    far_m: float = DEPTH_COLOR_FAR_M,
) -> np.ndarray:
    """(H,W) depth(m) -> BGR (JET)，near..far 线性均匀映射（近红远蓝）。"""
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    near_m, far_m = _normalize_depth_color_range(near_m, far_m)
    span = far_m - near_m
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    # 近红远蓝：near -> 255, far -> 0（near>far 时自动反向）
    t = (d[valid] - near_m) / span
    u8[valid] = np.clip(np.rint(255.0 * (1.0 - t)), 0, 255).astype(np.uint8)
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


def _compute_peak_map(hists: np.ndarray) -> np.ndarray:
    """显示用 peak = max(hist_k[PEAK_BIN_LO..PEAK_BIN_HI])。

    hist_k = hist * PULSES / sat_value，sat_value = bin62*1024 + bin63。
    支持 (H,W,64) 或 (N,H,W,64)。
    """
    h = np.asarray(hists, dtype=np.float32)
    if h.ndim not in (3, 4) or h.shape[-1] < TOF_C:
        raise ValueError(f"bad hists shape: {h.shape}")
    sat_value = h[..., 62] * TAIL_BASE + h[..., 63]
    sat_value = np.where(sat_value > 0.0, sat_value, float(PULSES))
    hist_k = h[..., PEAK_BIN_LO : PEAK_BIN_HI + 1] * (float(PULSES) / sat_value[..., None])
    return np.max(hist_k, axis=-1).astype(np.float32, copy=False)


def _run_infer(net, device, hists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hist (H,W,64) -> pred_depth, snr, conf, peak, reflectance.

    peak 不取网络输出，改为 bin [PEAK_BIN_LO, PEAK_BIN_HI] 上 hist_k 的最大值。
    """
    import torch

    h = np.asarray(hists, dtype=np.float32)
    with torch.inference_mode():
        inp = torch.from_numpy(h).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=True)
        dist_t, conf_t, _peak_t, reflectance_t, snr_t = net(inp)
        pred_depth = dist_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        snr = snr_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        conf = conf_t[0, 0].cpu().numpy().astype(np.float32, copy=False)
        reflectance = reflectance_t[0, 0].cpu().numpy().astype(np.float32, copy=False)

    peak = _compute_peak_map(h)

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


def _run_infer_batch(
    net, device, hists_nhwc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """hist (N,H,W,64) -> dist/snr/conf/peak/refl，各 (N,H,W)。"""
    import torch

    h = np.asarray(hists_nhwc, dtype=np.float32)
    if h.ndim != 4 or h.shape[1:] != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad batch hists shape: {h.shape}")
    with torch.inference_mode():
        inp = torch.from_numpy(h).permute(0, 3, 1, 2).to(
            device=device, dtype=torch.float32, non_blocking=True,
        )
        dist_t, conf_t, _peak_t, reflectance_t, snr_t = net(inp)
        dist = dist_t[:, 0].cpu().numpy().astype(np.float32, copy=False)
        snr = snr_t[:, 0].cpu().numpy().astype(np.float32, copy=False)
        conf = conf_t[:, 0].cpu().numpy().astype(np.float32, copy=False)
        reflectance = reflectance_t[:, 0].cpu().numpy().astype(np.float32, copy=False)

    peak = _compute_peak_map(h)

    invalid = (~np.isfinite(dist)) | (dist <= 0.0)
    if np.any(invalid):
        dist = dist.copy()
        snr = snr.copy()
        conf = conf.copy()
        peak = peak.copy()
        reflectance = reflectance.copy()
        dist[invalid] = 0.0
        snr[invalid] = 0.0
        conf[invalid] = 0.0
        peak[invalid] = 0.0
        reflectance[invalid] = 0.0

    return dist, snr, conf, peak, reflectance


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
    mirror: bool,
    bag_dist_m: np.ndarray | None = None,
    depth_near_m: float = DEPTH_COLOR_NEAR_M,
    depth_far_m: float = DEPTH_COLOR_FAR_M,
    peak_lo: float = PEAK_DISP_LO,
    peak_hi: float = PEAK_DISP_HI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """渲染面板 + 直方图 + bins 长条。

    BAG 模式布局（2x2）:
        第一行: DISTANCE | BAG_DIST
        第二行: PEAK     | REFLECTANCE

    ADB 模式布局（1+2）:
        第一行: DISTANCE（单图居左，右侧留黑）
        第二行: PEAK     | REFLECTANCE

    Args:
        bag_dist_m: 可选，(H,W) float32 距离(米)，来自 bag 中 alg/dtof_depth。
        depth_near_m / depth_far_m: 距离伪彩红/蓝端点(米)。
        peak_lo / peak_hi: PEAK 灰度显示范围。

    Returns: (view_bgr, hist_bgr, strip_bgr, tof_px, tof_py)
    """
    import cv2

    dist_for_disp = np.where(conf > 0.5, dist, 0.0)
    dist_big = cv2.resize(
        _orient_for_display(_colorize_depth(dist_for_disp, depth_near_m, depth_far_m), rotate_90, mirror),
        (show_w, show_h), interpolation=cv2.INTER_NEAREST,
    )

    peak_u8 = _peak_to_u8(peak, peak_lo, peak_hi)
    peak_bgr = cv2.cvtColor(
        cv2.resize(_orient_for_display(peak_u8, rotate_90, mirror), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    refl_u8 = np.clip(np.rint(np.clip(reflectance, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    refl_bgr = cv2.cvtColor(
        cv2.resize(_orient_for_display(refl_u8, rotate_90, mirror), (show_w, show_h), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )

    if bag_dist_m is not None:
        bag_panel = cv2.resize(
            _orient_for_display(_colorize_depth(bag_dist_m, depth_near_m, depth_far_m), rotate_90, mirror),
            (show_w, show_h), interpolation=cv2.INTER_NEAREST,
        )

    mx = int(np.clip(mouse_x, 0, show_w * 2 - 1))
    my = int(np.clip(mouse_y, 0, show_h * 2 - 1))
    tile_x0 = 0 if mx < show_w else show_w
    tile_y0 = 0 if my < show_h else show_h
    px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, show_w, show_h, rotate_90, mirror)
    dx, dy = _pixel_to_disp_xy(px, py, show_w, show_h, rotate_90, mirror)

    panels = [dist_big, peak_bgr, refl_bgr]
    if bag_dist_m is not None:
        panels.append(bag_panel)
    for img in panels:
        _draw_marker(img, dx, dy)

    _with_text(dist_big, "DISTANCE")
    _with_text(peak_bgr, f"PEAK [{peak_lo:.0f},{peak_hi:.0f}]")
    _with_text(refl_bgr, "REFLECTANCE")
    if bag_dist_m is not None:
        _with_text(bag_panel, "BAG_DIST")

    if bag_dist_m is not None:
        row1 = np.hstack([dist_big, bag_panel])
    else:
        row1 = np.hstack([dist_big, np.zeros((show_h, show_w, 3), dtype=np.uint8)])
    row2 = np.hstack([peak_bgr, refl_bgr])

    view = np.vstack([row1, row2])

    effective_bin = _effective_bin_from_output(
        hists[py, px, :],
        float(dist[py, px]),
        float(conf[py, px]),
    )
    hist_img = _render_histogram_bgr(hists[py, px, :], effective_bin=effective_bin)
    strip_img = _render_bins_strip_bgr(hists[py, px, :], px, py, effective_bin=effective_bin)
    return view, hist_img, strip_img, px, py


# ======================== 单窗口仪表盘合成 ========================

DASH_PAD = 14
DASH_GAP = 14
DASH_BANNER_H = 46
DASH_BG = (24, 22, 20)               # 深色背景 (BGR)
DASH_PANEL_BG = (34, 31, 28)
DASH_PANEL_BORDER = (78, 72, 66)
DASH_ACCENT = (60, 190, 255)         # 强调色（暖橙）
DASH_TITLE_COLOR = (244, 244, 246)
DASH_SUB_COLOR = (172, 172, 182)

# 主视图在合成画布中的固定偏移（鼠标坐标换算用）
MAIN_OFFSET_X = DASH_PAD
MAIN_OFFSET_Y = DASH_BANNER_H + DASH_PAD


def _render_depth_colorbar(
    h: int,
    near_m: float,
    far_m: float,
    w: int = CBAR_W,
    query_m: float | None = None,
) -> tuple[np.ndarray, dict]:
    """绘制竖直距离色条（上红下蓝），返回 (图像, ramp 局部几何)。"""
    import cv2  # type: ignore

    near_m, far_m = _normalize_depth_color_range(near_m, far_m)
    img = np.zeros((max(int(h), 1), max(int(w), 1), 3), dtype=np.uint8)
    img[:] = DASH_PANEL_BG

    top_pad, bot_pad = 36, 36
    bar_x0, bar_w = 10, 18
    y0 = top_pad
    y1 = max(y0 + 1, int(h) - bot_pad)
    ramp_h = y1 - y0

    ramp = np.linspace(255.0, 0.0, ramp_h, dtype=np.float32).reshape(-1, 1)
    u8 = np.clip(np.rint(ramp), 0, 255).astype(np.uint8)
    u8 = np.repeat(u8, bar_w, axis=1)
    cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    img[y0:y1, bar_x0:bar_x0 + bar_w] = cmap
    cv2.rectangle(img, (bar_x0 - 1, y0 - 1), (bar_x0 + bar_w, y1), DASH_PANEL_BORDER, 1, cv2.LINE_AA)

    cv2.putText(img, "RED", (bar_x0 + bar_w + 6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{near_m:.1f}m", (bar_x0 + bar_w + 4, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DASH_TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, f"{far_m:.1f}m", (bar_x0 + bar_w + 4, int(h) - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, DASH_TITLE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, "BLUE", (bar_x0 + bar_w + 4, int(h) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 80), 1, cv2.LINE_AA)

    geom = {
        "y0": y0,
        "y1": y1,
        "x0": bar_x0,
        "x1": bar_x0 + bar_w,
        "near_m": near_m,
        "far_m": far_m,
    }

    if query_m is not None and np.isfinite(query_m):
        span = far_m - near_m
        frac = float(np.clip((float(query_m) - near_m) / span, 0.0, 1.0))
        qy = int(round(y0 + frac * (ramp_h - 1)))
        cv2.line(img, (2, qy), (bar_x0 + bar_w + 2, qy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            img, f"{float(query_m):.2f}m",
            (2, max(14, qy - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img, geom


def _cbar_query_m(mouse_x: int, mouse_y: int, cbar_rect: dict) -> float | None:
    """若鼠标在色条 ramp 上，返回该位置对应距离；否则 None。"""
    x0 = cbar_rect.get("x0")
    x1 = cbar_rect.get("x1")
    y0 = cbar_rect.get("y0")
    y1 = cbar_rect.get("y1")
    if x0 is None or x1 is None or y0 is None or y1 is None or y1 <= y0:
        return None
    if not (x0 - 6 <= mouse_x <= x1 + 40 and y0 <= mouse_y <= y1):
        return None
    frac = (float(mouse_y) - float(y0)) / float(y1 - y0)
    frac = float(np.clip(frac, 0.0, 1.0))
    near_m = float(cbar_rect.get("near_m", DEPTH_COLOR_NEAR_M))
    far_m = float(cbar_rect.get("far_m", DEPTH_COLOR_FAR_M))
    return near_m + frac * (far_m - near_m)


def _draw_seek_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    progress_frac: float,
    progress_label: str = "",
    bar_rect_out: dict | None = None,
) -> None:
    """在右下角绘制 BAG 回放进度条卡片。

    bar_rect_out: 若提供，写入进度条可点击区域 {x0,y0,x1,y1}（画布坐标）。
    """
    import cv2  # type: ignore

    cv2.rectangle(canvas, (x, y), (x + w, y + h), DASH_PANEL_BG, -1)
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), DASH_PANEL_BORDER, 1, cv2.LINE_AA)

    _draw_progress_bar(
        canvas, x, w, y + (h - 14) // 2, y + h,
        float(progress_frac), progress_label, bar_rect_out,
    )


def _draw_progress_bar(
    canvas: np.ndarray,
    x: int,
    w: int,
    top_y: int,
    card_bottom: int,
    frac: float,
    label: str,
    bar_rect_out: dict | None,
) -> None:
    """在信息卡底部绘制可拖动进度条（BAG 回放用）。"""
    import cv2  # type: ignore

    bar_h = 14
    bx0 = x + 16
    bx1 = x + w - 16
    by0 = int(top_y)
    by1 = by0 + bar_h
    if bx1 <= bx0 or by1 + 6 > card_bottom:
        return

    if label:
        cv2.putText(canvas, label, (bx0, by0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, DASH_SUB_COLOR, 1, cv2.LINE_AA)

    f = float(np.clip(frac, 0.0, 1.0))
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (52, 48, 44), -1)
    fx = bx0 + int(round((bx1 - bx0) * f))
    if fx > bx0:
        cv2.rectangle(canvas, (bx0, by0), (fx, by1), DASH_ACCENT, -1)
    cv2.rectangle(canvas, (bx0 - 1, by0 - 1), (bx1, by1), DASH_PANEL_BORDER, 1, cv2.LINE_AA)
    cv2.circle(canvas, (int(np.clip(fx, bx0, bx1)), by0 + bar_h // 2), 7,
               DASH_TITLE_COLOR, -1, cv2.LINE_AA)

    if bar_rect_out is not None:
        bar_rect_out["x0"] = bx0
        bar_rect_out["x1"] = bx1
        bar_rect_out["y0"] = by0
        bar_rect_out["y1"] = by1


_FIELD_LABELS = {
    "near_m": "最近距离 (m)",
    "far_m": "最远距离 (m)",
    "peak_lo": "最小亮度",
    "peak_hi": "最大亮度",
}
_FIELD_KEYS = ("near_m", "far_m", "peak_lo", "peak_hi")
_CN_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)


def _get_cn_font(size: int = 16):
    try:
        from PIL import ImageFont  # type: ignore
    except Exception:
        return None
    for path in _CN_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    try:
        from PIL import ImageFont  # type: ignore
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_label_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: tuple[int, int, int] = DASH_SUB_COLOR,
    size: int = 15,
) -> None:
    """绘制中文/英文标签；优先 PIL，失败回退 OpenCV ASCII。"""
    import cv2  # type: ignore

    font = _get_cn_font(size)
    if font is None:
        cv2.putText(
            canvas, text, (x, y + size - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA,
        )
        return
    try:
        from PIL import Image, ImageDraw  # type: ignore

        h, w = canvas.shape[:2]
        # 只在局部贴文字，避免整图画布往返
        tw = max(8, int(font.getlength(text)) + 4) if hasattr(font, "getlength") else 180
        th = size + 8
        x0 = max(0, min(x, w - 1))
        y0 = max(0, min(y, h - 1))
        x1 = max(x0 + 1, min(w, x0 + tw))
        y1 = max(y0 + 1, min(h, y0 + th))
        patch = canvas[y0:y1, x0:x1]
        if patch.size == 0:
            return
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        rgb_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((2, 1), text, font=font, fill=rgb_color)
        canvas[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            canvas, text, (x, y + size - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA,
        )


def _format_disp_value(key: str, value: float) -> str:
    if key in ("near_m", "far_m"):
        return f"{float(value):.1f}"
    return f"{int(round(float(value)))}"


def _begin_field_edit(ctrl: dict, key: str) -> None:
    if key not in _FIELD_KEYS:
        return
    ctrl["focus"] = key
    ctrl["edit"] = _format_disp_value(key, float(ctrl.get(key, 0.0)))
    ctrl["drag"] = None


def _cancel_field_edit(ctrl: dict) -> None:
    ctrl["focus"] = None
    ctrl["edit"] = ""


def _commit_field_edit(ctrl: dict) -> bool:
    """提交当前输入框；成功返回 True。"""
    key = ctrl.get("focus")
    if key not in _FIELD_KEYS:
        return False
    raw = str(ctrl.get("edit", "")).strip()
    if raw == "" or raw in (".", "-", "-."):
        _cancel_field_edit(ctrl)
        return False
    try:
        val = float(raw)
    except Exception:
        _cancel_field_edit(ctrl)
        return False
    if key in ("near_m", "far_m"):
        ctrl[key] = round(val, 1)
    else:
        ctrl[key] = float(int(round(val)))
    _sync_disp_ctrl(ctrl)
    _cancel_field_edit(ctrl)
    return True


def _handle_disp_key(ctrl: dict, key_code: int) -> bool:
    """输入框键盘处理；返回 True 表示已消费该按键。"""
    focus = ctrl.get("focus")
    if focus is None:
        return False

    k = int(key_code) & 0xFF
    # Enter
    if k in (13, 10):
        _commit_field_edit(ctrl)
        return True
    # Esc
    if k == 27:
        _cancel_field_edit(ctrl)
        return True
    # Backspace
    if k in (8, 127):
        ctrl["edit"] = str(ctrl.get("edit", ""))[:-1]
        return True
    # digits / dot
    if ord("0") <= k <= ord("9"):
        edit = str(ctrl.get("edit", ""))
        if len(edit) < 12:
            ctrl["edit"] = edit + chr(k)
        return True
    if k == ord(".") and focus in ("near_m", "far_m"):
        edit = str(ctrl.get("edit", ""))
        if "." not in edit and len(edit) < 12:
            ctrl["edit"] = edit + "."
        return True
    # 编辑中吞掉空格等其它键，避免误触发播放/录像
    return True


def _draw_text_field(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value_text: str,
    focused: bool,
    accent_bgr: tuple[int, int, int],
    rect_out: dict | None = None,
) -> None:
    """绘制带标签的文字输入框。"""
    import cv2  # type: ignore

    _draw_label_text(canvas, label, x, y, DASH_SUB_COLOR, size=14)
    box_y = y + 20
    box_h = h - 22
    fill = (44, 40, 36) if not focused else (58, 48, 36)
    border = accent_bgr if focused else DASH_PANEL_BORDER
    cv2.rectangle(canvas, (x, box_y), (x + w, box_y + box_h), fill, -1)
    cv2.rectangle(canvas, (x, box_y), (x + w, box_y + box_h), border, 1, cv2.LINE_AA)
    if focused:
        cv2.rectangle(canvas, (x + 1, box_y + 1), (x + w - 1, box_y + box_h - 1), accent_bgr, 1, cv2.LINE_AA)

    show = value_text + ("|" if focused else "")
    cv2.putText(
        canvas, show, (x + 10, box_y + box_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, DASH_TITLE_COLOR, 1, cv2.LINE_AA,
    )

    if rect_out is not None:
        rect_out["x0"] = x
        rect_out["x1"] = x + w
        rect_out["y0"] = box_y
        rect_out["y1"] = box_y + box_h


def _draw_disp_control_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    near_m: float,
    far_m: float,
    peak_lo: float,
    peak_hi: float,
    progress_frac: float | None = None,
    progress_label: str = "",
    bar_rect_out: dict | None = None,
    field_rects_out: dict | None = None,
    focus_key: str | None = None,
    edit_text: str = "",
) -> None:
    """右下角显示控制卡：可选 seek + 四个文字输入框。"""
    import cv2  # type: ignore

    if w < 180 or h < 100:
        return

    cv2.rectangle(canvas, (x, y), (x + w, y + h), DASH_PANEL_BG, -1)
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), DASH_PANEL_BORDER, 1, cv2.LINE_AA)

    pad_x = 14
    cur_y = y + 18
    _draw_label_text(canvas, "显示范围", x + pad_x, cur_y - 2, DASH_ACCENT, size=16)
    # 标题与下方内容拉开，避免和 FRAME 标签重叠
    cur_y += 30

    if progress_frac is not None:
        # 先留出 FRAME 文字高度，再画进度条
        if progress_label:
            _draw_label_text(
                canvas, progress_label, x + pad_x, cur_y, DASH_TITLE_COLOR, size=14,
            )
            cur_y += 22
        bar_h = 14
        _draw_progress_bar(
            canvas, x, w, cur_y, y + h,
            float(progress_frac), "", bar_rect_out,
        )
        cur_y += bar_h + 22

    near_m, far_m = _normalize_depth_color_range(near_m, far_m)
    peak_lo, peak_hi = _normalize_peak_disp_range(peak_lo, peak_hi)
    values = {
        "near_m": near_m,
        "far_m": far_m,
        "peak_lo": peak_lo,
        "peak_hi": peak_hi,
    }
    accents = {
        "near_m": (60, 80, 255),
        "far_m": (220, 160, 60),
        "peak_lo": (120, 200, 140),
        "peak_hi": (120, 200, 140),
    }

    gap = 12
    col_w = (w - pad_x * 2 - gap) // 2
    row_h = 54
    row_gap = 12
    positions = (
        ("near_m", 0, 0),
        ("far_m", 1, 0),
        ("peak_lo", 0, 1),
        ("peak_hi", 1, 1),
    )

    if field_rects_out is not None:
        field_rects_out.clear()

    for key, col, row in positions:
        fx = x + pad_x + col * (col_w + gap)
        fy = cur_y + row * (row_h + row_gap)
        if fy + row_h > y + h - 24:
            continue
        focused = focus_key == key
        text = edit_text if focused else _format_disp_value(key, float(values[key]))
        rect: dict = {}
        _draw_text_field(
            canvas, fx, fy, col_w, row_h,
            _FIELD_LABELS[key], text, focused, accents[key], rect_out=rect,
        )
        if field_rects_out is not None and rect:
            field_rects_out[key] = rect

    tip_y = y + h - 12
    tip = "点击输入  Enter确认  Esc取消"
    if focus_key:
        tip = f"正在编辑: {_FIELD_LABELS.get(str(focus_key), '')}"
    _draw_label_text(canvas, tip, x + pad_x, tip_y - 14, (140, 140, 150), size=13)


def _hit_field_key(x: int, y: int, field_rects: dict) -> str | None:
    for key, r in field_rects.items():
        if r.get("x0") is None:
            continue
        if int(r["x0"]) <= x <= int(r["x1"]) and int(r["y0"]) <= y <= int(r["y1"]):
            return str(key)
    return None


def _compose_dashboard(
    view: np.ndarray,
    hist_img: np.ndarray,
    strip_img: np.ndarray,
    status_text: str = "",
    hover_lines: list[str] | None = None,
    progress_frac: float | None = None,
    progress_label: str = "",
    bar_rect_out: dict | None = None,
    depth_near_m: float = DEPTH_COLOR_NEAR_M,
    depth_far_m: float = DEPTH_COLOR_FAR_M,
    peak_lo: float = PEAK_DISP_LO,
    peak_hi: float = PEAK_DISP_HI,
    cbar_rect_out: dict | None = None,
    field_rects_out: dict | None = None,
    focus_key: str | None = None,
    edit_text: str = "",
    mouse_xy: tuple[int, int] | None = None,
) -> np.ndarray:
    """把主视图 / 色条 / 直方图 / bins 表格合成到单张深色仪表盘画布。

    布局：
        ┌── banner ──────────────────────────────────────┐
        │ NN ToF Realtime                     <status>    │
        ├──────────┬──────┬────────┬─────────────────────┤
        │  MAIN    │ CBAR │  BINS  │  HOVER INFO          │
        │ (2x2)    │色条  │ (table)│  HIST                │
        │          │      │        │  显示范围输入(右下)   │
        └──────────┴──────┴────────┴─────────────────────┘
    主视图固定置于 (MAIN_OFFSET_X, MAIN_OFFSET_Y)，鼠标坐标据此换算。
    """
    import cv2  # type: ignore

    mh, mw = view.shape[:2]
    sh, sw = strip_img.shape[:2]
    hh, hw = hist_img.shape[:2]
    hover = [ln for ln in (hover_lines or []) if ln]
    hover_row_h = 22
    # +1 预留色条查询行 cmap Xm
    hover_h = (10 + (len(hover) + 1) * hover_row_h + 8) if hover else 0

    pad, gap, banner = DASH_PAD, DASH_GAP, DASH_BANNER_H
    right_h = hover_h + (gap if hover_h else 0) + hh
    content_h = max(mh, sh, right_h)

    # 先用鼠标位置对色条做一次查询（几何与放置一致）
    cbar_x = pad + mw + gap
    cbar_y = banner + pad
    prelim_near, prelim_far = _normalize_depth_color_range(depth_near_m, depth_far_m)
    query_m = None
    if mouse_xy is not None:
        # 色条 ramp 在局部坐标，需先合成后再精确查询；这里用与 _render 相同的局部几何预估
        local = {
            "x0": cbar_x + 10,
            "x1": cbar_x + 10 + 18,
            "y0": cbar_y + 36,
            "y1": cbar_y + max(37, mh - 36),
            "near_m": prelim_near,
            "far_m": prelim_far,
        }
        query_m = _cbar_query_m(int(mouse_xy[0]), int(mouse_xy[1]), local)

    cbar_img, cbar_geom = _render_depth_colorbar(
        mh, depth_near_m, depth_far_m, w=CBAR_W, query_m=query_m,
    )
    cw = int(cbar_img.shape[1])

    main_x, main_y = pad, banner + pad
    strip_x, strip_y = cbar_x + cw + gap, banner + pad
    hist_x = strip_x + sw + gap
    hover_y = banner + pad
    hist_y = hover_y + hover_h + (gap if hover_h else 0)

    canvas_w = hist_x + hw + pad
    canvas_h = banner + pad + content_h + pad

    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = DASH_BG

    cv2.rectangle(canvas, (0, 0), (canvas_w, banner), (38, 35, 32), -1)
    cv2.line(canvas, (0, banner - 1), (canvas_w, banner - 1), DASH_ACCENT, 2, cv2.LINE_AA)
    cv2.circle(canvas, (pad + 6, banner // 2), 6, DASH_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(canvas, "NN  ToF  Realtime", (pad + 22, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, DASH_TITLE_COLOR, 2, cv2.LINE_AA)
    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(canvas, status_text, (canvas_w - tw - pad, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, DASH_SUB_COLOR, 1, cv2.LINE_AA)

    def place(img: np.ndarray, x: int, y: int) -> None:
        ih, iw = img.shape[:2]
        canvas[y:y + ih, x:x + iw] = img
        cv2.rectangle(canvas, (x - 1, y - 1), (x + iw, y + ih), DASH_PANEL_BORDER, 1, cv2.LINE_AA)

    place(view, main_x, main_y)
    place(cbar_img, cbar_x, cbar_y)
    place(strip_img, strip_x, strip_y)

    if cbar_rect_out is not None:
        cbar_rect_out["x0"] = cbar_x + int(cbar_geom["x0"])
        cbar_rect_out["x1"] = cbar_x + int(cbar_geom["x1"])
        cbar_rect_out["y0"] = cbar_y + int(cbar_geom["y0"])
        cbar_rect_out["y1"] = cbar_y + int(cbar_geom["y1"])
        cbar_rect_out["near_m"] = float(cbar_geom["near_m"])
        cbar_rect_out["far_m"] = float(cbar_geom["far_m"])

    if hover_h > 0:
        cv2.rectangle(canvas, (hist_x, hover_y), (hist_x + hw, hover_y + hover_h), DASH_PANEL_BG, -1)
        cv2.rectangle(
            canvas, (hist_x - 1, hover_y - 1), (hist_x + hw, hover_y + hover_h),
            DASH_PANEL_BORDER, 1, cv2.LINE_AA,
        )
        lines = list(hover)
        if query_m is not None:
            lines.append(f"cmap {float(query_m):.2f}m")
        for i, ln in enumerate(lines):
            cv2.putText(
                canvas, ln, (hist_x + 12, hover_y + 10 + (i + 1) * hover_row_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, DASH_TITLE_COLOR, 1, cv2.LINE_AA,
            )

    place(hist_img, hist_x, hist_y)

    info_x = hist_x
    info_y = hist_y + hh + gap
    info_w = hw
    info_h = (main_y + mh) - info_y
    if info_h > 80:
        _draw_disp_control_card(
            canvas, info_x, info_y, info_w, info_h,
            near_m=depth_near_m,
            far_m=depth_far_m,
            peak_lo=peak_lo,
            peak_hi=peak_hi,
            progress_frac=progress_frac,
            progress_label=progress_label,
            bar_rect_out=bar_rect_out,
            field_rects_out=field_rects_out,
            focus_key=focus_key,
            edit_text=edit_text,
        )

    return canvas


# ======================== BAG/MCAP 解析 ========================

def _parse_vp_tof_info(payload: bytes):
    """解析 VpTofInfo 消息，返回 (TofInfoHeader, hist(30,40,64)) 或 None。

    自动兼容两种协议（按 payload 长度判别，新包比旧包多 tof_metadata 5120B）：
      - V2(新): header(40) -> reserved(16) -> tof_metadata(5120) -> tof_raw
                字段顺序变化，且新增 vspad。raw 偏移 = 40+16+5120 = 5176。
      - V1(旧): header(40) -> tof_raw -> reserved(16)。raw 偏移 = 40。
    """
    n = len(payload)
    if n >= _BAG_PAYLOAD_SIZE_V2:
        # 新协议 V2
        vals = struct.unpack_from(_VPI_HEADER_FMT_V2, payload, 0)
        header = TofInfoHeader(
            is_valid=vals[1], frame_id=vals[4], timestamp_us=vals[0],
            work_mode=vals[2], bin_mode=vals[3], light_count=vals[5],
            expo_time=vals[6], pulse_width=vals[7], rx_temp=vals[8], tx_temp=vals[9],
            vspad=vals[10],
        )
        raw_off = _VPI_HEADER_SIZE_V2 + _BAG_RESERVED_BYTES + _BAG_METADATA_BYTES  # 5176
    elif n >= _BAG_PAYLOAD_SIZE:
        # 旧协议 V1
        vals = struct.unpack_from(_VPI_HEADER_FMT, payload, 0)
        header = TofInfoHeader(
            is_valid=vals[0], frame_id=vals[1], timestamp_us=vals[2],
            work_mode=vals[3], bin_mode=vals[4], light_count=vals[5],
            expo_time=vals[6], pulse_width=vals[7], rx_temp=vals[8], tx_temp=vals[9],
        )
        raw_off = _VPI_HEADER_SIZE
    else:
        return None

    raw_all = np.frombuffer(payload[raw_off:raw_off + _BAG_RAW_BYTES], dtype="<u2")
    if raw_all.size < _BAG_PIXELS * TOF_C:
        return None
    hist = raw_all[:_BAG_PIXELS * TOF_C].reshape(TOF_H, TOF_W, TOF_C).copy()
    return header, hist


def _parse_vp_dtof_depth(payload: bytes):
    """解析 VpDtofDepth 消息，返回 (frame_id, dist_mm(30,40), conf(30,40)) 或 None。"""
    if len(payload) < _DEPTH_PAYLOAD_SIZE:
        return None
    data = memoryview(payload)
    frame_id = int.from_bytes(data[8:12], byteorder="little", signed=False)
    dist_off = _DEPTH_HDR_SIZE
    conf_off = dist_off + _DEPTH_DIST_BYTES
    dist = np.frombuffer(data[dist_off:dist_off + _DEPTH_DIST_BYTES], dtype="<u2").reshape(TOF_H, TOF_W).copy()
    conf = np.frombuffer(data[conf_off:conf_off + _DEPTH_CONF_BYTES], dtype=np.uint8).reshape(TOF_H, TOF_W).copy()
    return frame_id, dist, conf


def _parse_mcap_message(topic: str, payload: bytes):
    """按 topic 解析一条消息，无法识别则返回 None。"""
    if topic == _BAG_RAW_TOPIC:
        return _parse_vp_tof_info(payload)
    if topic == _BAG_DEPTH_TOPIC:
        return _parse_vp_dtof_depth(payload)
    return None


def _mcap_chunk_jobs(bag_path: Path, topics: set[str]):
    """读取 summary，返回 (chunk 偏移列表, channel_id->topic)；无索引则 None。"""
    from mcap.reader import make_reader

    with bag_path.open("rb") as f:
        try:
            summary = make_reader(f).get_summary()
        except Exception:
            return None
    if summary is None or not summary.chunk_indexes:
        return None
    channel_topics = {cid: (ch.topic or "") for cid, ch in summary.channels.items()}
    offsets: list[int] = []
    for chunk_index in summary.chunk_indexes:
        for channel_id in chunk_index.message_index_offsets:
            if channel_topics.get(channel_id) in topics:
                offsets.append(int(chunk_index.chunk_start_offset))
                break
    if not offsets:
        return None
    return offsets, channel_topics


def _parse_mcap_chunk_range(
    bag_path: str,
    offsets: list[int],
    channel_topics: dict[int, str],
    wanted: tuple[str, ...],
) -> tuple[dict[str, list], dict[str, int]]:
    """顺序解析一段 chunk，解压与 numpy 拷贝可与其它线程并行。"""
    from mcap.data_stream import ReadDataStream
    from mcap.records import Chunk, Message
    from mcap.stream_reader import breakup_chunk

    wanted_set = set(wanted)
    out: dict[str, list] = {t: [] for t in wanted}
    cnt: dict[str, int] = {t: 0 for t in wanted}
    with open(bag_path, "rb") as f:
        for off in offsets:
            f.seek(off + 1 + 8)
            chunk = Chunk.read(ReadDataStream(f))
            for record in breakup_chunk(chunk):
                if not isinstance(record, Message):
                    continue
                topic = channel_topics.get(record.channel_id, "")
                if topic not in wanted_set:
                    continue
                cnt[topic] += 1
                parsed = _parse_mcap_message(topic, record.data)
                if parsed is not None:
                    out[topic].append(parsed)
    return out, cnt


def _load_bag_topics_sequential(bag_path: Path, topics: tuple[str, ...]) -> tuple[dict[str, list], dict[str, int]]:
    """无 chunk 索引时的单线程顺序读取。"""
    from mcap.reader import NonSeekingReader, make_reader

    wanted = set(topics)
    out: dict[str, list] = {t: [] for t in topics}
    cnt: dict[str, int] = {t: 0 for t in topics}

    def consume(msg_iter) -> None:
        for _, channel, message in msg_iter:
            topic = channel.topic or ""
            if topic not in wanted:
                continue
            cnt[topic] += 1
            parsed = _parse_mcap_message(topic, message.data)
            if parsed is not None:
                out[topic].append(parsed)

    with bag_path.open("rb") as f:
        try:
            consume(make_reader(f).iter_messages())
        except Exception as exc:
            print(f"[WARN] {bag_path.name}: make_reader 失败 ({exc})，顺序读取")
            f.seek(0)
            try:
                consume(NonSeekingReader(f).iter_messages(log_time_order=False))
            except Exception as exc2:
                print(f"[WARN] {bag_path.name}: 顺序读取结束 ({exc2})，已保留可读帧")
    return out, cnt


def _split_even(items: list[int], n: int) -> list[list[int]]:
    n = max(1, min(n, len(items)))
    size = (len(items) + n - 1) // n
    return [items[i:i + size] for i in range(0, len(items), size)]


def _finalize_bag_topics(
    out: dict[str, list], cnt: dict[str, int],
) -> tuple[dict[str, list], dict[str, int]]:
    if _BAG_RAW_TOPIC in out:
        out[_BAG_RAW_TOPIC].sort(key=lambda x: (x[0].timestamp_us, x[0].frame_id))
    return out, cnt


def _load_bag_topics(bag_path: Path, topics: tuple[str, ...]) -> tuple[dict[str, list], dict[str, int]]:
    """按 chunk 并行加载若干 topic；无索引则回退顺序读取。"""
    jobs = _mcap_chunk_jobs(bag_path, set(topics))
    if jobs is None:
        return _finalize_bag_topics(*_load_bag_topics_sequential(bag_path, topics))

    offsets, channel_topics = jobs
    n_workers = min(_BAG_LOAD_WORKERS, len(offsets))
    groups = _split_even(offsets, n_workers)
    print(
        f"[INFO] 并行加载 {bag_path.name}: chunks={len(offsets)} workers={len(groups)}"
    )
    out: dict[str, list] = {t: [] for t in topics}
    cnt: dict[str, int] = {t: 0 for t in topics}
    try:
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            futs = [
                ex.submit(
                    _parse_mcap_chunk_range,
                    str(bag_path),
                    group,
                    channel_topics,
                    topics,
                )
                for group in groups
            ]
            for fut in futs:
                part, part_cnt = fut.result()
                for topic in topics:
                    out[topic].extend(part.get(topic, []))
                    cnt[topic] += int(part_cnt.get(topic, 0))
    except Exception as exc:
        print(f"[WARN] 并行加载失败 ({exc})，改为顺序读取")
        return _finalize_bag_topics(*_load_bag_topics_sequential(bag_path, topics))

    return _finalize_bag_topics(out, cnt)


def _depth_list_to_map(items: list) -> dict:
    depth_map: dict = {}
    for parsed in items:
        fid, dist_mm, conf = parsed
        depth_map[fid] = (dist_mm, conf)
    return depth_map


def _load_bag_depth_map(bag_path: Path) -> dict:
    """从 BAG/MCAP 加载 alg/dtof_depth topic，返回 {frame_id: (dist_mm, conf)}。"""
    print(f"[INFO] 加载 bag depth ({_BAG_DEPTH_TOPIC}): {bag_path.name}")
    loaded, cnt = _load_bag_topics(bag_path, (_BAG_DEPTH_TOPIC,))
    depth_map = _depth_list_to_map(loaded[_BAG_DEPTH_TOPIC])
    print(f"[OK] bag depth 帧数: {len(depth_map)} (topic={cnt[_BAG_DEPTH_TOPIC]})")
    return depth_map


def _load_bag_frames(bag_path: Path) -> list:
    """从 BAG/MCAP 文件加载所有 VpTofInfo 帧，返回 [(TofInfoHeader, hist), ...]。"""
    print(f"[INFO] 扫描: {bag_path}")
    loaded, cnt = _load_bag_topics(bag_path, (_BAG_RAW_TOPIC,))
    frames = loaded[_BAG_RAW_TOPIC]
    print(f"[OK] {bag_path.name}: topic帧={cnt[_BAG_RAW_TOPIC]}, 有效帧={len(frames)}")
    return frames


# ======================== BAG 模式 ========================

def _run_bag_mode(bag_path_str: str) -> int:
    """从 BAG/MCAP 文件读取帧，批量推理后交互式浏览。"""
    bag_path = Path(bag_path_str).resolve()
    if not bag_path.exists():
        print(f"[ERR] 文件不存在: {bag_path}")
        return 1

    return _run_bag_paths_mode([bag_path])


def _bag_path_sort_key(path: Path) -> tuple:
    """让 0.bag, 1.bag, ... 10.bag 按数字顺序排列。"""
    try:
        stem_key: tuple = (0, int(path.stem))
    except ValueError:
        stem_key = (1, path.stem.lower())
    return tuple(part.lower() for part in path.parent.parts) + stem_key


def _discover_dtof_bags(dir_path: Path) -> list[Path]:
    """递归查找设备日志中 dtof_depth_bag 目录里的 BAG/MCAP 文件。"""
    bags = [
        path
        for path in dir_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in (".bag", ".mcap")
        and any(part.lower() == "dtof_depth_bag" for part in path.parts)
    ]
    return sorted(bags, key=_bag_path_sort_key)


def _run_bag_paths_mode(bag_paths: list[Path]) -> int:
    """合并一个或多个 BAG/MCAP 文件中的 ToF raw 帧并回放。"""
    merged_frames: list[tuple[Path, TofInfoHeader, np.ndarray]] = []
    bag_depth_map: dict = {}

    for bag_path in bag_paths:
        print(f"[INFO] 扫描: {bag_path}")
        try:
            loaded, cnt = _load_bag_topics(
                bag_path, (_BAG_RAW_TOPIC, _BAG_DEPTH_TOPIC),
            )
        except Exception as exc:
            print(f"[WARN] 跳过无法读取的录包 {bag_path.name}: {exc}")
            continue

        frames = loaded[_BAG_RAW_TOPIC]
        depth_items = loaded[_BAG_DEPTH_TOPIC]
        bag_depth_map.update(_depth_list_to_map(depth_items))
        merged_frames.extend((bag_path, header, hist) for header, hist in frames)
        print(
            f"[OK] {bag_path.name}: tof topic={cnt[_BAG_RAW_TOPIC]} 有效={len(frames)}, "
            f"depth topic={cnt[_BAG_DEPTH_TOPIC]} 有效={len(depth_items)}"
        )

    merged_frames.sort(key=lambda item: (item[1].timestamp_us, item[1].frame_id))
    frames = [(header, hist) for _, header, hist in merged_frames]
    if not frames:
        print("[ERR] 没有有效帧")
        return 1

    hists = [f[1] for f in frames]
    labels = [
        f"{bag_path.name}  frame_id {header.frame_id}"
        for bag_path, header, _ in merged_frames
    ]
    bag_dists = [
        _bag_depth_entry_to_m(bag_depth_map.get(f[0].frame_id)) for f in frames
    ]
    print(f"[OK] 合并完成: BAG={len(bag_paths)}, ToF raw={len(frames)}")
    return _browse_frames(hists, labels, bag_dists)


def _bag_depth_entry_to_m(entry) -> np.ndarray | None:
    if entry is None:
        return None
    dist_mm, conf = entry
    dist_m = dist_mm.astype(np.float32) / 1000.0
    dist_m[conf == 0] = 0.0
    return dist_m


def _load_raw_dir_frames(dir_path: Path) -> list[tuple[str, np.ndarray]]:
    """加载目录下所有 .raw 文件，返回 [(文件名, hist(30,40,64)), ...]。"""
    files = sorted(p for p in dir_path.iterdir() if p.suffix.lower() == ".raw")
    out: list[tuple[str, np.ndarray]] = []
    for p in files:
        try:
            raw_u16 = np.frombuffer(p.read_bytes(), dtype=np.uint16)
        except Exception as exc:
            print(f"[WARN] 读取失败 {p.name}: {exc}")
            continue
        if raw_u16.size < TOF_H * TOF_W * TOF_C:
            print(f"[WARN] 跳过 {p.name}: 数据不足 ({raw_u16.size} u16)")
            continue
        out.append((p.name, tof_histograms_from_u16(raw_u16)))
    return out


def _run_dir_mode(dir_path_str: str) -> int:
    """浏览目录内的 .raw，或合并设备日志 dtof_depth_bag 中的录包。"""
    dir_path = Path(dir_path_str).resolve()
    if not dir_path.is_dir():
        print(f"[ERR] 目录不存在: {dir_path}")
        return 1

    print(f"[INFO] 扫描目录: {dir_path}")
    frames = _load_raw_dir_frames(dir_path)
    if frames:
        print(f"[OK] {dir_path.name}: 有效 raw 帧={len(frames)}")
        return _browse_frames([f[1] for f in frames], [f[0] for f in frames], None)

    bag_paths = _discover_dtof_bags(dir_path)
    if bag_paths:
        print(f"[INFO] 找到 dtof_depth_bag 录包 {len(bag_paths)} 个")
        return _run_bag_paths_mode(bag_paths)

    print("[ERR] 目录内没有可用的 .raw，也没有 dtof_depth_bag 录包")
    return 1


def _browse_frames(
    hists_list: list[np.ndarray],
    labels: list[str],
    bag_dists: list[np.ndarray | None] | None,
) -> int:
    """批量推理后交互式浏览一组帧（BAG 与目录模式共用）。"""
    import cv2
    import torch

    cv2.setUseOptimized(True)
    rotate_90 = bool(ROTATE_90)
    mirror = bool(MIRROR)
    show_w, show_h = _get_show_size(rotate_90)

    nn_dir = Path(__file__).resolve().parent
    if str(nn_dir) in sys.path:
        sys.path.remove(str(nn_dir))
    sys.path.insert(0, str(nn_dir))
    from net import Network

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    n = len(hists_list)
    print(f"[INFO] 批量推理 {n} 帧 (batch={_BAG_INFER_BATCH}, device={device})...")
    all_dist = np.zeros((n, TOF_H, TOF_W), dtype=np.float32)
    all_snr = np.zeros_like(all_dist)
    all_conf = np.zeros_like(all_dist)
    all_peak = np.zeros_like(all_dist)
    all_refl = np.zeros_like(all_dist)

    for start in range(0, n, _BAG_INFER_BATCH):
        end = min(start + _BAG_INFER_BATCH, n)
        batch_h = np.stack(
            [hists_list[i] for i in range(start, end)], axis=0,
        ).astype(np.float32, copy=False)
        d, s, c, p, r = _run_infer_batch(net, device, batch_h)
        all_dist[start:end] = d
        all_snr[start:end] = s
        all_conf[start:end] = c
        all_peak[start:end] = p
        all_refl[start:end] = r
        print(f"  [{end}/{n}]")
    print("[OK] 推理完成")

    win = "NN_REALTIME"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    mouse: dict = {"x": 0, "y": 0}
    bar_rect: dict = {}          # 进度条画布坐标，由 _compose_dashboard 回填
    cbar_rect: dict = {}         # 距离色条 ramp 画布坐标
    field_rects: dict = {}       # 显示范围输入框命中区
    disp_ctrl = _default_disp_ctrl()
    seek: dict = {"idx": None}   # 拖动进度条产生的目标帧

    def _seek_from_x(mx: int) -> None:
        x0 = bar_rect.get("x0")
        x1 = bar_rect.get("x1")
        if x0 is None or x1 is None or x1 <= x0:
            return
        frac = (float(mx) - float(x0)) / float(x1 - x0)
        seek["idx"] = int(round(float(np.clip(frac, 0.0, 1.0)) * (n - 1)))

    def _in_bar(x: int, y: int) -> bool:
        if bar_rect.get("y0") is None:
            return False
        return (
            bar_rect["x0"] - 8 <= x <= bar_rect["x1"] + 8
            and bar_rect["y0"] - 8 <= y <= bar_rect["y1"] + 8
        )

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        mouse["x"], mouse["y"] = int(x), int(y)
        ev = int(event)
        if ev == int(cv2.EVENT_LBUTTONDOWN):
            key = _hit_field_key(x, y, field_rects)
            if key is not None:
                _begin_field_edit(disp_ctrl, key)
            else:
                if disp_ctrl.get("focus") is not None:
                    _commit_field_edit(disp_ctrl)
                if _in_bar(x, y):
                    disp_ctrl["drag"] = "seek"
                    _seek_from_x(x)
        elif ev == int(cv2.EVENT_MOUSEMOVE) and (int(flags) & int(cv2.EVENT_FLAG_LBUTTON)):
            if disp_ctrl.get("drag") == "seek":
                _seek_from_x(x)
        elif ev == int(cv2.EVENT_LBUTTONUP):
            disp_ctrl["drag"] = None

    cv2.setMouseCallback(win, on_mouse)

    playing = False
    play_interval_ms = int(round(1000.0 / _BAG_PLAY_HZ))
    next_play_ms = 0
    idx = 0
    print(f"[INFO] 共 {n} 帧, A/D ← → 切帧, 空格 播放/暂停, 右下角输入显示范围, ESC 退出")

    while True:
        if seek["idx"] is not None:
            idx = max(0, min(int(seek["idx"]), n - 1))
            seek["idx"] = None
            playing = False

        now_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if playing and now_ms >= next_play_ms:
            if idx < n - 1:
                idx += 1
            else:
                playing = False
            next_play_ms = now_ms + play_interval_ms

        hist = hists_list[idx]
        _sync_disp_ctrl(disp_ctrl)
        depth_near_m = float(disp_ctrl["near_m"])
        depth_far_m = float(disp_ctrl["far_m"])
        peak_lo = float(disp_ctrl["peak_lo"])
        peak_hi = float(disp_ctrl["peak_hi"])

        bag_dist_m = bag_dists[idx] if bag_dists is not None else None

        view, hist_img, strip_img, px, py = _render_view(
            all_dist[idx], all_conf[idx], all_peak[idx], all_refl[idx],
            hist.astype(np.float32),
            mouse["x"] - MAIN_OFFSET_X, mouse["y"] - MAIN_OFFSET_Y,
            show_w, show_h, rotate_90, mirror,
            bag_dist_m=bag_dist_m,
            depth_near_m=depth_near_m,
            depth_far_m=depth_far_m,
            peak_lo=peak_lo,
            peak_hi=peak_hi,
        )

        hover_lines = [
            f"dist {float(all_dist[idx][py, px]):.3f}m",
        ]
        if bag_dist_m is not None:
            hover_lines.append(f"bag_dist {float(bag_dist_m[py, px]):.3f}m")
        hover_lines.extend([
            f"snr {float(all_snr[idx][py, px]):.3f}",
            f"peak {float(all_peak[idx][py, px]):.3f}",
            f"reflectance {float(all_refl[idx][py, px]) * 100.0:.3f}%",
        ])

        label = labels[idx] if idx < len(labels) else ""
        status = f"{label}   {idx}/{n - 1}   {'PLAY' if playing else 'PAUSE'}"
        progress_frac = (idx / (n - 1)) if n > 1 else 0.0
        canvas = _compose_dashboard(
            view, hist_img, strip_img, status,
            hover_lines=hover_lines,
            progress_frac=progress_frac,
            progress_label=f"FRAME  {idx}/{n - 1}",
            bar_rect_out=bar_rect,
            depth_near_m=depth_near_m,
            depth_far_m=depth_far_m,
            peak_lo=peak_lo,
            peak_hi=peak_hi,
            cbar_rect_out=cbar_rect,
            field_rects_out=field_rects,
            focus_key=disp_ctrl.get("focus"),
            edit_text=str(disp_ctrl.get("edit", "")),
            mouse_xy=(int(mouse["x"]), int(mouse["y"])),
        )
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(20)
        if key < 0:
            continue
        if _handle_disp_key(disp_ctrl, key):
            continue
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            playing = not playing
            next_play_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if key in (2424832, ord("a"), ord("A")) and idx > 0:
            idx -= 1
        if key in (2555904, ord("d"), ord("D")) and idx < n - 1:
            idx += 1

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="realtime.py — ADB 实时 / BAG 回放 / raw 或设备日志目录浏览")
    parser.add_argument("source", nargs="?", default=None,
                        help="BAG/MCAP 文件、.raw 目录或含 dtof_depth_bag 的设备日志目录；不指定则 ADB 实时模式")
    args = parser.parse_args()

    if args.source:
        if Path(args.source).is_dir():
            return _run_dir_mode(args.source)
        return _run_bag_mode(args.source)

    # ---- ADB 实时模式 ----
    rotate_90 = bool(ROTATE_90)
    mirror = bool(MIRROR)
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
    if str(nn_dir) in sys.path:
        sys.path.remove(str(nn_dir))
    sys.path.insert(0, str(nn_dir))

    from net import Network  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network().to(device)
    net.eval()

    cv2.namedWindow("NN_REALTIME", cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}
    field_rects: dict = {}
    disp_ctrl = _default_disp_ctrl()

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        mouse["x"] = int(x)
        mouse["y"] = int(y)
        ev = int(event)
        if ev == int(cv2.EVENT_LBUTTONDOWN):
            key = _hit_field_key(x, y, field_rects)
            if key is not None:
                _begin_field_edit(disp_ctrl, key)
            elif disp_ctrl.get("focus") is not None:
                _commit_field_edit(disp_ctrl)

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
    last_color_range: tuple = (None, None, None, None)
    frame_cache: np.ndarray | None = None

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
            _sync_disp_ctrl(disp_ctrl)
            depth_near_m = float(disp_ctrl["near_m"])
            depth_far_m = float(disp_ctrl["far_m"])
            peak_lo = float(disp_ctrl["peak_lo"])
            peak_hi = float(disp_ctrl["peak_hi"])
            color_range = (depth_near_m, depth_far_m, peak_lo, peak_hi)
            need_redraw = (
                got_new_frame
                or (mouse_xy != last_mouse_xy)
                or (frame_cache is None)
                or (rec_on != last_rec_on)
                or (color_range != last_color_range)
                or (disp_ctrl.get("focus") is not None)
            )

            if need_redraw:
                view, hist_new, strip_new, px, py = _render_view(
                    cached_pred_depth, cached_conf, cached_peak, cached_reflectance,
                    cached_in,
                    mouse_xy[0] - MAIN_OFFSET_X, mouse_xy[1] - MAIN_OFFSET_Y,
                    show_w, show_h, rotate_90, mirror,
                    depth_near_m=depth_near_m,
                    depth_far_m=depth_far_m,
                    peak_lo=peak_lo,
                    peak_hi=peak_hi,
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

                hover_lines = [
                    f"dist {float(cached_pred_depth[py, px]):.3f}m",
                    f"snr {float(cached_snr[py, px]):.3f}",
                    f"peak {float(cached_peak[py, px]):.3f}",
                    f"reflectance {float(cached_reflectance[py, px]) * 100.0:.3f}%",
                ]
                if rec_writer is not None:
                    cv2.circle(view, (show_w * 2 - 24, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.putText(view, "REC", (show_w * 2 - 72, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2, cv2.LINE_AA)
                if rec_err:
                    cv2.putText(
                        view,
                        f"rec err: {rec_err}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                status = (
                    f"io {io_fps:.1f} | infer {infer_fps:.1f} | ui {ui_fps:.1f} fps"
                    + ("   * REC" if rec_on else "")
                )
                frame_cache = _compose_dashboard(
                    view, hist_new, strip_new, status,
                    hover_lines=hover_lines,
                    depth_near_m=depth_near_m,
                    depth_far_m=depth_far_m,
                    peak_lo=peak_lo,
                    peak_hi=peak_hi,
                    field_rects_out=field_rects,
                    focus_key=disp_ctrl.get("focus"),
                    edit_text=str(disp_ctrl.get("edit", "")),
                    mouse_xy=mouse_xy,
                )
                last_mouse_xy = mouse_xy
                last_rec_on = rec_on
                last_color_range = color_range

            ui_cnt += 1
            cv2.imshow("NN_REALTIME", frame_cache)
            if rec_writer is not None and frame_cache is not None:
                rec_writer.write(frame_cache)

            k = int(cv2.waitKey(1) & 0xFF)
            if k != 255 and _handle_disp_key(disp_ctrl, k):
                continue
            if k == 32:  # Space: toggle recording
                if rec_writer is None:
                    rec_err = ""
                    try:
                        RECORD_DIR.mkdir(parents=True, exist_ok=True)
                        rec_path = str(RECORD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(rec_path, fourcc, max(float(REC_FPS), 1.0), (frame_cache.shape[1], frame_cache.shape[0]))
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

