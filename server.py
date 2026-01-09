#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
server.py

供 client.py 直接 import 使用的“同进程采集服务”：
- client 启动时 import 本模块并 start()，雷达开始持续转动并实时采集
- client 循环中调用 snapshot_xyz() 获取“最近 1 秒”的点，用 cv2 实时显示

注意：
- OpenPyLivox 的实时数据路径是写 OPL BIN；这里采用“同进程尾读 OPL BIN”解析为点云。
- 为了避免文件末尾出现半包导致丢数据，这里实现了一个带残留缓存的 TailParser。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

import openpylivox as opl


HERE = Path(__file__).resolve().parent
TMP_DIR = HERE / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _read_header_wait(path: Path, timeout_sec: float = 8.0) -> Tuple[int, int]:
    """
    等待 OPL BIN 头写入完成，并返回 (firmwareType, dataType)。
    OPL BIN 头：11 bytes "OPENPYLIVOX" + int16 firmwareType + int16 dataType
    """
    t0 = time.time()
    while True:
        if path.exists() and path.stat().st_size >= 15:
            with open(path, "rb") as f:
                magic = f.read(11).decode("utf-8", errors="ignore")
                if magic == "OPENPYLIVOX":
                    fw = int(np.frombuffer(f.read(2), dtype="<i2")[0])
                    dt = int(np.frombuffer(f.read(2), dtype="<i2")[0])
                    return fw, dt
        if time.time() - t0 > timeout_sec:
            raise TimeoutError(f"等待 BIN 头超时：{path}")
        time.sleep(0.05)


@dataclass
class Chunk:
    t: np.ndarray  # float64
    x: np.ndarray  # float32
    y: np.ndarray  # float32
    z: np.ndarray  # float32

    @property
    def t_last(self) -> float:
        return float(self.t[-1]) if self.t.size else float("-inf")


class SlidingWindow:
    def __init__(self, *, max_seconds: float = 1.0) -> None:
        self.max_seconds = float(max_seconds)
        self._chunks: Deque[Chunk] = deque()
        self._t_latest: Optional[float] = None

    def add(self, *, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        if t.size == 0:
            return
        self._t_latest = float(t[-1])
        self._chunks.append(Chunk(t=t, x=x, y=y, z=z))
        self._prune()

    def _prune(self) -> None:
        if self._t_latest is None:
            return
        thr = self._t_latest - self.max_seconds
        while self._chunks and self._chunks[0].t_last < thr:
            self._chunks.popleft()

    def snapshot(self, *, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._t_latest is None or not self._chunks:
            return (np.empty((0,), dtype=np.float32),) * 3

        thr = self._t_latest - self.max_seconds
        t_all = np.concatenate([c.t for c in self._chunks], axis=0)
        x_all = np.concatenate([c.x for c in self._chunks], axis=0)
        y_all = np.concatenate([c.y for c in self._chunks], axis=0)
        z_all = np.concatenate([c.z for c in self._chunks], axis=0)

        m = t_all >= thr
        x_all, y_all, z_all = x_all[m], y_all[m], z_all[m]

        n = int(x_all.size)
        if max_points is not None and n > int(max_points):
            idx = np.random.choice(n, size=int(max_points), replace=False)
            x_all, y_all, z_all = x_all[idx], y_all[idx], z_all[idx]

        return x_all, y_all, z_all


class _TailParser:
    """
    负责从 OPL BIN 追加读取字节流，并解析为点（带残留 buffer 防止半包丢失）。
    """

    def __init__(self, *, firmware_type: int, data_type: int) -> None:
        self.firmware_type = int(firmware_type)
        self.data_type = int(data_type)
        self.buf = bytearray()

        # 支持常见 Cartesian
        # - firmware=1, dataType=0: 21 bytes/pt (x,y,z int32 + intensity u1 + t f8)
        # - firmware=1, dataType=2: 22 bytes/pt (x,y,z int32 + intensity u1 + tag u1 + t f8)
        # - firmware=1, dataType=4: 36 bytes/2pts (两组 x,y,z,intensity,tag + t f8)
        # - firmware>1, dataType=0: 22 bytes/pt (x,y,z int32 + intensity u1 + t f8 + returnNum u1)
        if self.data_type == 0 and self.firmware_type == 1:
            self.rec_size = 21
        elif self.data_type == 0 and self.firmware_type > 1:
            self.rec_size = 22
        elif self.data_type == 2 and self.firmware_type == 1:
            self.rec_size = 22
        elif self.data_type == 4 and self.firmware_type == 1:
            self.rec_size = 36
        else:
            self.rec_size = 0

    def feed(self, data: bytes) -> None:
        if data:
            self.buf.extend(data)

    def parse(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.rec_size <= 0:
            return (np.empty((0,), dtype=np.float64),) * 4

        n = len(self.buf) // self.rec_size
        if n <= 0:
            return (np.empty((0,), dtype=np.float64),) * 4

        take = n * self.rec_size
        chunk = memoryview(self.buf)[:take]
        # 保留剩余半包
        self.buf = bytearray(memoryview(self.buf)[take:])

        if self.data_type == 0 and self.firmware_type == 1:
            dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("i", "u1"), ("t", "<f8")])
            arr = np.frombuffer(chunk, dtype=dt, count=n)
            t = arr["t"].astype(np.float64, copy=False)
            x = (arr["x"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            y = (arr["y"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            z = (arr["z"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            return t, x, y, z

        if self.data_type == 0 and self.firmware_type > 1:
            dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("i", "u1"), ("t", "<f8"), ("rn", "u1")])
            arr = np.frombuffer(chunk, dtype=dt, count=n)
            t = arr["t"].astype(np.float64, copy=False)
            x = (arr["x"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            y = (arr["y"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            z = (arr["z"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            return t, x, y, z

        if self.data_type == 2:
            dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("i", "u1"), ("tag", "u1"), ("t", "<f8")])
            arr = np.frombuffer(chunk, dtype=dt, count=n)
            t = arr["t"].astype(np.float64, copy=False)
            x = (arr["x"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            y = (arr["y"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            z = (arr["z"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
            return t, x, y, z

        # data_type == 4
        dt = np.dtype(
            [
                ("x1", "<i4"),
                ("y1", "<i4"),
                ("z1", "<i4"),
                ("i1", "u1"),
                ("tag1", "u1"),
                ("x2", "<i4"),
                ("y2", "<i4"),
                ("z2", "<i4"),
                ("i2", "u1"),
                ("tag2", "u1"),
                ("t", "<f8"),
            ]
        )
        arr = np.frombuffer(chunk, dtype=dt, count=n)
        t0 = arr["t"].astype(np.float64, copy=False)
        t = np.repeat(t0, 2)

        x = np.empty((n * 2,), dtype=np.float32)
        y = np.empty((n * 2,), dtype=np.float32)
        z = np.empty((n * 2,), dtype=np.float32)

        x[0::2] = (arr["x1"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        y[0::2] = (arr["y1"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        z[0::2] = (arr["z1"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        x[1::2] = (arr["x2"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        y[1::2] = (arr["y2"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        z[1::2] = (arr["z2"].astype(np.float64, copy=False) / 1000.0).astype(np.float32)
        return t, x, y, z


class LivoxRealtimeServer:
    """
    同进程实时采集服务：start 后雷达持续转动并采集；snapshot_xyz 返回最近 1 秒点云。
    """

    def __init__(self, *, max_seconds: float = 1.0) -> None:
        self._max_seconds = float(max_seconds)
        self._lock = threading.Lock()
        self._win = SlidingWindow(max_seconds=self._max_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._sensor: Optional[opl.openpylivox] = None
        self._bin_path: Optional[Path] = None
        self._bin_file = None
        self._parser: Optional[_TailParser] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        sensor = opl.openpylivox(True)
        if not sensor.auto_connect():
            raise RuntimeError("无法连接 Livox（auto_connect 失败）")

        sensor.lidarSpinUp()
        sensor.dataStart_RT_B()
        sensor.setLidarReturnMode(2)
        sensor.setIMUdataPush(False)
        sensor.setRainFogSuppression(True)

        # 持续写 BIN
        # 注意：openpylivox._saveDataToFile 会把 duration 原样写进 captureStream.duration，
        # 如果传 0.0 会导致立即结束（你看到的 “closed BINARY file ...” 就是这个原因）。
        # 所以这里用“接近 4 年”的最大值（必须 < 126230400）。
        LONG_DURATION_SEC = 126230399.0
        stream_bin = TMP_DIR / "stream.bin"
        try:
            if stream_bin.exists():
                stream_bin.unlink()
        except Exception:
            pass
        sensor.saveDataToFile(str(stream_bin), 0.0, LONG_DURATION_SEC)

        fw, dt = _read_header_wait(stream_bin, timeout_sec=8.0)
        f = open(stream_bin, "rb")
        f.seek(15, 0)

        self._sensor = sensor
        self._bin_path = stream_bin
        self._bin_file = f
        self._parser = _TailParser(firmware_type=fw, data_type=dt)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="LivoxRealtimeServer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._bin_file is not None
        assert self._parser is not None

        while not self._stop.is_set():
            data = self._bin_file.read(1024 * 256)  # 256KB
            if not data:
                time.sleep(0.005)
                continue

            self._parser.feed(data)
            while True:
                t, x, y, z = self._parser.parse()
                if t.size == 0:
                    break
                with self._lock:
                    self._win.add(t=t, x=x, y=y, z=z)

    def snapshot_xyz(self, *, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            return self._win.snapshot(max_points=max_points)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

        try:
            if self._bin_file is not None:
                self._bin_file.close()
        except Exception:
            pass
        self._bin_file = None

        try:
            if self._sensor is not None:
                self._sensor.dataStop()
                self._sensor.lidarSpinDown()
                self._sensor.disconnect()
        except Exception:
            pass
        self._sensor = None

        self._parser = None


def main() -> int:
    """
    兼容：也允许单独运行 server.py（会一直采集，直到 Ctrl+C）。
    """
    srv = LivoxRealtimeServer()
    srv.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


