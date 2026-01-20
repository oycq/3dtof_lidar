#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof_server.py

同进程 ToF 实时采集服务（与 server.py 的 LivoxRealtimeServer 风格一致）：
- 后台线程持续触发设备侧生成 /tmp/tof.raw，并拉取到本机内存（bytes）
- 把最新帧转换为“反射率图”（强度 = histogram sum），映射为 uint8 灰度（30x40）
- 将“反射率图 + 原始 tof.raw bytes”放入一个容量受限的队列，供 client/server 随时取最新

退出：
- 线程是 daemon=True；进程退出时直接退出即可
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np

from tof3d import ToF3DParams, tof_histograms_from_u16


HERE = Path(__file__).resolve().parent
TMP_DIR = HERE / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ToFFrame:
    ts: float
    reflect_u8: np.ndarray  # (30,40) uint8
    raw_bytes: bytes  # 原始 tof.raw（用于落盘/复现）


class ToFRealtimeServer:
    """
    同进程实时 ToF：start 后后台线程持续采集；get_latest 返回最新一帧（不出队）。
    """

    def __init__(
        self,
        *,
        queue_maxlen: int = 5,
        min_peak_count: float = 100.0,
        target_fps: float = 10.0,
    ) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: Deque[ToFFrame] = deque(maxlen=int(max(queue_maxlen, 1)))

        self._params = ToF3DParams(min_peak_count=float(min_peak_count))
        self._target_dt = 1.0 / float(max(target_fps, 1.0))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ToFRealtimeServer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.8)
        self._thread = None

    def get_latest(self) -> Optional[ToFFrame]:
        with self._lock:
            return self._q[-1] if self._q else None

    # ========= 采集实现 =========

    @staticmethod
    def _adb_trigger_generate_raw() -> bool:
        """
        触发设备侧生成 /tmp/tof.raw（与 get_tof.py / client.py 保持一致的 /tmp/sv 机制）。
        """
        import subprocess

        cmd = "if [ -e /tmp/sv ]; then rm /tmp/sv && rm /tmp/tof.raw; fi && touch /tmp/sv"
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
    def _adb_read_raw_bytes() -> Optional[bytes]:
        """
        读取设备侧 /tmp/tof.raw 内容为 bytes。
        优先用 adb exec-out（不落盘），失败则 fallback 到 adb pull tmp/tof.raw（会落盘）。
        """
        import subprocess

        # 1) 尝试 exec-out（更快、不会写 tmp 文件）
        try:
            r = subprocess.run(
                ["adb", "exec-out", "cat", "/tmp/tof.raw"],
                timeout=1.2,
                check=False,
                capture_output=True,
            )
            if int(r.returncode) == 0 and r.stdout and len(r.stdout) >= 5120:
                return bytes(r.stdout)
        except Exception:
            pass

        # 2) fallback: pull 到本地 tmp/tof.raw，再读入内存
        try:
            local = TMP_DIR / "tof.raw"
            p = subprocess.run(
                ["adb", "pull", "/tmp/tof.raw", str(local)],
                timeout=1.2,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if int(p.returncode) != 0 or (not local.exists()) or local.stat().st_size < 5120:
                return None
            return local.read_bytes()
        except Exception:
            return None

    @staticmethod
    def _tof_intensity_to_u8(intensity_sum: np.ndarray, *, gamma: float = 2.2, target_mean: float = 0.18) -> np.ndarray:
        """
        复刻 cali/check.py 的 ToF 强度显示映射：
        - 用整图 mean 归一化到 target_mean
        - 再做 gamma（1/gamma）
        """
        if intensity_sum.size == 0:
            return np.zeros((30, 40), dtype=np.uint8)
        v = np.asarray(intensity_sum, dtype=np.float32)
        mean = float(np.mean(v)) if v.size else 0.0
        if mean <= 0.0:
            return np.zeros(v.shape, dtype=np.uint8)
        k = mean / float(target_mean)
        k = max(k, 1e-6)
        n = v / k
        n = np.clip(n, 0.0, 1.0)
        if float(gamma) > 0:
            n = np.power(n, 1.0 / float(gamma))
        return np.clip(np.rint(n * 255.0), 0.0, 255.0).astype(np.uint8)

    def _run(self) -> None:
        fail_sleep = 0.15
        while not self._stop.is_set():
            t0 = time.time()

            ok = self._adb_trigger_generate_raw()
            if not ok:
                time.sleep(fail_sleep)
                continue

            # 给设备一点点写文件时间（过短容易拿到空/半包）
            time.sleep(0.05)
            raw_bytes = self._adb_read_raw_bytes()
            if not raw_bytes:
                time.sleep(fail_sleep)
                continue

            # 解析为 uint16（小端），再算反射率
            raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            hist = tof_histograms_from_u16(raw_u16, params=self._params).astype(np.float32, copy=False)  # (30,40,64)
            if hist.size == 0:
                time.sleep(fail_sleep)
                continue

            inten = hist.sum(axis=2).astype(np.float32, copy=False)  # (30,40)
            reflect_u8 = self._tof_intensity_to_u8(inten)

            frame = ToFFrame(ts=time.time(), reflect_u8=reflect_u8, raw_bytes=raw_bytes)
            with self._lock:
                self._q.append(frame)

            dt = time.time() - t0
            # 简单限帧，避免 adb 过载
            sleep = self._target_dt - dt
            if sleep > 0:
                time.sleep(min(sleep, 0.2))


def main() -> int:
    # 允许单独运行：只采集不显示（用于验证 adb/设备是否正常）
    srv = ToFRealtimeServer()
    srv.start()
    try:
        while True:
            time.sleep(1.0)
            f = srv.get_latest()
            if f is not None:
                print(f"[ToF] ts={f.ts:.2f} bytes={len(f.raw_bytes)} reflect={tuple(f.reflect_u8.shape)}")
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


