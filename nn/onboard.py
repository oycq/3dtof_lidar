#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nn/onboard.py

On-board ToF quick preview:
- read /tmp/tof.output
- expected shape: 5x30x40 float32
- channels (C++ write order): dist/conf/peak/reflectance/snr
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

TOF_H = 30
TOF_W = 40
PIXELS = TOF_H * TOF_W
OUT_C = 5

SHOW_W = 520
SHOW_H = 390
HEADER_H = 56
REC_FPS = 20.0

REMOTE_TM_DIR = "/tmp"
REMOTE_OUTPUT_PATH = "/tmp/tof.output"

LOCAL_CACHE_DIR = Path("./tmp")
LOCAL_OUTPUT_PATH = LOCAL_CACHE_DIR / "tof.output"
RECORD_DIR = Path("./tmp")

EXPECTED_BYTES = OUT_C * PIXELS * 4
READ_RETRY = 3
ADB_READ_TIMEOUT_S = 0.9
ADB_RETRY_SLEEP_S = 0.01
FPS_STAT_INTERVAL_S = 0.5
EPS = 1e-6
DISP_GAMMA = 1.2
SNR_SHOW_MAX = 10.0


def _bytes_to_output_maps(raw: bytes, remote_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(raw) < int(EXPECTED_BYTES):
        raise ValueError(f"short bytes in {remote_path}: got {len(raw)}, expect >= {EXPECTED_BYTES}")
    data = np.frombuffer(raw[:EXPECTED_BYTES], dtype=np.dtype("<f4"))
    if data.size != OUT_C * PIXELS:
        raise ValueError(f"bad float count in {remote_path}: got {data.size}, expect {OUT_C * PIXELS}")
    maps = data.reshape(OUT_C, TOF_H, TOF_W).copy()
    dist = maps[0]
    conf = maps[1]
    _peak = maps[2]  # noqa: F841  # read but currently unused by UI
    reflect = maps[3]
    snr = maps[4]
    return dist, conf, reflect, snr


def _adb_read_file_bytes(remote_path: str, local_path: Path, *, expected_bytes: int, retry: int) -> bytes:
    expected_bytes = int(expected_bytes)
    retry = int(max(retry, 0))

    for k in range(retry + 1):
        try:
            if local_path.exists():
                local_path.unlink(missing_ok=True)
            p = subprocess.run(
                ["adb", "pull", remote_path, str(local_path)],
                timeout=float(ADB_READ_TIMEOUT_S),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if int(p.returncode) != 0:
                if k < retry:
                    time.sleep(float(ADB_RETRY_SLEEP_S))
                continue
            if (not local_path.exists()) or int(local_path.stat().st_size) < expected_bytes:
                if k < retry:
                    time.sleep(float(ADB_RETRY_SLEEP_S))
                continue
            data = local_path.read_bytes()
            if len(data) >= expected_bytes:
                return bytes(data[:expected_bytes])
        except Exception:
            pass
        if k < retry:
            time.sleep(float(ADB_RETRY_SLEEP_S))

    raise RuntimeError(f"adb read failed: {remote_path}")


def _pull_output_via_adb() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_raw = _adb_read_file_bytes(
        REMOTE_OUTPUT_PATH,
        LOCAL_OUTPUT_PATH,
        expected_bytes=EXPECTED_BYTES,
        retry=READ_RETRY,
    )
    return _bytes_to_output_maps(output_raw, REMOTE_OUTPUT_PATH)


def _colorize_depth_realtime(depth_m: np.ndarray, conf: np.ndarray) -> np.ndarray:
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0) & np.isfinite(c) & (c > 0.5)
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    y = 1.8 / np.maximum(d[valid], EPS)
    u8[valid] = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_gray01(x: np.ndarray) -> np.ndarray:
    u8 = np.clip(np.rint(np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=2)


def _colorize_snr(snr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    s = np.asarray(snr, dtype=np.float32)
    m = np.asarray(valid, dtype=bool)
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(s[m] / float(SNR_SHOW_MAX), 0.0, 1.0), 1.0 / float(DISP_GAMMA))
    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _with_text(img_bgr: np.ndarray, text: str, y: int = 24) -> np.ndarray:
    import cv2  # type: ignore

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
    import cv2  # type: ignore

    xx = int(np.clip(x, 0, img_bgr.shape[1] - 1))
    yy = int(np.clip(y, 0, img_bgr.shape[0] - 1))
    cv2.circle(img_bgr, (xx, yy), 3, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img_bgr, (xx, yy), 2, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e

    print(f"[tm]   {REMOTE_TM_DIR}")
    print(f"[read] {REMOTE_OUTPUT_PATH}")
    print(f"[shape] {OUT_C}x{TOF_H}x{TOF_W}, dtype=float32")
    print("[chan] dist/conf/peak/reflectance/snr (C++ write order)")
    print(f"[adb]  pull: {REMOTE_OUTPUT_PATH}")
    print("[mask] dist: conf<=0.5 -> black")
    print(f"[rec]  save mp4 to {RECORD_DIR}")

    cv2.namedWindow("ONBOARD_REALTIME", cv2.WINDOW_AUTOSIZE)
    mouse = {"x": 0, "y": 0}

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if int(event) != int(cv2.EVENT_MOUSEMOVE):
            return
        mouse["x"] = int(x)
        mouse["y"] = int(y)

    cv2.setMouseCallback("ONBOARD_REALTIME", on_mouse)

    wait_ms = 1

    dist = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    conf = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    reflect = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    snr = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    frame_idx = -1
    last_err = ""
    rec_writer: object | None = None
    rec_path = ""
    rec_err = ""
    read_fps = 0.0
    ui_fps = 0.0
    read_cnt = 0
    ui_cnt = 0
    fps_tick = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            try:
                dist, conf, reflect, snr = _pull_output_via_adb()
                frame_idx += 1
                last_err = ""
                read_cnt += 1
            except Exception as e:
                last_err = str(e)

            ui_cnt += 1
            dt_fps = now - fps_tick
            if dt_fps >= float(FPS_STAT_INTERVAL_S):
                inv_dt = 1.0 / max(dt_fps, 1e-6)
                read_fps = float(read_cnt) * inv_dt
                ui_fps = float(ui_cnt) * inv_dt
                read_cnt = 0
                ui_cnt = 0
                fps_tick = now

            dist_big = cv2.resize(_colorize_depth_realtime(dist, conf), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            conf_big = cv2.resize(_colorize_gray01(conf), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            reflect_big = cv2.resize(_colorize_gray01(reflect), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
            snr_big = cv2.resize(_colorize_snr(snr, conf > 0.5), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)

            mx = int(np.clip(mouse.get("x", 0), 0, SHOW_W * 2 - 1))
            my = int(np.clip(int(mouse.get("y", 0)) - int(HEADER_H), 0, SHOW_H * 2 - 1))
            tile_x0 = 0 if mx < SHOW_W else SHOW_W
            tile_y0 = 0 if my < SHOW_H else SHOW_H
            px, py = _disp_xy_to_pixel(mx - tile_x0, my - tile_y0, SHOW_W, SHOW_H)
            dx, dy = _pixel_to_disp_xy(px, py, SHOW_W, SHOW_H)

            for img in [dist_big, conf_big, reflect_big, snr_big]:
                _draw_marker(img, dx, dy)

            _with_text(dist_big, "DIST (conf>0.5)")
            _with_text(conf_big, "CONF")
            _with_text(reflect_big, "REFLECT")
            _with_text(snr_big, "SNR")

            hover1 = (
                f"idx={frame_idx}  read_fps={read_fps:.1f}  ui_fps={ui_fps:.1f}  "
                f"hover=({px},{py})  dist={float(dist[py, px]):.4f}m  conf={float(conf[py, px]):.4f}"
            )
            hover2 = f"reflect={float(reflect[py, px]):.4f}  snr={float(snr[py, px]):.4f}"

            view = np.vstack(
                [
                    np.zeros((HEADER_H, SHOW_W * 2, 3), dtype=np.uint8),
                    np.hstack([dist_big, snr_big]),
                    np.hstack([reflect_big, conf_big]),
                ]
            )
            cv2.putText(view, hover1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(view, hover2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            if rec_writer is not None:
                cv2.circle(view, (SHOW_W * 2 - 24, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(view, "REC", (SHOW_W * 2 - 72, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2, cv2.LINE_AA)

            if frame_idx < 0:
                cv2.putText(view, "NO DATA", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
                if last_err:
                    cv2.putText(view, last_err, (10, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            elif last_err:
                cv2.putText(
                    view,
                    f"last pull failed: {last_err}",
                    (10, HEADER_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 160, 255),
                    1,
                    cv2.LINE_AA,
                )

            if rec_writer is not None:
                rec_writer.write(view)

            cv2.imshow("ONBOARD_REALTIME", view)
            key = int(cv2.waitKey(wait_ms) & 0xFF)
            if key == 32:  # Space: toggle recording
                if rec_writer is None:
                    rec_err = ""
                    try:
                        RECORD_DIR.mkdir(parents=True, exist_ok=True)
                        rec_path = str(RECORD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(rec_path, fourcc, max(float(REC_FPS), 1.0), (view.shape[1], view.shape[0]))
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
            if key == 27:
                break

    finally:
        if rec_writer is not None:
            rec_writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
