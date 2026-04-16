#!/usr/bin/env python3
"""
MCAP/BAG 可视化工具（VpDtofDepth）

功能：
1. 通过 BAG_NAME 指定当前目录下的单个 bag/mcap 文件
2. 读取指定 topic（默认 alg/dtof_depth）的 VpDtofDepth payload
3. 使用 cv2.imshow 实时显示：
   - dist: (2000 / dist_mm).clip(0, 1) * 255，再使用 JET 伪彩
   - peak: (peak / peak.mean() * 65).clip(0, 255)，uint8 灰度

依赖：
    pip install mcap numpy opencv-python
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from mcap.exceptions import EndOfFile
from mcap.reader import NonSeekingReader, make_reader

TOF_H = 30
TOF_W = 40
PIXELS = TOF_H * TOF_W
HEADER_SIZE = 16  # uint64 + uint32 + uint8 + reserved[3]
DIST_SIZE = PIXELS * 2
CONF_SIZE = PIXELS
PEAK_SIZE = PIXELS * 2
PAYLOAD_SIZE = HEADER_SIZE + DIST_SIZE + CONF_SIZE + PEAK_SIZE
# 直接在这里指定当前目录下要处理的 bag/mcap 文件名（例如 "229.bag"）。
BAG_NAME = "7.bag"
# 直接在这里指定 topic。
TOPIC = "alg/dtof_depth"
# 显示尺寸（单列）。
VIEW_WIDTH = 400
VIEW_HEIGHT = 300
PLAY_HZ = 10.0


def parse_vp_dtof_depth(
    payload: bytes,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(payload) < PAYLOAD_SIZE:
        return None

    data = memoryview(payload)
    timestamp_us = int.from_bytes(data[0:8], byteorder="little", signed=False)
    frame_id = int.from_bytes(data[8:12], byteorder="little", signed=False)
    is_valid = int(data[12])
    dist_offset = HEADER_SIZE
    conf_offset = dist_offset + DIST_SIZE
    peak_offset = conf_offset + CONF_SIZE

    dist = np.frombuffer(
        data[dist_offset : dist_offset + DIST_SIZE], dtype="<u2"
    ).reshape(TOF_H, TOF_W).copy()
    conf = np.frombuffer(
        data[conf_offset : conf_offset + CONF_SIZE], dtype=np.uint8
    ).reshape(TOF_H, TOF_W).copy()
    peak = np.frombuffer(
        data[peak_offset : peak_offset + PEAK_SIZE], dtype="<u2"
    ).reshape(TOF_H, TOF_W).copy()
    return timestamp_us, frame_id, is_valid, dist, conf, peak


def build_views(
    dist: np.ndarray, conf: np.ndarray, peak: np.ndarray, resize_wh: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    dist_float = dist.astype(np.float32)
    dist_safe = np.where(dist_float <= 0.0, 1.0, dist_float)
    dist_u8 = (2000.0 / dist_safe).clip(0, 1) * 255.0
    dist_u8 = dist_u8.astype(np.uint8)
    dist_color = cv2.applyColorMap(dist_u8, cv2.COLORMAP_JET)
    # 置信度为 0 的点在距离图上强制显示为黑色
    dist_color[conf == 0] = (0, 0, 0)

    peak_float = peak.astype(np.float32)
    peak_mean = float(peak_float.mean())
    if peak_mean <= 1e-6:
        peak_u8 = np.zeros_like(peak, dtype=np.uint8)
    else:
        peak_u8 = (peak_float / peak_mean * 65.0).clip(0, 255).astype(np.uint8)

    # 30x40 -> 300x400，最近邻避免插值造成假纹理
    dist_color_show = cv2.resize(
        dist_color, resize_wh, interpolation=cv2.INTER_NEAREST
    )
    peak_show = cv2.resize(peak_u8, resize_wh, interpolation=cv2.INTER_NEAREST)
    peak_gray_show = cv2.cvtColor(peak_show, cv2.COLOR_GRAY2BGR)
    return dist_color_show, peak_gray_show


@dataclass
class FrameData:
    src_file: str
    seq_in_file: int
    timestamp_us: int
    frame_id: int
    is_valid: int
    dist: np.ndarray  # uint16, 30x40, mm
    conf: np.ndarray  # uint8, 30x40
    peak: np.ndarray  # uint16, 30x40


class ViewerState:
    def __init__(self, frames: list[FrameData], resize_wh: tuple[int, int]) -> None:
        self.frames = frames
        self.resize_w, self.resize_h = resize_wh
        self.banner_h = 52
        self.idx = 0
        self.hover_xy: tuple[int, int] | None = None  # (row, col)
        self.needs_redraw = True
        self.playing = False
        self.play_interval_ms = int(round(1000.0 / PLAY_HZ))
        self.next_play_ms = 0
        self.play_button_rect = (0, 0, 0, 0)  # x1, y1, x2, y2
        self.hover_panel: int | None = None  # 0=dist, 1=peak


def load_frames(files: list[Path], topic_filter: str) -> list[FrameData]:
    frames: list[FrameData] = []
    for bag_path in files:
        print(f"[INFO] 扫描: {bag_path}")
        total_topic = 0
        skipped = 0

        def consume_messages(message_iter) -> None:
            nonlocal total_topic, skipped
            for _, channel, message in message_iter:
                topic = channel.topic or "unknown_topic"
                if topic != topic_filter:
                    continue
                total_topic += 1
                parsed = parse_vp_dtof_depth(message.data)
                if parsed is None:
                    skipped += 1
                    continue
                timestamp_us, frame_id, is_valid, dist, conf, peak = parsed
                frames.append(
                    FrameData(
                        src_file=bag_path.name,
                        seq_in_file=total_topic - 1,
                        timestamp_us=timestamp_us,
                        frame_id=frame_id,
                        is_valid=is_valid,
                        dist=dist,
                        conf=conf,
                        peak=peak,
                    )
                )

        with bag_path.open("rb") as f:
            try:
                consume_messages(make_reader(f).iter_messages())
            except Exception as exc:
                print(
                    f"[WARN] {bag_path.name}: make_reader 失败 ({exc})，"
                    "改为顺序读取"
                )
                f.seek(0)
                try:
                    consume_messages(
                        NonSeekingReader(f).iter_messages(log_time_order=False)
                    )
                except Exception as exc2:
                    print(
                        f"[WARN] {bag_path.name}: 顺序读取结束 ({exc2})，"
                        "已保留前面可读帧"
                    )
        print(
            f"[OK] {bag_path.name}: topic帧={total_topic}, 有效帧={total_topic - skipped}, "
            f"无效帧={skipped}"
        )
    return frames


def draw_frame(state: ViewerState) -> np.ndarray:
    frame = state.frames[state.idx]
    dist_view, peak_view = build_views(
        frame.dist, frame.conf, frame.peak, (state.resize_w, state.resize_h)
    )
    img_row = np.hstack([dist_view, peak_view])
    canvas_w = img_row.shape[1]
    banner = np.zeros((state.banner_h, canvas_w, 3), dtype=np.uint8)
    canvas = np.vstack([banner, img_row])

    if state.hover_xy is not None and state.hover_panel is not None:
        r, c = state.hover_xy
        panel_offset_x = 0 if state.hover_panel == 0 else state.resize_w
        cell_w = state.resize_w / TOF_W
        cell_h = state.resize_h / TOF_H
        x1 = panel_offset_x + int(c * cell_w)
        y1 = state.banner_h + int(r * cell_h)
        x2 = panel_offset_x + int((c + 1) * cell_w) - 1
        y2 = state.banner_h + int((r + 1) * cell_h) - 1
        cx = panel_offset_x + int((c + 0.5) * cell_w)
        cy = state.banner_h + int((r + 0.5) * cell_h)
        radius = max(3, int(min(cell_w, cell_h) * 0.35))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 0, 0), 1)
        cv2.circle(canvas, (cx, cy), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), radius, (0, 255, 255), 1, cv2.LINE_AA)

    # 顶部黑底白字信息条（banner 区域已预留，这里只需绘制文字等内容）
    left_text = (
        f"frame {state.idx + 1}/{len(state.frames)} | "
        f"src={frame.src_file}"
    )
    cv2.putText(
        canvas,
        left_text,
        (8, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"ts={frame.timestamp_us} us | frame_id={frame.frame_id}",
        (8, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    button_w = 128
    button_h = 30
    button_x1 = canvas.shape[1] - button_w - 8
    button_y1 = 10
    button_x2 = button_x1 + button_w
    button_y2 = button_y1 + button_h
    state.play_button_rect = (button_x1, button_y1, button_x2, button_y2)
    if state.playing:
        button_text = "Pause"
        button_color = (60, 80, 240)
    else:
        button_text = "Play"
        button_color = (70, 170, 70)
    cv2.rectangle(canvas, (button_x1, button_y1), (button_x2, button_y2), button_color, -1)
    cv2.rectangle(canvas, (button_x1, button_y1), (button_x2, button_y2), (255, 255, 255), 1)
    cv2.putText(
        canvas,
        button_text,
        (button_x1 + 20, button_y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if state.hover_xy is not None:
        r, c = state.hover_xy
        dist_m = float(frame.dist[r, c]) / 1000.0
        peak_v = int(frame.peak[r, c])
        info = f"dist={dist_m:.3f} m, peak={peak_v}"
        cv2.putText(
            canvas,
            info,
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def on_trackbar(pos: int, state: ViewerState) -> None:
    state.idx = max(0, min(pos, len(state.frames) - 1))
    state.needs_redraw = True


def on_mouse(event: int, x: int, y: int, _flags: int, state: ViewerState) -> None:
    if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = state.play_button_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            state.playing = not state.playing
            state.next_play_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
            state.needs_redraw = True
            return

    y_img = y - state.banner_h
    if y_img < 0 or y_img >= state.resize_h:
        if state.hover_xy is not None or state.hover_panel is not None:
            state.hover_xy = None
            state.hover_panel = None
            state.needs_redraw = True
        return

    # 两列显示：x<width 为 dist 列，x>=width 为 peak 列
    panel = 0 if x < state.resize_w else 1
    x_local = x if panel == 0 else x - state.resize_w
    if x_local < 0 or x_local >= state.resize_w:
        if state.hover_xy is not None or state.hover_panel is not None:
            state.hover_xy = None
            state.hover_panel = None
            state.needs_redraw = True
        return

    col = int(x_local * TOF_W / state.resize_w)
    row = int(y_img * TOF_H / state.resize_h)
    row = max(0, min(row, TOF_H - 1))
    col = max(0, min(col, TOF_W - 1))
    hover = (row, col)
    if state.hover_xy != hover or state.hover_panel != panel:
        state.hover_xy = hover
        state.hover_panel = panel
        state.needs_redraw = True


def run_interactive_viewer(frames: list[FrameData], resize_wh: tuple[int, int]) -> None:
    if not frames:
        print("[WARN] 没有可显示帧")
        return

    state = ViewerState(frames=frames, resize_wh=resize_wh)
    window_name = "VpDtofDepth Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, len(frames) - 1, lambda p: on_trackbar(p, state))
    cv2.setMouseCallback(window_name, lambda e, x, y, f, u: on_mouse(e, x, y, f, state))

    while True:
        now_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
        if state.playing and now_ms >= state.next_play_ms:
            if state.idx < len(frames) - 1:
                state.idx += 1
            else:
                state.playing = False
            cv2.setTrackbarPos("frame", window_name, state.idx)
            state.needs_redraw = True
            state.next_play_ms = now_ms + state.play_interval_ms

        if state.needs_redraw:
            canvas = draw_frame(state)
            cv2.imshow(window_name, canvas)
            state.needs_redraw = False

        key = cv2.waitKeyEx(20)
        if key in (27, ord("q"), ord("Q")):
            break
        if key == ord(" "):
            state.playing = not state.playing
            state.next_play_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
            state.needs_redraw = True
        if key in (2424832, ord("a"), ord("A")) and state.idx > 0:  # left
            state.idx -= 1
            cv2.setTrackbarPos("frame", window_name, state.idx)
            state.needs_redraw = True
        if key in (2555904, ord("d"), ord("D")) and state.idx < len(frames) - 1:  # right
            state.idx += 1
            cv2.setTrackbarPos("frame", window_name, state.idx)
            state.needs_redraw = True


def main() -> int:
    if not BAG_NAME:
        print('[WARN] 请先在脚本里设置 BAG_NAME，例如 BAG_NAME = "229.bag"')
        return 1

    input_path = (Path.cwd() / BAG_NAME).resolve()
    if not input_path.exists():
        print(f"[WARN] 指定文件不存在: {input_path}")
        return 1

    files = [input_path]
    topic_filter = TOPIC
    resize_wh = (VIEW_WIDTH, VIEW_HEIGHT)

    try:
        frames = load_frames(files, topic_filter=topic_filter)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERR] 读取数据失败: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] 已加载总帧数: {len(frames)}")
    run_interactive_viewer(frames, resize_wh=resize_wh)

    cv2.destroyAllWindows()
    print("[DONE] 全部处理结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
