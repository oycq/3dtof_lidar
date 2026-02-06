#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
client.py

交互式查看“最近 1 秒”点云的 2D 投影（cv2.imshow）。

交互：
- 运行 client.py 会自动 import 并启动 server（同进程后台线程采集）
- ESC：退出（会停止采集并断开连接）

2D 投影规则（按你之前的约定）：
- 图像大小：800x800
- x 为深度（距离），x 越大越亮；只保留 x>0
- y 轴朝左（y 越大越靠左）
- z 轴朝上（z 越大越靠上）
- 视场角过滤：x-z 平面总 FOV 70°（±35°），theta = atan2(z, x)
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from lidar_server import LivoxRealtimeServer
from tof_server import ToFRealtimeServer

CAPTURE_SECONDS = 2.0  # 显示/渲染“最近多少秒”的点云（采集滑窗长度）
DATA_DIR = Path(__file__).resolve().parent / "data"
SHOW_DONE_POPUP_ON_SAVE = True  # 一个场景数据保存完成后：弹窗 Done（不退出采集）

IMG_W = 700
IMG_H = 700
FOV_XZ_DEG = 70.0
HALF_FOV = np.deg2rad(FOV_XZ_DEG / 2.0)
MAX_RANGE_M = 20.0  # 固定量程（米）
NEAR_SAT_M = 1.0  # 近距离饱和：x<1m 直接当作 1m（亮度 255），避免过曝/除零
USE_COLORMAP = True  # 美观显示：用颜色表示“亮度”（亮度仍由距离决定）
COLORMAP_NAME = "turbo"
AUTO_EXPOSURE = True  # 自适应亮度/对比度：避免“远了全黑”
AE_LOW_PCT = 2.0  # 百分位拉伸：低端裁剪
AE_HIGH_PCT = 98.0  # 百分位拉伸：高端裁剪
AE_GAMMA = 0.8  # <1 会提亮暗部（更容易看清远处）；=1 关闭 gamma
AE_MEAN_SECONDS = 2.0  # 量程平滑：用最近 N 秒的均值（更稳定，不跳）

# ToF 显示尺寸（与 cali/check.py 对齐）
TOF_W = 40
TOF_H = 30
TOF_SHOW_W = 300
TOF_SHOW_H = 400

def _render_2d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    # 固定视场角（死的，不随点分布变化）：
    # - 水平：yaw = atan2(y, x)，±35°
    # - 竖直：pitch = atan2(z, x)，±35°
    # 只看前方：x>0
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    m = (x > 0) & (np.abs(yaw) <= HALF_FOV) & (np.abs(pitch) <= HALF_FOV)
    x, y, z = x[m], y[m], z[m]
    if x.size == 0:
        return np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    # 亮度反比例：1m=255, 2m≈128, 4m≈64（I≈255/x）
    # - 近处饱和：x<NEAR_SAT_M 视作 NEAR_SAT_M
    # - 远处上限：x>MAX_RANGE_M 视作 MAX_RANGE_M（保持“最大量程 20m”的约定）
    depth_m = np.clip(x, NEAR_SAT_M, MAX_RANGE_M)
    depth_u8 = np.clip(np.rint(255.0 / depth_m), 0.0, 255.0).astype(np.uint8)

    # 角度 -> 像素（固定映射，不做 min/max 拉伸）
    yaw = np.arctan2(y, x)
    pitch = np.arctan2(z, x)
    col = ((HALF_FOV - yaw) / (2.0 * HALF_FOV) * (IMG_W - 1)).astype(np.int32)
    row = ((HALF_FOV - pitch) / (2.0 * HALF_FOV) * (IMG_H - 1)).astype(np.int32)
    col = np.clip(col, 0, IMG_W - 1)
    row = np.clip(row, 0, IMG_H - 1)

    img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    # 同一像素多个点：取更远的（更亮/更热色）
    np.maximum.at(img, (row, col), depth_u8)
    return img


class _AutoExposeState:
    def __init__(self) -> None:
        # recent samples: (timestamp, lo, hi)
        self.samples: deque[tuple[float, float, float]] = deque()


def _auto_expose_u8(img: np.ndarray, st: _AutoExposeState, now_ts: float) -> np.ndarray:
    """对灰度图做轻量“自动曝光/对比度拉伸”（仅基于非零像素），并对量程做平滑。"""
    if (not AUTO_EXPOSURE) or img.size == 0:
        return img
    nz = img[img > 0]
    if nz.size < 500:  # 点太少就别折腾，避免闪烁
        return img
    lo_now = float(np.percentile(nz, AE_LOW_PCT))
    hi_now = float(np.percentile(nz, AE_HIGH_PCT))
    if hi_now <= lo_now + 1.0:
        return img

    # 最近 N 秒滑动均值：更稳定，减少量程跳变
    st.samples.append((now_ts, lo_now, hi_now))
    cutoff = now_ts - float(max(AE_MEAN_SECONDS, 0.05))
    while st.samples and st.samples[0][0] < cutoff:
        st.samples.popleft()
    if not st.samples:
        return img
    lo = float(np.mean([s[1] for s in st.samples]))
    hi = float(np.mean([s[2] for s in st.samples]))
    if hi <= lo + 1.0:
        return img

    out = (img.astype(np.float32) - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if AE_GAMMA != 1.0:
        out = np.power(out, AE_GAMMA)
    out = (out * 255.0).astype(np.uint8)
    out[img == 0] = 0
    return out


def _try_save_tof_raw(dest_raw: Path) -> bool:
    """
    按 get_tof.py 的方式尝试获取 tof.raw：
    - adb shell 触发生成 /tmp/tof.raw（通过 /tmp/sv_tof 机制）
    - adb pull /tmp/tof.raw 到本地
    """
    try:
        dest_raw.parent.mkdir(parents=True, exist_ok=True)
        # 触发设备侧生成 tof.raw
        trigger_cmd = "if [ -e /tmp/sv_tof ]; then rm /tmp/sv_tof && rm /tmp/tof.raw; fi && touch /tmp/sv_tof"
        subprocess.run(["adb", "shell", trigger_cmd], check=False, capture_output=True, text=True)

        # 等待设备写文件（给一点时间，避免 pull 到空文件）
        time.sleep(0.08)

        pull = subprocess.run(
            ["adb", "pull", "/tmp/tof.raw", str(dest_raw)],
            check=False,
            capture_output=True,
            text=True,
        )
        return pull.returncode == 0 and dest_raw.exists() and dest_raw.stat().st_size > 0
    except Exception:
        return False


def _show_done_popup(*, cv2, duration_ms: int = 400) -> None:
    """
    采集完成提示：绿色背景，白字 'Done'，duration_ms 后自动关闭。
    注意：cv2.waitKey 参数单位是毫秒。
    """
    w, h = 360, 160
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (0, 200, 0)  # BGR: green

    text = "Done"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max((w - tw) // 2, 0)
    y = max((h + th) // 2, th)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    win = "采集成功"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, img)
    cv2.waitKey(int(max(duration_ms, 1)))
    cv2.destroyWindow(win)


def _save_snapshot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    view_img: np.ndarray,
    tof_raw_bytes: bytes | None,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = DATA_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 点云：保存最近 CAPTURE_SECONDS 的快照（npz 通用、读写快）
    npz_path = out_dir / f"points_last{CAPTURE_SECONDS:.1f}s.npz"
    np.savez_compressed(
        npz_path,
        x=x.astype(np.float32, copy=False),
        y=y.astype(np.float32, copy=False),
        z=z.astype(np.float32, copy=False),
        capture_seconds=float(CAPTURE_SECONDS),
        saved_unix_ts=float(time.time()),
    )

    # 2) 当前 imshow 画面（png，无损）
    try:
        import cv2  # type: ignore

        cv2.imwrite(str(out_dir / "view.png"), view_img)
    except Exception:
        pass

    # 3) tof.raw（优先保存“ToF server 的最新帧”；没有则 fallback 到 adb 拉取一次）
    dest_raw = out_dir / "tof.raw"
    if tof_raw_bytes:
        try:
            dest_raw.write_bytes(tof_raw_bytes)
        except Exception:
            _try_save_tof_raw(dest_raw)
    else:
        _try_save_tof_raw(dest_raw)

    print(f"[SAVE] -> {out_dir} (pts={int(x.size)})")


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e
    colormap = cv2.COLORMAP_TURBO

    cv2.namedWindow("LIDAR (ESC=quit)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("TOF_REFLECT", cv2.WINDOW_AUTOSIZE)

    srv = LivoxRealtimeServer(max_seconds=CAPTURE_SECONDS)
    srv.start()

    tof_srv = ToFRealtimeServer(queue_maxlen=5, min_peak_count=100.0, target_fps=10.0)
    tof_srv.start()

    last_img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    last_pts = 0
    last_ts = 0.0
    ae_state = _AutoExposeState()

    last_tof = np.zeros((TOF_SHOW_H, TOF_SHOW_W, 3), dtype=np.uint8)
    last_tof_bytes: bytes | None = None

    try:
        while True:
            # 实时刷新（采集在后台线程持续进行）
            x, y, z = srv.snapshot_xyz()
            now_ts = time.time()
            depth_img = _auto_expose_u8(_render_2d(x, y, z), ae_state, now_ts)
            # 亮度由距离决定；可选把“亮度”上色（颜色仅用于美观）
            if USE_COLORMAP:
                last_img = cv2.applyColorMap(depth_img, colormap)
            else:
                last_img = depth_img
            last_pts = int(x.size)
            last_ts = now_ts
            cv2.setWindowTitle(
                "LIDAR (ESC=quit)",
                f"LIDAR | last={CAPTURE_SECONDS:.1f}s | pts={last_pts} | fov=70| I≈255/x(m) | "
                f"range=1~{MAX_RANGE_M:.0f}m | "
                f"ae={'on' if AUTO_EXPOSURE else 'off'}(mean={AE_MEAN_SECONDS:.1f}s) | "
                f"color={(COLORMAP_NAME if USE_COLORMAP else 'off')} | t={last_ts:.2f}",
            )

            # ToF：实时拿最新反射率图显示（没有数据就保持上一帧）
            tof_frame = tof_srv.get_latest()
            if tof_frame is not None and isinstance(tof_frame.reflect_u8, np.ndarray) and tof_frame.reflect_u8.size:
                inten_u8 = tof_frame.reflect_u8
                inten_u8 = cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE)
                inten_u8 = cv2.flip(inten_u8, 1) 
                inten_big = cv2.resize(inten_u8, (TOF_SHOW_W, TOF_SHOW_H), interpolation=cv2.INTER_NEAREST)
                last_tof = cv2.cvtColor(inten_big, cv2.COLOR_GRAY2BGR)
                last_tof_bytes = tof_frame.raw_bytes

            cv2.imshow("LIDAR (ESC=quit)", last_img)
            cv2.imshow("TOF_REFLECT", last_tof)

            key = int(cv2.waitKey(1) & 0xFF)
            if key == 32:  # SPACE：保存当前“最近 N 秒”点云 + tof.raw + jpg
                # 同步保存：实现更简单；保存时 UI 会短暂卡顿
                _save_snapshot(x=x, y=y, z=z, view_img=last_img.copy(), tof_raw_bytes=last_tof_bytes)
                if SHOW_DONE_POPUP_ON_SAVE:
                    _show_done_popup(cv2=cv2, duration_ms=400)
            if key == 27:  # ESC
                break
    finally:
        srv.stop()
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


