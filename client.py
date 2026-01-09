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
import time

import numpy as np

from server import LivoxRealtimeServer

CAPTURE_SECONDS = 2.0  # 显示/渲染“最近多少秒”的点云（采集滑窗长度）

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


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("缺少依赖 opencv-python，请先执行：py -m pip install opencv-python") from e
    colormap = cv2.COLORMAP_TURBO

    cv2.namedWindow("OpenPyLivox - 2D (ESC=quit)", cv2.WINDOW_AUTOSIZE)

    srv = LivoxRealtimeServer(max_seconds=CAPTURE_SECONDS)
    srv.start()

    last_img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    last_pts = 0
    last_ts = 0.0
    ae_state = _AutoExposeState()

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
                "OpenPyLivox - 2D (ESC=quit)",
                f"OpenPyLivox - 2D | last={CAPTURE_SECONDS:.1f}s | pts={last_pts} | fov=±35° | I≈255/x(m) | "
                f"range=1~{MAX_RANGE_M:.0f}m | "
                f"ae={'on' if AUTO_EXPOSURE else 'off'}(mean={AE_MEAN_SECONDS:.1f}s) | "
                f"color={(COLORMAP_NAME if USE_COLORMAP else 'off')} | t={last_ts:.2f}",
            )

            cv2.imshow("OpenPyLivox - 2D (ESC=quit)", last_img)
            key = int(cv2.waitKey(1) & 0xFF)
            if key == 27:  # ESC
                break
    finally:
        srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


