#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3d.py

基于 nn/check.py 的推理流程，显示预测深度的 3D 点云（Open3D 交互窗口）：
- 鼠标左键拖动旋转，滚轮缩放，右键平移
- 键盘 4/6 切换样本
- 键盘 -/= 调整置信阈值（低于阈值的点不显示）
- 键盘 R 重新自适应视角，Q 或 ESC 退出

同时保留 2D 检查窗口（参考 check.py）：
- INPUT / GT / PRED / P_OK 四宫格

距离显示上限：30 米（>30m 的点直接丢弃）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

TOF_H = 30
TOF_W = 40
TOF_C = 64
HIST_BINS = 62

MAX_DIST_M = 30.0
EPS = 1e-6
DISP_GAMMA = 1.2

SHOW_W = 390
SHOW_H = 520
HEADER_H = 32

# 传感器 FOV 为 60°*80°：长边使用 80°，短边使用 60°。
FOV_LONG_DEG = 80.0
FOV_SHORT_DEG = 60.0


def _find_pairs(train_dir: Path) -> List[Tuple[Path, Path]]:
    ins = sorted(train_dir.glob("input_*.npy"))
    pairs: List[Tuple[Path, Path]] = []
    for ip in ins:
        op = train_dir / ip.name.replace("input_", "output_", 1)
        if op.exists():
            pairs.append((ip, op))
    return pairs


def _load_pair(ip: Path, op: Path) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(str(ip)).astype(np.float32, copy=False)
    y = np.load(str(op)).astype(np.float32, copy=False)
    if x.shape != (TOF_H, TOF_W, TOF_C):
        raise ValueError(f"bad input shape: {x.shape} ({ip})")
    if y.shape != (TOF_H, TOF_W):
        raise ValueError(f"bad output shape: {y.shape} ({op})")
    return x, y


def _with_text(img_bgr: np.ndarray, text: str) -> np.ndarray:
    import cv2  # type: ignore

    out = img_bgr.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _render_input_intensity_u8(hists: np.ndarray) -> np.ndarray:
    h = hists.astype(np.float32, copy=False)
    inten = np.sum(h[:, :, :HIST_BINS], axis=2)
    if not inten.size:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    m = np.isfinite(inten)
    if not np.any(m):
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    vmin = float(np.percentile(inten[m], 5))
    vmax = float(np.percentile(inten[m], 95))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
        return np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    norm = np.clip((inten - vmin) / (vmax - vmin), 0.0, 1.0)
    gamma = float(DISP_GAMMA)
    if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        norm = np.power(norm, 1.0 / gamma)
    return np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)


def _inv_depth_range_from_depth(depth_m: np.ndarray) -> tuple[float, float] | None:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return None
    inv_v = 1.0 / np.clip(d[valid], EPS, np.inf)
    vmin = float(np.min(inv_v))
    vmax = float(np.max(inv_v))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _colorize_depth_with_range(depth_m: np.ndarray, inv_vmin: float, inv_vmax: float) -> np.ndarray:
    import cv2  # type: ignore

    d = np.asarray(depth_m, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        return np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)

    inv = np.zeros_like(d, dtype=np.float32)
    inv[valid] = 1.0 / np.clip(d[valid], EPS, np.inf)

    vmin = float(inv_vmin)
    vmax = float(inv_vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    u8 = np.zeros((TOF_H, TOF_W), dtype=np.uint8)
    u8[valid] = np.clip(np.rint((inv[valid] - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    bgr[~valid] = (0, 0, 0)
    return bgr


def _colorize_prob(prob: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=np.float32)
    m = valid.astype(bool)

    gamma = float(DISP_GAMMA)
    disp = np.zeros((TOF_H, TOF_W), dtype=np.float32)
    disp[m] = np.power(np.clip(p[m], 0.0, 1.0), 1.0 / gamma)

    u8 = np.clip(np.rint(disp * 255.0), 0, 255).astype(np.uint8)
    bgr = np.stack([u8, u8, u8], axis=2)
    bgr[~m] = (0, 0, 0)
    return bgr


def _depth_to_point_cloud(
    depth_m: np.ndarray,
    intensity_u8: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    inten = np.asarray(intensity_u8, dtype=np.float32)
    c = np.asarray(conf, dtype=np.float32)
    if depth.ndim != 2 or inten.shape != depth.shape or c.shape != depth.shape:
        raise ValueError("depth/intensity/conf shape mismatch")

    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    mask = np.isfinite(depth) & (depth > 0.0) & (depth <= float(MAX_DIST_M)) & np.isfinite(c) & (c >= float(conf_thr))
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    d = depth[mask]
    fov_x = FOV_LONG_DEG if w >= h else FOV_SHORT_DEG
    fov_y = FOV_SHORT_DEG if w >= h else FOV_LONG_DEG
    fx = (w * 0.5) / np.tan(np.deg2rad(float(fov_x) * 0.5))
    fy = (h * 0.5) / np.tan(np.deg2rad(float(fov_y) * 0.5))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5

    u = xx[mask]
    v = yy[mask]
    x = (u - cx) / fx * d
    y = -(v - cy) / fy * d
    # 使用右手系并让“前方”落在负 Z，避免初始从背面观察。
    z = -d

    pts = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
    g = np.clip(inten[mask] / 255.0, 0.0, 1.0)
    cols = np.stack([g, g, g], axis=1).astype(np.float32, copy=False)
    return pts, cols


def _build_2d_view(
    x: np.ndarray,
    gt: np.ndarray,
    pred_depth: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    idx: int,
    total: int,
    points_n: int,
) -> np.ndarray:
    import cv2  # type: ignore

    inten_u8 = _render_input_intensity_u8(x)
    inten_u8 = cv2.flip(cv2.rotate(inten_u8, cv2.ROTATE_90_CLOCKWISE), 1)
    in_big = cv2.resize(inten_u8, (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    in_bgr = cv2.cvtColor(in_big, cv2.COLOR_GRAY2BGR)

    gt_for_disp = np.where(np.isfinite(gt) & (gt > 0.0) & (gt <= float(MAX_DIST_M)), gt, 0.0)
    inv_range = _inv_depth_range_from_depth(gt_for_disp)
    if inv_range is None:
        gt_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
        pred_bgr = np.zeros((TOF_H, TOF_W, 3), dtype=np.uint8)
    else:
        inv_vmin, inv_vmax = inv_range
        gt_bgr = _colorize_depth_with_range(gt_for_disp, inv_vmin, inv_vmax)
        pred_bgr = _colorize_depth_with_range(pred_depth, inv_vmin, inv_vmax)

    conf_mask = conf >= float(conf_thr)
    if np.any(~conf_mask):
        pred_bgr = pred_bgr.copy()
        pred_bgr[~conf_mask] = (0, 0, 0)

    valid_all = np.ones((TOF_H, TOF_W), dtype=bool)
    prob_bgr = _colorize_prob(conf, valid_all)

    gt_big = cv2.resize(cv2.flip(cv2.rotate(gt_bgr, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST)
    pred_big = cv2.resize(
        cv2.flip(cv2.rotate(pred_bgr, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST
    )
    prob_big = cv2.resize(
        cv2.flip(cv2.rotate(prob_bgr, cv2.ROTATE_90_CLOCKWISE), 1), (SHOW_W, SHOW_H), interpolation=cv2.INTER_NEAREST
    )

    in_bgr = _with_text(in_bgr, f"INPUT(refl, gamma={DISP_GAMMA:g})")
    gt_big = _with_text(gt_big, "GT(<=30m)")
    pred_big = _with_text(pred_big, "PRED(conf mask)")
    prob_big = _with_text(prob_big, "P_OK(+-7%)")

    top = np.hstack([in_bgr, gt_big])
    bot = np.hstack([pred_big, prob_big])
    view = np.vstack([top, bot])

    header = np.zeros((HEADER_H, view.shape[1], 3), dtype=np.uint8)
    header_txt = (
        f"sample {idx + 1}/{total}  "
        f"conf_thr {conf_thr:.2f}  "
        f"points {points_n}  "
        f"max_dist {MAX_DIST_M:.1f}m  "
        f"keys: 4/6 -/= R Q/ESC"
    )
    header = _with_text(header, header_txt)
    return np.vstack([header, view])


def main() -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("missing dependency opencv-python, run: py -m pip install opencv-python") from e
    try:
        import torch
    except Exception as e:
        raise RuntimeError("missing dependency torch") from e
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("missing dependency open3d, run: py -m pip install open3d") from e

    nn_dir = Path(__file__).resolve().parent
    train_dir = nn_dir / "train_data"
    ckpt_path = nn_dir / "model_last.pt"

    if str(nn_dir) not in sys.path:
        sys.path.insert(0, str(nn_dir))
    from net import Network  # noqa: E402

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train_data dir: {train_dir}")
    pairs = _find_pairs(train_dir)
    if not pairs:
        raise FileNotFoundError(f"no input/output pairs found under: {train_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Network(in_channels=TOF_C).to(device)
    net.eval()

    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        net.load_state_dict(sd, strict=True)
        print(f"[load] {ckpt_path}")
    else:
        print(f"[warn] missing checkpoint: {ckpt_path} (use random weights)")

    state = {
        "idx": 0,
        "conf_thr": 0.50,
        "dirty": True,
        "need_fit": False,
    }

    cv2.namedWindow("CHECK_3D_AUX", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("pThr%", "CHECK_3D_AUX", 50, 100, lambda _: None)

    pcd = o3d.geometry.PointCloud()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0.0, 0.0, 0.0])
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TOF_3D_POINT_CLOUD", width=1280, height=860)
    vis.add_geometry(pcd)
    vis.add_geometry(frame)

    ropt = vis.get_render_option()
    ropt.background_color = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    ropt.point_size = 5.0
    ropt.show_coordinate_frame = True

    # 先跑一次事件循环，确保窗口尺寸已初始化，避免 ViewControl 警告。
    vis.poll_events()
    vis.update_renderer()

    latest_view = np.zeros((HEADER_H + SHOW_H * 2, SHOW_W * 2, 3), dtype=np.uint8)

    def _mark_dirty() -> None:
        state["dirty"] = True

    def _on_prev(v: o3d.visualization.Visualizer) -> bool:
        state["idx"] = (int(state["idx"]) - 1) % len(pairs)
        _mark_dirty()
        return False

    def _on_next(v: o3d.visualization.Visualizer) -> bool:
        state["idx"] = (int(state["idx"]) + 1) % len(pairs)
        _mark_dirty()
        return False

    def _on_thr_down(v: o3d.visualization.Visualizer) -> bool:
        state["conf_thr"] = float(np.clip(float(state["conf_thr"]) - 0.05, 0.0, 1.0))
        cv2.setTrackbarPos("pThr%", "CHECK_3D_AUX", int(round(float(state["conf_thr"]) * 100.0)))
        _mark_dirty()
        return False

    def _on_thr_up(v: o3d.visualization.Visualizer) -> bool:
        state["conf_thr"] = float(np.clip(float(state["conf_thr"]) + 0.05, 0.0, 1.0))
        cv2.setTrackbarPos("pThr%", "CHECK_3D_AUX", int(round(float(state["conf_thr"]) * 100.0)))
        _mark_dirty()
        return False

    def _on_refit(v: o3d.visualization.Visualizer) -> bool:
        state["need_fit"] = True
        return False

    def _on_quit(v: o3d.visualization.Visualizer) -> bool:
        v.close()
        return False

    vis.register_key_callback(ord("4"), _on_prev)
    vis.register_key_callback(ord("6"), _on_next)
    vis.register_key_callback(ord("-"), _on_thr_down)
    vis.register_key_callback(ord("="), _on_thr_up)
    vis.register_key_callback(ord("R"), _on_refit)
    vis.register_key_callback(ord("Q"), _on_quit)

    while True:
        # 同步 trackbar -> conf_thr
        p_thr = float(cv2.getTrackbarPos("pThr%", "CHECK_3D_AUX")) / 100.0
        if abs(p_thr - float(state["conf_thr"])) > 1e-6:
            state["conf_thr"] = p_thr
            _mark_dirty()

        if bool(state["dirty"]):
            ip, op = pairs[int(state["idx"])]
            x, gt = _load_pair(ip, op)

            with torch.no_grad():
                inp = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                out_t = net.forward_train(inp)
                pred_depth = out_t["dist"][:, 0, :, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                conf = out_t["conf"][:, 0, :, :].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

            inten_u8 = _render_input_intensity_u8(x)
            # 与四宫格显示一致：rot90CW + flipH 等价于转置，这里用同一坐标系构点云。
            pred_disp = np.ascontiguousarray(pred_depth.T)
            conf_disp = np.ascontiguousarray(conf.T)
            inten_disp = np.ascontiguousarray(inten_u8.T)
            points, colors = _depth_to_point_cloud(pred_disp, inten_disp, conf_disp, float(state["conf_thr"]))

            if points.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
                pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

            vis.update_geometry(pcd)
            vis.update_geometry(frame)
            state["need_fit"] = True

            latest_view = _build_2d_view(
                x=x,
                gt=gt,
                pred_depth=pred_depth,
                conf=conf,
                conf_thr=float(state["conf_thr"]),
                idx=int(state["idx"]),
                total=len(pairs),
                points_n=int(points.shape[0]),
            )
            print(
                f"[sample {int(state['idx']) + 1}/{len(pairs)}] "
                f"conf_thr={float(state['conf_thr']):.2f} "
                f"points={points.shape[0]} "
                f"max_dist={MAX_DIST_M:.1f}m"
            )
            state["dirty"] = False

        cv2.imshow("CHECK_3D_AUX", latest_view)
        k = int(cv2.waitKey(1) & 0xFF)
        if k in (27, ord("q"), ord("Q")):
            break
        if k == ord("4"):
            state["idx"] = (int(state["idx"]) - 1) % len(pairs)
            _mark_dirty()
        elif k == ord("6"):
            state["idx"] = (int(state["idx"]) + 1) % len(pairs)
            _mark_dirty()
        elif k == ord("-"):
            state["conf_thr"] = float(np.clip(float(state["conf_thr"]) - 0.05, 0.0, 1.0))
            cv2.setTrackbarPos("pThr%", "CHECK_3D_AUX", int(round(float(state["conf_thr"]) * 100.0)))
            _mark_dirty()
        elif k == ord("="):
            state["conf_thr"] = float(np.clip(float(state["conf_thr"]) + 0.05, 0.0, 1.0))
            cv2.setTrackbarPos("pThr%", "CHECK_3D_AUX", int(round(float(state["conf_thr"]) * 100.0)))
            _mark_dirty()
        elif k in (ord("r"), ord("R")):
            state["need_fit"] = True

        alive = vis.poll_events()
        if not alive:
            break
        vis.update_renderer()

        if bool(state["need_fit"]):
            vis.reset_view_point(True)
            state["need_fit"] = False

    vis.destroy_window()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
