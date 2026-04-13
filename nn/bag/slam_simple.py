#!/usr/bin/env python3
"""
3D voxel SLAM for 30×40 dToF sensor.

- Correlative scan matching against the occupancy map
  (coarse-to-fine grid search over x, y, θ — NOT point-to-point ICP)
- Full 30×40 → 3D point cloud per frame
- Open3D live voxel grid visualization with height coloring

Controls (Open3D window):
    Left-drag     rotate
    Shift+drag    pan
    Scroll        zoom
    Close window  exit

Dependencies:
    pip install open3d opencv-python numpy mcap
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERR] open3d required — run: pip install open3d")

from unpack_mcap import BAG_NAME, TOF_H, TOF_W, TOPIC, load_frames, build_views

# ── sensor geometry ───────────────────────────────────────────────────
FOV_H_DEG = 55.0
FOV_V_DEG = FOV_H_DEG * TOF_H / TOF_W
MIN_DIST_MM, MAX_DIST_MM = 80, 4000

_h_step = FOV_H_DEG / max(TOF_W - 1, 1)
_v_step = FOV_V_DEG / max(TOF_H - 1, 1)
_h_ang = np.deg2rad((np.arange(TOF_W, dtype=np.float64) - (TOF_W - 1) / 2.0) * _h_step)
_v_ang = np.deg2rad(((TOF_H - 1) / 2.0 - np.arange(TOF_H, dtype=np.float64)) * _v_step)

_cv = np.cos(_v_ang)[:, None]
_sv = np.sin(_v_ang)[:, None]
_ch = np.cos(_h_ang)[None, :]
_sh = np.sin(_h_ang)[None, :]

_RAY_X = (_cv * _ch).ravel()
_RAY_Y = (_cv * _sh).ravel()
_RAY_Z = np.broadcast_to(_sv, (TOF_H, TOF_W)).ravel().copy()

_MID = slice(max(0, TOF_H // 2 - 2), min(TOF_H, TOF_H // 2 + 3))
_COS_2D, _SIN_2D = np.cos(_h_ang), np.sin(_h_ang)

# ── occupancy / map ──────────────────────────────────────────────────
MAP_PX = 800
MAP_M = 20.0
_MM_PER_PX = MAP_M * 1000.0 / MAP_PX
_MAP_HALF = MAP_PX // 2
_PAD = 200
_PAD_SIZE = MAP_PX + 2 * _PAD

# ── correlative scan matcher ─────────────────────────────────────────
MATCH_XY_COARSE = 500       # mm search range (each side)
MATCH_XY_STEP_C = 40        # mm
MATCH_TH_COARSE_DEG = 20.0
MATCH_TH_STEP_C_DEG = 2.0
MATCH_XY_FINE = 60          # mm
MATCH_XY_STEP_F = 5         # mm
MATCH_TH_FINE_DEG = 3.0
MATCH_TH_STEP_F_DEG = 0.3
MATCH_BLUR_SIGMA = 2.5      # pixels — smooths occupancy for gradient

# ── reflectance (from net.py) ─────────────────────────────────────────
REFLECT_K = 156250.0 / 3   # calibration constant (IS_6321)

# ── 3D display ────────────────────────────────────────────────────────
VOXEL_M = 0.05
DISPLAY_EVERY = 3
COMPACT_EVERY = 80


# =====================================================================
#  Scan / point-cloud extraction
# =====================================================================

def make_scan_2d(dist: np.ndarray, conf: np.ndarray) -> np.ndarray:
    band = dist[_MID].astype(np.int32)
    ok = (conf[_MID] > 0) & (band >= MIN_DIST_MM) & (band <= MAX_DIST_MM)
    col_ok = ok.any(axis=0)
    if not col_ok.any():
        return np.empty((0, 2), np.float64)
    med = np.ma.median(np.ma.array(band, mask=~ok), axis=0).filled(0).astype(np.float64)
    idx = np.flatnonzero(col_ok)
    r = med[idx]
    return np.column_stack((r * _COS_2D[idx], r * _SIN_2D[idx]))


def make_cloud_3d(dist: np.ndarray, conf: np.ndarray,
                   peak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full 30×40 → (Nx3 xyz_mm, N reflectance ∈ [0,1])."""
    d = dist.ravel().astype(np.float64)
    mask = (conf.ravel() > 0) & (d >= MIN_DIST_MM) & (d <= MAX_DIST_MM)
    dv = d[mask]
    pv = peak.ravel().astype(np.float64)[mask]
    dist_m = dv / 1000.0
    refl = np.clip(dist_m * dist_m * pv / REFLECT_K, 0.0, 1.0)
    xyz = np.column_stack((_RAY_X[mask] * dv, _RAY_Y[mask] * dv, _RAY_Z[mask] * dv))
    return xyz, refl


# =====================================================================
#  Pose helpers  (x_mm, y_mm, theta_rad)
# =====================================================================

def pose_to_mat(x: float, y: float, th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]], np.float64)


def apply_tf(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts.copy()
    return pts @ M[:2, :2].T + M[:2, 2]


def apply_tf_3d(M2d: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    if pts3d.shape[0] == 0:
        return pts3d.copy()
    xy = pts3d[:, :2] @ M2d[:2, :2].T + M2d[:2, 2]
    return np.column_stack((xy, pts3d[:, 2]))


# =====================================================================
#  Occupancy grid
# =====================================================================

def _mm2px(x_mm: float, y_mm: float) -> tuple[int, int]:
    return (int(np.clip(_MAP_HALF + x_mm / _MM_PER_PX, 0, MAP_PX - 1)),
            int(np.clip(_MAP_HALF - y_mm / _MM_PER_PX, 0, MAP_PX - 1)))


def update_occ(logodds: np.ndarray, rpx: tuple[int, int], pts2d_mm: np.ndarray) -> None:
    if pts2d_mm.shape[0] == 0:
        return
    epx = np.column_stack((
        np.clip(_MAP_HALF + pts2d_mm[:, 0] / _MM_PER_PX, 0, MAP_PX - 1).astype(np.int32),
        np.clip(_MAP_HALF - pts2d_mm[:, 1] / _MM_PER_PX, 0, MAP_PX - 1).astype(np.int32)))
    free = np.zeros((MAP_PX, MAP_PX), np.uint8)
    for ex, ey in epx:
        cv2.line(free, rpx, (int(ex), int(ey)), 1, 1)
    logodds -= free.astype(np.int16)
    np.add.at(logodds, (epx[:, 1], epx[:, 0]), 4)
    np.clip(logodds, -40, 40, out=logodds)


# =====================================================================
#  Correlative scan matcher
#
#  For each candidate (x, y, θ), rotate the scan by θ, translate by
#  (x, y), project endpoints onto the smoothed occupancy map, and sum
#  up the scores.  The pose with the highest score wins.
#
#  Vectorised over (dx, dy) for each θ candidate; typically runs in
#  a few milliseconds per frame.
# =====================================================================

def make_score_map(logodds: np.ndarray) -> np.ndarray:
    """Gaussian-blurred occupancy → smooth score field for matching."""
    return cv2.GaussianBlur(logodds.astype(np.float32), (0, 0),
                            sigmaX=MATCH_BLUR_SIGMA)


def _grid_search(score_map_padded: np.ndarray,
                 scan_2d: np.ndarray,
                 x0: float, y0: float, th0: float,
                 xy_range: float, xy_step: float,
                 th_range_deg: float, th_step_deg: float,
                 ) -> tuple[float, float, float, float]:
    """
    Brute-force search over (x0±xy_range, y0±xy_range, th0±th_range).
    score_map_padded is already padded by _PAD on each side.
    Returns (best_x, best_y, best_th, best_score).
    """
    dxs = np.arange(-xy_range, xy_range + xy_step * 0.5, xy_step)
    dys = np.arange(-xy_range, xy_range + xy_step * 0.5, xy_step)
    dths = np.deg2rad(np.arange(-th_range_deg, th_range_deg + th_step_deg * 0.5,
                                th_step_deg))

    best_score = -1e18
    best = (x0, y0, th0)

    for dth in dths:
        th = th0 + dth
        c, s = np.cos(th), np.sin(th)
        pts_rot = scan_2d @ np.array([[c, s], [-s, c]])  # (N, 2) rotated

        # base pixel coords at pose (x0, y0)
        base_px = (_MAP_HALF + (pts_rot[:, 0] + x0) / _MM_PER_PX + _PAD)
        base_py = (_MAP_HALF - (pts_rot[:, 1] + y0) / _MM_PER_PX + _PAD)

        # per-candidate pixel shifts
        dx_px = dxs / _MM_PER_PX            # (Ndx,)
        dy_px = -dys / _MM_PER_PX           # (Ndy,) — y-axis flipped in pixel space

        all_px = np.clip(
            (base_px[None, :] + dx_px[:, None]).astype(np.int32),
            0, _PAD_SIZE - 1)               # (Ndx, N)
        all_py = np.clip(
            (base_py[None, :] + dy_px[:, None]).astype(np.int32),
            0, _PAD_SIZE - 1)               # (Ndy, N)

        # score[i, j] = Σ_k map[py[j,k], px[i,k]]   — fully vectorised
        scores = score_map_padded[
            all_py[None, :, :],   # (1, Ndy, N)
            all_px[:, None, :]    # (Ndx, 1, N)
        ].sum(axis=2)             # (Ndx, Ndy)

        flat = int(scores.argmax())
        sc = float(scores.ravel()[flat])
        if sc > best_score:
            bi, bj = divmod(flat, scores.shape[1])
            best_score = sc
            best = (x0 + dxs[bi], y0 + dys[bj], th)

    return (*best, best_score)


def correlative_match(logodds: np.ndarray, scan_2d: np.ndarray,
                      x0: float, y0: float, th0: float,
                      ) -> tuple[float, float, float, float]:
    """Two-pass coarse → fine scan-to-map matching."""
    sm = make_score_map(logodds)
    sm_pad = np.zeros((_PAD_SIZE, _PAD_SIZE), np.float32)
    sm_pad[_PAD:_PAD + MAP_PX, _PAD:_PAD + MAP_PX] = sm

    # coarse pass
    cx, cy, ct, _ = _grid_search(
        sm_pad, scan_2d, x0, y0, th0,
        MATCH_XY_COARSE, MATCH_XY_STEP_C,
        MATCH_TH_COARSE_DEG, MATCH_TH_STEP_C_DEG)

    # fine pass around coarse result
    fx, fy, ft, score = _grid_search(
        sm_pad, scan_2d, cx, cy, ct,
        MATCH_XY_FINE, MATCH_XY_STEP_F,
        MATCH_TH_FINE_DEG, MATCH_TH_STEP_F_DEG)

    return fx, fy, ft, score


# =====================================================================
#  Open3D helpers
# =====================================================================

def refl_to_gray(refl: np.ndarray) -> np.ndarray:
    """Reflectance [0,1] → Nx3 grayscale (0%=black, 100%=white)."""
    t = np.clip(refl, 0.0, 1.0).ravel()
    return np.column_stack((t, t, t))


def make_ground_grid(extent_m: float = 10.0, step_m: float = 1.0) -> o3d.geometry.LineSet:
    pts, lines, colors = [], [], []
    idx = 0
    n = int(extent_m / step_m)
    for i in range(-n, n + 1):
        v = i * step_m
        for a, b in [([v, -extent_m, 0.0], [v, extent_m, 0.0]),
                      ([-extent_m, v, 0.0], [extent_m, v, 0.0])]:
            pts.extend([a, b])
            lines.append([idx, idx + 1])
            colors.append([0.0, 0.35, 0.0] if i == 0 else [0.22, 0.22, 0.22])
            idx += 2
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_axes(length: float = 0.5) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(
        [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    ls.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    ls.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return ls


def make_camera_frustum(x_mm: float, y_mm: float, th: float,
                        length_m: float = 0.3) -> o3d.geometry.LineSet:
    rx_m, ry_m = x_mm / 1000.0, y_mm / 1000.0
    c, s = np.cos(th), np.sin(th)
    R2d = np.array([[c, -s], [s, c]])

    fh = np.deg2rad(FOV_H_DEG / 2.0)
    fv = np.deg2rad(FOV_V_DEG / 2.0)
    cf, sf = np.cos(fh), np.sin(fh)
    cvf, svf = np.cos(fv), np.sin(fv)
    L = length_m
    corners_local = np.array([
        [cvf * cf, -cvf * sf,  svf],
        [cvf * cf,  cvf * sf,  svf],
        [cvf * cf,  cvf * sf, -svf],
        [cvf * cf, -cvf * sf, -svf],
    ]) * L
    fwd_local = np.array([[L, 0.0, 0.0]])

    all_local = np.vstack(([[0.0, 0.0, 0.0]], corners_local, fwd_local))
    xy_rot = all_local[:, :2] @ R2d.T
    all_global = np.column_stack((xy_rot[:, 0] + rx_m, xy_rot[:, 1] + ry_m, all_local[:, 2]))

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1],
             [0, 5]]
    colors = [[0.0, 0.9, 0.9]] * 8 + [[1.0, 0.2, 0.2]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_global)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def build_voxels(pts_m: np.ndarray, peak: np.ndarray,
                  voxel_size: float) -> o3d.geometry.VoxelGrid | None:
    if pts_m.shape[0] < 3:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_m)
    pcd.colors = o3d.utility.Vector3dVector(refl_to_gray(peak))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


# =====================================================================
#  Main
# =====================================================================

def _draw_sensor_view(fr, view_idx: int, slam_idx: int, n_frames: int,
                      pose: tuple[float, float, float]) -> np.ndarray:
    """Render the cv2 sensor preview: dist (JET) + peak (gray) + info bar."""
    resize_wh = (400, 300)
    dist_view, peak_view = build_views(fr.dist, fr.conf, fr.peak, resize_wh)
    canvas = np.hstack([dist_view, peak_view])

    bar_h = 48
    bar = np.zeros((bar_h, canvas.shape[1], 3), np.uint8)
    px, py, th = pose
    line1 = (f"view {view_idx + 1}/{n_frames}  |  "
             f"SLAM at {slam_idx + 1}/{n_frames}  |  "
             f"pose ({px:+.0f}, {py:+.0f})mm {np.degrees(th):+.1f}deg")
    line2 = "Right/D=step  Left/A=back  Space=play/pause  Q=quit"
    cv2.putText(bar, line1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


# arrow key codes (Windows cv2.waitKeyEx)
_KEY_LEFT = 2424832
_KEY_RIGHT = 2555904


def main() -> int:
    if not BAG_NAME:
        print('[WARN] set BAG_NAME in unpack_mcap.py, e.g. "0.bag"')
        return 1
    bag = (Path.cwd() / BAG_NAME).resolve()
    if not bag.exists():
        print(f"[WARN] not found: {bag}")
        return 1
    try:
        frames = load_frames([bag], topic_filter=TOPIC)
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1
    if not frames:
        print("[WARN] no frames")
        return 1

    N = len(frames)
    print(f"[INFO] {N} frames loaded")
    print("[INFO] Right/D = step SLAM forward   Left/A = browse back   "
          "Space = play/pause   Q = quit")

    # ── SLAM state ────────────────────────────────────────────────────
    pose_x, pose_y, pose_th = 0.0, 0.0, 0.0
    logodds = np.zeros((MAP_PX, MAP_PX), np.int16)
    pts_chunks: list[np.ndarray] = []
    refl_chunks: list[np.ndarray] = []
    all_pts_m = np.empty((0, 3), np.float64)
    all_refl = np.empty((0,), np.float64)
    traj_mm: list[tuple[float, float]] = [(0.0, 0.0)]
    total_dist_mm = 0.0

    # seed with frame 0
    sc0 = make_scan_2d(frames[0].dist, frames[0].conf)
    cl0_xyz, cl0_refl = make_cloud_3d(frames[0].dist, frames[0].conf, frames[0].peak)
    T0 = pose_to_mat(0.0, 0.0, 0.0)
    if sc0.shape[0] > 0:
        update_occ(logodds, _mm2px(0, 0), apply_tf(T0, sc0))
    if cl0_xyz.shape[0] > 0:
        pts_chunks.append(apply_tf_3d(T0, cl0_xyz) / 1000.0)
        refl_chunks.append(cl0_refl)

    slam_idx = 0   # last frame processed by SLAM
    view_idx = 0   # frame shown in sensor preview

    # ── Open3D ────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D ToF SLAM", 1280, 800)
    vis.add_geometry(make_ground_grid())
    vis.add_geometry(make_axes(0.5))

    traj_ls = o3d.geometry.LineSet()
    vis.add_geometry(traj_ls)
    cur_voxels: o3d.geometry.VoxelGrid | None = None
    cam_frustum: o3d.geometry.LineSet | None = None

    opt = vis.get_render_option()
    opt.background_color = np.array([0.06, 0.06, 0.10])
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front([-0.35, -0.25, 0.65])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.35)

    # ── cv2 sensor preview ────────────────────────────────────────────
    cv2.namedWindow("Sensor", cv2.WINDOW_AUTOSIZE)

    need_3d_refresh = True
    playing = False
    window_open = True

    # ── helper: advance SLAM by one frame ─────────────────────────────
    def slam_step() -> bool:
        """Process next frame. Returns True if a frame was processed."""
        nonlocal slam_idx, pose_x, pose_y, pose_th, total_dist_mm
        nonlocal need_3d_refresh
        if slam_idx >= N - 1:
            return False
        slam_idx += 1
        fr = frames[slam_idx]
        sc = make_scan_2d(fr.dist, fr.conf)
        cl_xyz, cl_refl = make_cloud_3d(fr.dist, fr.conf, fr.peak)

        if sc.shape[0] >= 4:
            prev_x, prev_y = pose_x, pose_y
            pose_x, pose_y, pose_th, score = correlative_match(
                logodds, sc, pose_x, pose_y, pose_th)
            step_mm = np.hypot(pose_x - prev_x, pose_y - prev_y)
            total_dist_mm += step_mm

            T = pose_to_mat(pose_x, pose_y, pose_th)
            g2d = apply_tf(T, sc)
            update_occ(logodds, _mm2px(pose_x, pose_y), g2d)
            traj_mm.append((pose_x, pose_y))

            if slam_idx % 50 == 0 or slam_idx < 10:
                print(f"\n  [{slam_idx:4d}] pose=({pose_x:+.0f}, {pose_y:+.0f})mm "
                      f"th={np.degrees(pose_th):+.1f}° "
                      f"step={step_mm:.0f}mm score={score:.0f}")

        if cl_xyz.shape[0] > 0:
            T = pose_to_mat(pose_x, pose_y, pose_th)
            pts_chunks.append(apply_tf_3d(T, cl_xyz) / 1000.0)
            refl_chunks.append(cl_refl)

        # periodic compact
        if slam_idx % COMPACT_EVERY == 0 and pts_chunks:
            merged = np.vstack(pts_chunks)
            merged_r = np.concatenate(refl_chunks)
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(merged)
            pcd_tmp.colors = o3d.utility.Vector3dVector(refl_to_gray(merged_r))
            pcd_tmp = pcd_tmp.voxel_down_sample(VOXEL_M * 0.5)
            ds_pts = np.asarray(pcd_tmp.points).copy()
            ds_refl = np.asarray(pcd_tmp.colors)[:, 0].copy()
            pts_chunks.clear()
            refl_chunks.clear()
            pts_chunks.append(ds_pts)
            refl_chunks.append(ds_refl)

        need_3d_refresh = True
        return True

    # ── helper: refresh 3D display ────────────────────────────────────
    def refresh_3d():
        nonlocal cur_voxels, cam_frustum, all_pts_m, all_refl, need_3d_refresh
        if not need_3d_refresh:
            return
        need_3d_refresh = False

        if pts_chunks:
            all_pts_m = np.vstack(pts_chunks)
            all_refl = np.concatenate(refl_chunks)
        vg = build_voxels(all_pts_m, all_refl, VOXEL_M)
        if vg is not None:
            if cur_voxels is not None:
                vis.remove_geometry(cur_voxels, reset_bounding_box=False)
            vis.add_geometry(vg, reset_bounding_box=(slam_idx <= 1))
            cur_voxels = vg

        if len(traj_mm) >= 2:
            arr = np.array(traj_mm) / 1000.0
            pts3 = np.column_stack((arr, np.zeros(len(arr))))
            ln = [[i, i + 1] for i in range(len(arr) - 1)]
            traj_ls.points = o3d.utility.Vector3dVector(pts3)
            traj_ls.lines = o3d.utility.Vector2iVector(ln)
            traj_ls.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * len(ln))
            vis.update_geometry(traj_ls)

        if cam_frustum is not None:
            vis.remove_geometry(cam_frustum, reset_bounding_box=False)
        cam_frustum = make_camera_frustum(pose_x, pose_y, pose_th, 0.25)
        vis.add_geometry(cam_frustum, reset_bounding_box=False)

        print(f"\r  SLAM frame {slam_idx + 1}/{N} | {all_pts_m.shape[0]} pts"
              f" | travel {total_dist_mm / 1000:.2f}m ", end="", flush=True)

    # ── main event loop ───────────────────────────────────────────────
    refresh_3d()

    while True:
        # sensor preview
        sensor_img = _draw_sensor_view(
            frames[view_idx], view_idx, slam_idx, N,
            (pose_x, pose_y, pose_th))
        cv2.imshow("Sensor", sensor_img)

        # Open3D
        if not vis.poll_events():
            window_open = False
            break
        vis.update_renderer()

        # auto-play mode
        if playing and slam_idx < N - 1:
            slam_step()
            view_idx = slam_idx
            refresh_3d()

        # keyboard (cv2)
        key = cv2.waitKeyEx(30)
        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (_KEY_RIGHT, ord("d"), ord("D")):
            if view_idx < slam_idx:
                view_idx += 1
            elif slam_idx < N - 1:
                slam_step()
                view_idx = slam_idx
                refresh_3d()
        elif key in (_KEY_LEFT, ord("a"), ord("A")):
            if view_idx > 0:
                view_idx -= 1
        elif key == ord(" "):
            playing = not playing
            print(f"\n  {'PLAY' if playing else 'PAUSE'}")

    print()
    cv2.destroyAllWindows()
    if window_open:
        print("[INFO] close 3D window to exit")
        vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
