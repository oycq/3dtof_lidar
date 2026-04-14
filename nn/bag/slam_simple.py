#!/usr/bin/env python3
"""
3D LiDAR SLAM for 30×40 dToF sensor — powered by KISS-ICP.

Uses the open-source KISS-ICP library (point-to-point ICP with adaptive
thresholding and voxel hash map) for scan registration, replacing the
hand-rolled correlative scan matcher.

Controls (Open3D window):
    Left-drag     rotate
    Shift+drag    pan
    Scroll        zoom
    Close window  exit

Controls (cv2 window):
    Right / D     step SLAM forward
    Left  / A     browse back
    Space         play / pause
    C / V         lower / raise ceiling clip  (±0.2 m)
    F             toggle floor clip on/off
    +  / -        increase / decrease point size
    Q / Esc       quit

Dependencies:
    pip install kiss-icp open3d opencv-python numpy mcap
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERR] open3d required — run: pip install open3d")

try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
except ImportError:
    sys.exit("[ERR] kiss-icp required — run: pip install kiss-icp")

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

# ── reflectance calibration ──────────────────────────────────────────
REFLECT_K = 156250.0 / 3

# ── 3D display ────────────────────────────────────────────────────────
POINT_SIZE = 4.0              # px — sparse sensor needs fat points
DOWNSAMPLE_VOXEL = 0.03       # keep more detail for a 30×40 sensor
COMPACT_EVERY = 120
CEIL_CLIP_DEFAULT = 100.0     # effectively off — user presses C to lower
FLOOR_CLIP_DEFAULT = -100.0   # effectively off — user presses F to enable
BBOX_UPDATE_EVERY = 30        # re-sync Open3D far-clip plane every N frames
CEIL_STEP = 0.2               # C/V key step


# =====================================================================
#  Point-cloud extraction
# =====================================================================

def make_cloud_3d(dist: np.ndarray, conf: np.ndarray,
                  peak: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full 30×40 → (Nx3 xyz in metres, N reflectance ∈ [0,1])."""
    d = dist.ravel().astype(np.float64)
    mask = (conf.ravel() > 0) & (d >= MIN_DIST_MM) & (d <= MAX_DIST_MM)
    dv = d[mask]
    pv = peak.ravel().astype(np.float64)[mask]
    dist_m = dv / 1000.0
    refl = np.clip(dist_m * dist_m * pv / REFLECT_K, 0.0, 1.0)
    xyz_m = np.column_stack((
        _RAY_X[mask] * dist_m,
        _RAY_Y[mask] * dist_m,
        _RAY_Z[mask] * dist_m,
    ))
    return xyz_m, refl


# =====================================================================
#  KISS-ICP config tailored for the dToF sensor
# =====================================================================

def make_kiss_config() -> KISSConfig:
    cfg = KISSConfig()
    cfg.data.max_range = MAX_DIST_MM / 1000.0   # 4.0 m
    cfg.data.min_range = MIN_DIST_MM / 1000.0    # 0.08 m
    cfg.data.deskew = False                       # no per-point timestamps
    cfg.mapping.voxel_size = 0.05                 # 5 cm voxels for indoor
    cfg.mapping.max_points_per_voxel = 20
    cfg.adaptive_threshold.initial_threshold = 1.0
    cfg.adaptive_threshold.min_motion_th = 0.01   # very small motions expected
    return cfg


# =====================================================================
#  Colour helpers
# =====================================================================

def _turbo_lut() -> np.ndarray:
    """256×3 float64 Turbo colormap LUT via cv2 (BGR→RGB)."""
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[0]
    return bgr[:, ::-1].astype(np.float64) / 255.0

_TURBO = _turbo_lut()


def height_refl_color(pts: np.ndarray, refl: np.ndarray,
                      z_lo: float = -1.0, z_hi: float = 2.5) -> np.ndarray:
    """
    Map height → Turbo hue, reflectance → brightness.
    Returns (N, 3) float64 in [0, 1].
    """
    z = pts[:, 2] if pts.shape[0] > 0 else np.empty(0)
    t = np.clip((z - z_lo) / max(z_hi - z_lo, 1e-6), 0.0, 1.0)
    idx = (t * 255).astype(np.int32)
    base_rgb = _TURBO[idx]                           # (N, 3) from turbo

    brightness = 0.35 + 0.65 * np.clip(refl, 0.0, 1.0)  # dim floor, bright walls
    return base_rgb * brightness[:, None]


def build_pointcloud(pts_m: np.ndarray, refl: np.ndarray,
                     ceil_z: float, floor_z: float,
                     clip_floor: bool) -> o3d.geometry.PointCloud | None:
    """Build a clipped, coloured Open3D point cloud."""
    if pts_m.shape[0] < 3:
        return None
    mask = pts_m[:, 2] <= ceil_z
    if clip_floor:
        mask &= pts_m[:, 2] >= floor_z
    p = pts_m[mask]
    r = refl[mask]
    if p.shape[0] < 3:
        return None
    z_lo = float(np.percentile(p[:, 2], 2))
    z_hi = float(np.percentile(p[:, 2], 98))
    colors = height_refl_color(p, r, z_lo, z_hi)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# =====================================================================
#  Open3D scene helpers
# =====================================================================

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


def make_camera_frustum(pose: np.ndarray,
                        length_m: float = 0.3) -> o3d.geometry.LineSet:
    """Draw a camera frustum at the given 4x4 pose."""
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
    R = pose[:3, :3]
    t = pose[:3, 3]
    all_global = (all_local @ R.T) + t

    line_idx = [[0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],
                [0, 5]]
    colors = [[0.0, 0.9, 0.9]] * 8 + [[1.0, 0.2, 0.2]]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_global)
    ls.lines = o3d.utility.Vector2iVector(line_idx)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# =====================================================================
#  Main
# =====================================================================

def _draw_sensor_view(fr, view_idx: int, slam_idx: int, n_frames: int,
                      pose: np.ndarray) -> np.ndarray:
    resize_wh = (400, 300)
    dist_view, peak_view = build_views(fr.dist, fr.conf, fr.peak, resize_wh)
    canvas = np.hstack([dist_view, peak_view])

    bar_h = 48
    bar = np.zeros((bar_h, canvas.shape[1], 3), np.uint8)
    tx, ty, tz = pose[:3, 3]
    line1 = (f"view {view_idx + 1}/{n_frames}  |  "
             f"SLAM at {slam_idx + 1}/{n_frames}  |  "
             f"pos ({tx:+.2f}, {ty:+.2f}, {tz:+.2f})m")
    line2 = "D=step A=back Space=play C/V=ceil F=floor +/-=ptsize Q=quit"
    cv2.putText(bar, line1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 180, 180), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


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
    print("[INFO] Right/D = step   Left/A = back   Space = play/pause")
    print("[INFO] C/V = ceiling clip   F = floor clip   +/- = point size   Q = quit")

    # ── KISS-ICP odometry ──────────────────────────────────────────────
    kiss_cfg = make_kiss_config()
    odometry = KissICP(config=kiss_cfg)

    pts_chunks: list[np.ndarray] = []
    refl_chunks: list[np.ndarray] = []
    all_pts_m = np.empty((0, 3), np.float64)
    all_refl = np.empty((0,), np.float64)
    poses: list[np.ndarray] = [np.eye(4)]
    total_dist_m = 0.0
    ceil_z = CEIL_CLIP_DEFAULT
    floor_z = FLOOR_CLIP_DEFAULT
    clip_floor = False

    # seed with first few valid frames so auto-fit has something to work with
    slam_idx = 0
    for si in range(min(N, 5)):
        cl_xyz, cl_refl = make_cloud_3d(frames[si].dist, frames[si].conf, frames[si].peak)
        if cl_xyz.shape[0] < 3:
            if si > 0:
                poses.append(odometry.last_pose.copy())
            continue
        odometry.register_frame(cl_xyz, timestamps=np.zeros(cl_xyz.shape[0]))
        cur = odometry.last_pose.copy()
        if si == 0:
            poses[0] = cur
        else:
            poses.append(cur)
        global_pts = (cl_xyz @ cur[:3, :3].T) + cur[:3, 3]
        pts_chunks.append(global_pts)
        refl_chunks.append(cl_refl)
        slam_idx = si

    view_idx = slam_idx

    # ── Open3D ────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D ToF SLAM (KISS-ICP)", 1280, 800)
    vis.add_geometry(make_ground_grid(extent_m=6.0, step_m=1.0))
    vis.add_geometry(make_axes(0.5))

    traj_ls = o3d.geometry.LineSet()
    vis.add_geometry(traj_ls)

    # seed the map point cloud so reset_view_point can auto-fit
    map_pcd = o3d.geometry.PointCloud()
    if pts_chunks:
        init_pts = np.vstack(pts_chunks)
        init_refl = np.concatenate(refl_chunks)
        init_pcd = build_pointcloud(init_pts, init_refl, ceil_z, floor_z, clip_floor)
        if init_pcd is not None:
            map_pcd.points = init_pcd.points
            map_pcd.colors = init_pcd.colors
    vis.add_geometry(map_pcd)
    cam_frustum: o3d.geometry.LineSet | None = None

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.point_size = POINT_SIZE
    opt.line_width = 3.0

    # auto-fit camera to the actual point cloud, then tilt to a 3/4 view
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_front([-0.30, -0.20, 0.55])
    ctr.set_up([0, 0, 1])
    ctr.change_field_of_view(step=5.0)            # widen FOV for headroom

    cv2.namedWindow("Sensor", cv2.WINDOW_AUTOSIZE)

    need_3d_refresh = True
    playing = False
    window_open = True

    # ── helper: advance SLAM by one frame ─────────────────────────────
    def slam_step() -> bool:
        nonlocal slam_idx, total_dist_m, need_3d_refresh
        if slam_idx >= N - 1:
            return False
        slam_idx += 1
        fr = frames[slam_idx]
        cl_xyz, cl_refl = make_cloud_3d(fr.dist, fr.conf, fr.peak)

        if cl_xyz.shape[0] < 3:
            poses.append(odometry.last_pose.copy())
            return True

        prev_pos = odometry.last_pose[:3, 3].copy()
        odometry.register_frame(cl_xyz, timestamps=np.zeros(cl_xyz.shape[0]))
        cur_pose = odometry.last_pose.copy()
        poses.append(cur_pose)

        step_m = np.linalg.norm(cur_pose[:3, 3] - prev_pos)
        total_dist_m += step_m

        global_pts = (cl_xyz @ cur_pose[:3, :3].T) + cur_pose[:3, 3]
        pts_chunks.append(global_pts)
        refl_chunks.append(cl_refl)

        if slam_idx % COMPACT_EVERY == 0 and pts_chunks:
            merged = np.vstack(pts_chunks)
            merged_r = np.concatenate(refl_chunks)
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(merged)
            r_col = np.clip(merged_r, 0, 1)
            pcd_tmp.colors = o3d.utility.Vector3dVector(
                np.column_stack((r_col, r_col, r_col)))
            pcd_tmp = pcd_tmp.voxel_down_sample(DOWNSAMPLE_VOXEL)
            ds_pts = np.asarray(pcd_tmp.points).copy()
            ds_refl = np.asarray(pcd_tmp.colors)[:, 0].copy()
            pts_chunks.clear()
            refl_chunks.clear()
            pts_chunks.append(ds_pts)
            refl_chunks.append(ds_refl)

        if slam_idx % 50 == 0 or slam_idx < 10:
            tx, ty, tz = cur_pose[:3, 3]
            print(f"\n  [{slam_idx:4d}] pos=({tx:+.2f}, {ty:+.2f}, {tz:+.2f})m  "
                  f"step={step_m:.3f}m")

        need_3d_refresh = True
        return True

    last_bbox_frame = -1

    # ── helper: refresh 3D display ────────────────────────────────────
    def refresh_3d(force: bool = False):
        nonlocal cam_frustum, all_pts_m, all_refl, need_3d_refresh, last_bbox_frame
        if not need_3d_refresh and not force:
            return
        need_3d_refresh = False

        if pts_chunks:
            all_pts_m = np.vstack(pts_chunks)
            all_refl = np.concatenate(refl_chunks)

        new_pcd = build_pointcloud(all_pts_m, all_refl, ceil_z, floor_z, clip_floor)
        if new_pcd is not None:
            map_pcd.points = new_pcd.points
            map_pcd.colors = new_pcd.colors

            # periodically re-sync the far/near clip planes as the map grows
            if slam_idx - last_bbox_frame >= BBOX_UPDATE_EVERY:
                last_bbox_frame = slam_idx
                cam_param = ctr.convert_to_pinhole_camera_parameters()
                vis.remove_geometry(map_pcd, reset_bounding_box=False)
                vis.add_geometry(map_pcd, reset_bounding_box=True)
                ctr.convert_from_pinhole_camera_parameters(cam_param, True)
            else:
                vis.update_geometry(map_pcd)

        if len(poses) >= 2:
            traj_pts = np.array([p[:3, 3] for p in poses])
            ln = [[i, i + 1] for i in range(len(traj_pts) - 1)]
            traj_ls.points = o3d.utility.Vector3dVector(traj_pts)
            traj_ls.lines = o3d.utility.Vector2iVector(ln)
            traj_ls.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * len(ln))
            vis.update_geometry(traj_ls)

        if cam_frustum is not None:
            vis.remove_geometry(cam_frustum, reset_bounding_box=False)
        cam_frustum = make_camera_frustum(odometry.last_pose, 0.25)
        vis.add_geometry(cam_frustum, reset_bounding_box=False)

        n_vis = np.asarray(map_pcd.points).shape[0]
        print(f"\r  frame {slam_idx + 1}/{N} | {n_vis} pts"
              f" | travel {total_dist_m:.2f}m   ",
              end="", flush=True)

    # ── main event loop ───────────────────────────────────────────────
    refresh_3d()

    while True:
        sensor_img = _draw_sensor_view(
            frames[view_idx], view_idx, slam_idx, N,
            odometry.last_pose)
        cv2.imshow("Sensor", sensor_img)

        if not vis.poll_events():
            window_open = False
            break
        vis.update_renderer()

        if playing and slam_idx < N - 1:
            slam_step()
            view_idx = slam_idx
            refresh_3d()

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
        elif key in (ord("c"), ord("C")):
            ceil_z -= CEIL_STEP
            print(f"\n  ceiling clip → {ceil_z:+.1f}m")
            refresh_3d(force=True)
        elif key in (ord("v"), ord("V")):
            ceil_z += CEIL_STEP
            print(f"\n  ceiling clip → {ceil_z:+.1f}m")
            refresh_3d(force=True)
        elif key in (ord("f"), ord("F")):
            clip_floor = not clip_floor
            print(f"\n  floor clip {'ON' if clip_floor else 'OFF'} ({floor_z:+.1f}m)")
            refresh_3d(force=True)
        elif key in (ord("+"), ord("=")):
            opt.point_size = min(opt.point_size + 1.0, 20.0)
            print(f"\n  point size → {opt.point_size:.0f}")
        elif key in (ord("-"), ord("_")):
            opt.point_size = max(opt.point_size - 1.0, 1.0)
            print(f"\n  point size → {opt.point_size:.0f}")

    print()
    cv2.destroyAllWindows()
    if window_open:
        print("[INFO] close 3D window to exit")
        vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
