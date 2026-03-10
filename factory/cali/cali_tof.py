#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cali_tof.py

标定参数：
- 焦距：fx = fy = f
- 光心：cx, cy
- ToF 偏置：bias（实际距离 = 测量距离 - bias）
- 平面参数：两个角度 ax, ay（用于表示平面法向量）

已知条件：
- 平面到镜头坐标系原点的距离固定为 1.01 米
- 输入默认是 20x20 深度图（即 400 点）

优化目标：
- 400 个点到该平面的距离误差 RMS 最小

优化算法：
1) differential_evolution 全局优化
2) least_squares 局部鲁棒精修（soft_l1）
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

IMG_W = 20
IMG_H = 20

PLANE_DISTANCE_M = 1.01


@dataclass(frozen=True)
class CalibResult:
    rms_m: float
    n_total: int
    n_valid: int
    depth_model: str
    plane_distance_m: float
    f: float
    cx: float
    cy: float
    bias: float
    ax: float
    ay: float
    plane_normal: list[float]
    optimizer_global: str
    optimizer_local: str


def _build_roi_uv() -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(0, IMG_W, dtype=np.float64)
    ys = np.arange(0, IMG_H, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    return u.reshape(-1), v.reshape(-1)


def _plane_normal_from_angles(ax: float, ay: float) -> np.ndarray:
    # 使用两个角度参数控制倾斜量，nz 固定正向，再归一化为单位法向量。
    n = np.array([math.tan(ax), math.tan(ay), 1.0], dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / n_norm


def _points_from_depth(
    depth_flat_m: np.ndarray,
    u_flat: np.ndarray,
    v_flat: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    bias: float,
    depth_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    d = depth_flat_m - float(bias)
    valid = np.isfinite(d) & (d > 0.02)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), valid

    uu = u_flat[valid]
    vv = v_flat[valid]
    dd = d[valid]

    x = (uu - float(cx)) / float(f)
    y = (vv - float(cy)) / float(f)

    if depth_model == "z":
        z = dd
        pts = np.stack([x * z, y * z, z], axis=1)
    else:
        dirs = np.stack([x, y, np.ones_like(x)], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = dirs * dd[:, None]
    return pts, valid


def _residuals(
    params: np.ndarray,
    depth_flat_m: np.ndarray,
    u_flat: np.ndarray,
    v_flat: np.ndarray,
    cx_fixed: float,
    cy_fixed: float,
    depth_model: str,
    plane_distance_m: float,
) -> np.ndarray:
    f, bias, ax, ay = [float(v) for v in params]
    pts, valid = _points_from_depth(depth_flat_m, u_flat, v_flat, f, cx_fixed, cy_fixed, bias, depth_model)
    if pts.shape[0] < 50:
        # 可用点太少时返回大残差，抑制无效解。
        return np.full((max(depth_flat_m.size, 1),), 10.0, dtype=np.float64)

    n = _plane_normal_from_angles(ax, ay)
    dist = pts @ n - float(plane_distance_m)

    out = np.full(depth_flat_m.shape, 0.0, dtype=np.float64)
    out[valid] = dist
    out[~valid] = 2.0
    return out


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(x))))


def _load_depth_map(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (IMG_H, IMG_W):
        raise ValueError(f"expect depth map shape {(IMG_H, IMG_W)}, got {arr.shape}")
    return arr


def _make_plane_mesh(
    pts: np.ndarray,
    n: np.ndarray,
    d: float,
    scale_pad: float = 0.05,
    min_half_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = n * float(d)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = np.cross(n, helper)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(n, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1e-12)

    if pts.size > 0:
        rel = pts - p0[None, :]
        a = rel @ e1
        b = rel @ e2
        a_min, a_max = float(np.min(a)), float(np.max(a))
        b_min, b_max = float(np.min(b)), float(np.max(b))
        a_pad = max((a_max - a_min) * scale_pad, min_half_size)
        b_pad = max((b_max - b_min) * scale_pad, min_half_size)
        a_lin = np.linspace(a_min - a_pad, a_max + a_pad, 20)
        b_lin = np.linspace(b_min - b_pad, b_max + b_pad, 20)
    else:
        a_lin = np.linspace(-min_half_size, min_half_size, 20)
        b_lin = np.linspace(-min_half_size, min_half_size, 20)

    aa, bb = np.meshgrid(a_lin, b_lin)
    xyz = p0[None, None, :] + aa[..., None] * e1[None, None, :] + bb[..., None] * e2[None, None, :]
    return xyz[..., 0], xyz[..., 1], xyz[..., 2]


def _show_plot(
    points: np.ndarray,
    residuals: np.ndarray,
    normal: np.ndarray,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("missing dependency matplotlib, run: py -m pip install matplotlib") from e

    px, py, pz = _make_plane_mesh(points, normal, PLANE_DISTANCE_M)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=residuals, cmap="coolwarm", s=18, alpha=0.9)
    ax.plot_surface(px, py, pz, alpha=0.35, color="tab:green", linewidth=0, antialiased=True)
    ax.scatter([0.0], [0.0], [0.0], c="k", s=40, marker="x")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("ToF points and calibrated plane")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _show_error_hist3d(
    error_map: np.ndarray,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("missing dependency matplotlib, run: py -m pip install matplotlib") from e

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    x = uu.reshape(-1).astype(np.float64)
    y = vv.reshape(-1).astype(np.float64)
    z0 = np.zeros_like(x, dtype=np.float64)
    val_m = error_map.reshape(-1).astype(np.float64)
    valid = np.isfinite(val_m)

    x = x[valid]
    y = y[valid]
    z0 = z0[valid]
    val_m = val_m[valid]
    val_cm = val_m * 100.0

    dx = np.full_like(x, 0.8, dtype=np.float64)
    dy = np.full_like(y, 0.8, dtype=np.float64)
    dz = val_cm.copy()
    zbase = np.where(dz >= 0.0, 0.0, dz)
    dz = np.abs(dz)

    # 固定量程 +-5cm，用于颜色映射
    norm = np.clip((val_cm + 5.0) / 10.0, 0.0, 1.0)
    colors = plt.cm.seismic(norm)

    ax.bar3d(x - 0.4, y - 0.4, zbase, dx, dy, dz, color=colors, shade=True, zsort="average")

    mappable = plt.cm.ScalarMappable(cmap="seismic")
    mappable.set_clim(-5.0, 5.0)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label("signed error (cm), fixed range [-5, +5]")

    ax.set_title("Per-pixel plane fitting error (3D histogram, 20x20)")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("error (cm)")
    ax.set_xlim(-0.5, IMG_W - 0.5)
    ax.set_ylim(-0.5, IMG_H - 0.5)
    ax.set_zlim(-5.0, 5.0)
    ax.view_init(elev=28, azim=-50)

    # 在图上显示最大/最小误差，便于快速判断。
    if val_cm.size > 0:
        vmax = float(np.max(val_cm))
        vmin = float(np.min(val_cm))
        fig.text(0.02, 0.02, f"min={vmin:+.3f} cm, max={vmax:+.3f} cm", fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _show_error_distribution_hist(
    residuals_m: np.ndarray,
    *,
    show: bool,
    save_path: Path | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("missing dependency matplotlib, run: py -m pip install matplotlib") from e

    errs_cm = np.asarray(residuals_m, dtype=np.float64).reshape(-1) * 100.0
    errs_cm = errs_cm[np.isfinite(errs_cm)]
    if errs_cm.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errs_cm, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_title("Error distribution of 400 points")
    ax.set_xlabel("signed error (cm)")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25, linestyle="--")

    mean_cm = float(np.mean(errs_cm))
    std_cm = float(np.std(errs_cm))
    rms_cm = float(np.sqrt(np.mean(errs_cm * errs_cm)))

    ax.axvline(mean_cm, color="tab:red", linewidth=1.5, label=f"mean={mean_cm:+.3f} cm")
    ax.axvline(0.0, color="tab:green", linewidth=1.2, linestyle="--", label="zero")
    ax.legend(loc="upper right")
    fig.text(0.02, 0.02, f"n={errs_cm.size}, std={std_cm:.3f} cm, rms={rms_cm:.3f} cm", fontsize=10)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="ToF 平面标定：优化 f, bias 和平面两个角度参数（cx/cy 固定中心）")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data.npy"),
        help="输入深度图 .npy，默认 data.npy（20x20）",
    )
    parser.add_argument(
        "--depth-model",
        choices=["ray", "z"],
        default="ray",
        help="depth 含义：ray=沿光线距离，z=相机z方向深度",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cali_tof_result.json"),
        help="输出 json 路径",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="可选：保存点云+平面图到文件（默认不保存）",
    )
    parser.add_argument(
        "--save-error-hist",
        "--save-error-matrix",
        type=Path,
        default=None,
        help="可选：保存误差3D直方图到文件（默认不保存）",
    )
    parser.add_argument(
        "--save-error-dist",
        type=Path,
        default=None,
        help="可选：保存400点误差统计直方图到文件（默认不保存）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出交互图窗（默认会弹窗）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    try:
        from scipy.optimize import differential_evolution, least_squares
    except Exception as e:
        raise RuntimeError("missing dependency scipy, run: py -m pip install scipy") from e

    depth_map = _load_depth_map(args.data)
    depth_flat = depth_map.reshape(-1)
    u_flat, v_flat = _build_roi_uv()

    cx0 = (IMG_W - 1) / 2.0
    cy0 = (IMG_H - 1) / 2.0

    # 参数顺序: [f, bias, ax, ay]，cx/cy 固定在图像中心。
    x0 = np.array(
        [
            27.5,  # f
            1.0,   # bias (按你的要求，初值为 1)
            0.0,   # ax
            0.0,   # ay
        ],
        dtype=np.float64,
    )

    bounds = [
        (5.0, 120.0),     # f
        (-0.5, 2.5),      # bias
        (-0.7, 0.7),      # ax
        (-0.7, 0.7),      # ay
    ]

    def objective_rms(p: np.ndarray) -> float:
        r = _residuals(p, depth_flat, u_flat, v_flat, cx0, cy0, args.depth_model, PLANE_DISTANCE_M)
        return _rms(r)

    de_res = differential_evolution(
        objective_rms,
        bounds=bounds,
        seed=int(args.seed),
        strategy="best1bin",
        popsize=24,
        maxiter=400,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.8,
        polish=False,
        updating="deferred",
        workers=1,
    )

    ls_res = least_squares(
        fun=lambda p: _residuals(p, depth_flat, u_flat, v_flat, cx0, cy0, args.depth_model, PLANE_DISTANCE_M),
        x0=np.asarray(de_res.x, dtype=np.float64),
        bounds=(
            np.array([b[0] for b in bounds], dtype=np.float64),
            np.array([b[1] for b in bounds], dtype=np.float64),
        ),
        method="trf",
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=3000,
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        verbose=0,
    )

    x_opt = np.asarray(ls_res.x, dtype=np.float64)
    r_opt = _residuals(x_opt, depth_flat, u_flat, v_flat, cx0, cy0, args.depth_model, PLANE_DISTANCE_M)
    rms_m = _rms(r_opt)

    f, bias, ax, ay = [float(v) for v in x_opt]
    cx = float(cx0)
    cy = float(cy0)
    ax_deg = math.degrees(ax)
    ay_deg = math.degrees(ay)
    normal = _plane_normal_from_angles(ax, ay)

    pts_opt, valid_mask = _points_from_depth(depth_flat, u_flat, v_flat, f, cx, cy, bias, args.depth_model)
    n_valid = int(np.sum(valid_mask))
    n_total = int(depth_flat.size)
    residual_valid = r_opt[valid_mask]
    error_map = np.full((IMG_H, IMG_W), np.nan, dtype=np.float64)
    error_map.reshape(-1)[valid_mask] = residual_valid

    res = CalibResult(
        rms_m=float(rms_m),
        n_total=n_total,
        n_valid=n_valid,
        depth_model=str(args.depth_model),
        plane_distance_m=float(PLANE_DISTANCE_M),
        f=f,
        cx=cx,
        cy=cy,
        bias=bias,
        ax=ax,
        ay=ay,
        plane_normal=normal.astype(float).tolist(),
        optimizer_global="scipy.optimize.differential_evolution",
        optimizer_local="scipy.optimize.least_squares(loss=soft_l1)",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    print("=== cali_tof result ===")
    print(f"data           : {args.data}")
    print(f"depth_model    : {args.depth_model}")
    print(f"points(valid)  : {n_valid}/{n_total}")
    print(f"rms            : {rms_m:.6f} m")
    print(f"f              : {f:.6f} px")
    print(f"cx, cy         : ({cx:.6f}, {cy:.6f}) px (fixed center)")
    print(f"bias           : {bias:.6f} m")
    print(f"ax, ay         : ({ax_deg:.6f}°, {ay_deg:.6f}°)")
    print(f"plane normal   : [{normal[0]:+.8f}, {normal[1]:+.8f}, {normal[2]:+.8f}]")
    print(f"plane distance : {PLANE_DISTANCE_M:.6f} m (fixed)")
    print(f"saved          : {args.out}")
    if args.save_plot is not None:
        print(f"saved          : {args.save_plot}")
    if not bool(args.no_show):
        print("plot           : interactive window shown")
    else:
        print("plot           : disabled by --no-show")

    _show_error_hist3d(
        error_map,
        show=(not bool(args.no_show)),
        save_path=args.save_error_hist,
    )
    _show_error_distribution_hist(
        residual_valid,
        show=(not bool(args.no_show)),
        save_path=args.save_error_dist,
    )
    _show_plot(
        pts_opt,
        residual_valid,
        normal,
        show=(not bool(args.no_show)),
        save_path=args.save_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


