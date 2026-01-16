#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tof_ball_detector.py

ToF ball detector module (pure detection, no visualization):
- Input: tof.raw
- Output: 2D pixel coord (centroid) in ToF image

Method:
- Find the brightest pixel by reflectivity (histogram sum), with peak threshold filtering.
- Compute a weighted centroid in a local window (window_size x window_size).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Allow running from cali/ directly: add project root to sys.path (to import tof3d).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tof3d import ToF3DParams, tof_histograms


@dataclass(frozen=True)
class ToFBallDetection2D:
    centroid_xy: Optional[Tuple[float, float]]  # (x,y) in ToF pixel coords


def detect_ball_tof_2d(
    tof_raw: str | Path,
    *,
    window_size: int = 5,
    min_peak: float = 100.0,
    valid_bins: int = 62,
) -> ToFBallDetection2D:
    """
    Return ToF 2D pixel coord (x,y).
    - centroid_xy is None if not detected.
    """
    p = Path(tof_raw)
    if not p.exists():
        return ToFBallDetection2D(centroid_xy=None)

    params = ToF3DParams(min_peak_count=float(min_peak))
    hists = tof_histograms(p, params=params).astype(np.float32, copy=False)  # (H,W,64)
    if hists.size == 0:
        return ToFBallDetection2D(centroid_xy=None)

    h, w, _ = hists.shape
    vb = int(np.clip(int(valid_bins), 1, hists.shape[2]))
    peak = hists[:, :, :vb].max(axis=2).astype(np.float32, copy=False)
    inten = hists.sum(axis=2).astype(np.float32, copy=False)
    inten = np.where(peak >= float(min_peak), inten, 0.0)
    if float(np.max(inten)) <= 0.0:
        return ToFBallDetection2D(centroid_xy=None)

    flat_idx = int(np.argmax(inten))
    y0, x0 = int(flat_idx // w), int(flat_idx % w)

    # window_size×window_size 强度加权重心
    r = max(0, int(window_size) // 2)
    xs = np.arange(max(0, x0 - r), min(w, x0 + r + 1), dtype=np.float32)
    ys = np.arange(max(0, y0 - r), min(h, y0 + r + 1), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    ww = inten[ys.astype(np.int32)[:, None], xs.astype(np.int32)[None, :]].astype(np.float32, copy=False)
    wsum = float(np.sum(ww))
    if wsum <= 0.0:
        cx, cy = float(x0), float(y0)
    else:
        cx = float(np.sum(xx * ww) / wsum)
        cy = float(np.sum(yy * ww) / wsum)

    return ToFBallDetection2D(centroid_xy=(cx, cy))


