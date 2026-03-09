from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DepthFeatureConfig:
    # ROI fractions relative to full frame (x0,x1,y0,y1), in [0,1]
    left_roi: Tuple[float, float, float, float]
    center_roi: Tuple[float, float, float, float]
    right_roi: Tuple[float, float, float, float]
    min_mm: int
    max_mm: int
    statistic: str  # "p10" | "median"
    ema_alpha: float  # 0..1 (higher = smoother)
    min_valid_ratio: float  # if below, treat as invalid


def _clip_roi(roi: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x0, x1, y0, y1 = roi
    x0 = float(min(1.0, max(0.0, x0)))
    x1 = float(min(1.0, max(0.0, x1)))
    y0 = float(min(1.0, max(0.0, y0)))
    y1 = float(min(1.0, max(0.0, y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, x1, y0, y1


def _roi_slices(h: int, w: int, roi: Tuple[float, float, float, float]) -> Tuple[slice, slice]:
    x0, x1, y0, y1 = _clip_roi(roi)
    xs = slice(int(round(x0 * w)), max(int(round(x1 * w)), int(round(x0 * w)) + 1))
    ys = slice(int(round(y0 * h)), max(int(round(y1 * h)), int(round(y0 * h)) + 1))
    return ys, xs


def _stat_mm(valid_mm: np.ndarray, statistic: str) -> Optional[float]:
    if valid_mm.size == 0:
        return None
    if statistic == "median":
        return float(np.median(valid_mm))
    # default: p10 (closer-object sensitive, but robust to sparse outliers)
    return float(np.percentile(valid_mm, 10))


def depth_mm_to_roi_cm(
    depth_mm: np.ndarray,
    *,
    cfg: DepthFeatureConfig,
) -> Tuple[Dict[str, Optional[float]], float]:
    """
    depth_mm (H,W) uint16(mm)에서 좌/중/우 ROI의 거리(cm)를 계산.
    반환:
      - distances_cm: {"left":..., "center":..., "right":...} (None 가능)
      - valid_ratio: 전체 프레임에서 valid depth 비율
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"depth_mm must be 2D, got shape={depth_mm.shape}")

    h, w = depth_mm.shape[:2]
    min_mm = int(cfg.min_mm)
    max_mm = int(cfg.max_mm)

    valid = (depth_mm > 0) & (depth_mm >= min_mm) & (depth_mm <= max_mm)
    valid_ratio = float(valid.mean()) if depth_mm.size else 0.0

    out: Dict[str, Optional[float]] = {"left": None, "center": None, "right": None}
    if valid_ratio < float(cfg.min_valid_ratio):
        return out, valid_ratio

    for name, roi in (("left", cfg.left_roi), ("center", cfg.center_roi), ("right", cfg.right_roi)):
        ys, xs = _roi_slices(h, w, roi)
        roi_mm = depth_mm[ys, xs]
        roi_valid = valid[ys, xs]
        vals = roi_mm[roi_valid].astype(np.float32)
        mm = _stat_mm(vals, cfg.statistic)
        out[name] = None if mm is None else float(mm) / 10.0  # mm -> cm

    return out, valid_ratio


class DepthFeatureSmoother:
    def __init__(self, *, ema_alpha: float) -> None:
        self.alpha = float(ema_alpha)
        self._prev = {"left": None, "center": None, "right": None}

    def update(self, distances_cm: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        a = self.alpha
        for k in ("left", "center", "right"):
            v = distances_cm.get(k)
            p = self._prev.get(k)
            if v is None:
                out[k] = p
                continue
            if p is None:
                out[k] = float(v)
            else:
                out[k] = float(a * p + (1.0 - a) * float(v))
            self._prev[k] = out[k]
        return out

