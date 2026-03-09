from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _fourcc() -> int:
    # mp4v is broadly available
    return cv2.VideoWriter_fourcc(*"mp4v")


@dataclass
class VideoWriters:
    rgb_path: Optional[Path]
    depth_overlay_path: Optional[Path]
    fps: int
    frame_size: Tuple[int, int]  # (w, h)

    _rgb_writer: Optional[cv2.VideoWriter] = None
    _depth_writer: Optional[cv2.VideoWriter] = None
    _last_rgb_write: float = 0.0
    _last_depth_write: float = 0.0

    def open(self) -> None:
        w, h = self.frame_size
        if self.rgb_path is not None:
            self._rgb_writer = cv2.VideoWriter(str(self.rgb_path), _fourcc(), self.fps, (w, h))
        if self.depth_overlay_path is not None:
            self._depth_writer = cv2.VideoWriter(
                str(self.depth_overlay_path), _fourcc(), self.fps, (w, h)
            )

    def close(self) -> None:
        if self._rgb_writer is not None:
            self._rgb_writer.release()
        if self._depth_writer is not None:
            self._depth_writer.release()

    def maybe_write_rgb(self, frame_bgr: np.ndarray) -> bool:
        if self._rgb_writer is None:
            return False
        now = time.monotonic()
        if now - self._last_rgb_write < (1.0 / float(self.fps)):
            return False
        self._last_rgb_write = now
        self._rgb_writer.write(frame_bgr)
        return True

    def maybe_write_depth_overlay(self, overlay_bgr: np.ndarray) -> bool:
        if self._depth_writer is None:
            return False
        now = time.monotonic()
        if now - self._last_depth_write < (1.0 / float(self.fps)):
            return False
        self._last_depth_write = now
        self._depth_writer.write(overlay_bgr)
        return True


def depth_to_colormap(
    depth_mm: np.ndarray,
    *,
    min_mm: int,
    max_mm: int,
    colormap_name: str = "JET",
) -> np.ndarray:
    # depth_mm is uint16 (mm)
    depth = depth_mm.astype(np.float32)
    depth = np.clip(depth, float(min_mm), float(max_mm))
    norm = (depth - float(min_mm)) / max(1.0, float(max_mm - min_mm))
    norm_u8 = (norm * 255.0).astype(np.uint8)
    cmap = getattr(cv2, f"COLORMAP_{colormap_name}", cv2.COLORMAP_JET)
    return cv2.applyColorMap(norm_u8, cmap)


def blend_overlay(rgb_bgr: np.ndarray, depth_color_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, depth_color_bgr, alpha, 0.0)

