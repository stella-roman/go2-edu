from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class KeyframeConfig:
    downsample_size: int
    scene_change_threshold: float
    min_keyframe_interval_sec: float


class SceneChangeKeyframeDetector:
    """
    Lightweight scene-change detector using mean absolute difference on downsampled grayscale frames.
    """

    def __init__(self, cfg: KeyframeConfig):
        self.cfg = cfg
        self._prev_small: Optional[np.ndarray] = None
        self._last_keyframe_ts: float = 0.0

    def score(self, frame_bgr: np.ndarray) -> float:
        sz = int(self.cfg.downsample_size)
        small = cv2.resize(frame_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_small is None:
            self._prev_small = gray
            return 0.0
        diff = np.mean(np.abs(gray - self._prev_small))
        self._prev_small = gray
        return float(diff)

    def is_keyframe(self, frame_bgr: np.ndarray, now: Optional[float] = None) -> Tuple[bool, float]:
        now = time.time() if now is None else float(now)
        s = self.score(frame_bgr)
        if s < float(self.cfg.scene_change_threshold):
            return False, s
        if (now - self._last_keyframe_ts) < float(self.cfg.min_keyframe_interval_sec):
            return False, s
        self._last_keyframe_ts = now
        return True, s

