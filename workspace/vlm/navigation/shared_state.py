from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DepthROISnapshot:
    ts: float
    left_cm: Optional[float]
    center_cm: Optional[float]
    right_cm: Optional[float]
    valid_ratio: float
    width: int
    height: int


class DepthSharedState:
    """
    카메라 스레드(작성) ↔ 메인/가이던스 스레드(읽기) 간에
    depth 요약(ROI 거리)을 안전하게 공유하기 위한 상태 객체.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[DepthROISnapshot] = None

    def update(self, snap: DepthROISnapshot) -> None:
        with self._lock:
            self._latest = snap

    def latest(self) -> Optional[DepthROISnapshot]:
        with self._lock:
            return self._latest

    @staticmethod
    def now_ts() -> float:
        return time.time()

