from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass(frozen=True)
class FramePacket:
    frame_id: str
    ts: float
    jpeg_bytes: bytes
    width: int
    height: int


class RingBuffer:
    def __init__(self, *, max_seconds: float, max_frames: int):
        self._max_seconds = float(max_seconds)
        self._max_frames = int(max_frames)
        self._buf: Deque[FramePacket] = deque()
        self._lock = threading.Lock()

    def push(self, pkt: FramePacket) -> None:
        with self._lock:
            self._buf.append(pkt)
            self._trim_locked(now=pkt.ts)

    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def within(self, window_sec: float, now: Optional[float] = None) -> List[FramePacket]:
        now = time.time() if now is None else float(now)
        cutoff = now - float(window_sec)
        with self._lock:
            return [p for p in list(self._buf) if p.ts >= cutoff]

    def _trim_locked(self, now: float) -> None:
        cutoff = float(now) - self._max_seconds
        while self._buf and (len(self._buf) > self._max_frames or self._buf[0].ts < cutoff):
            self._buf.popleft()

