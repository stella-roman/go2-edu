from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

from .ring_buffer import FramePacket, RingBuffer


@dataclass(frozen=True)
class SelectionConfig:
    max_images_per_turn: int
    recent_window_sec: float
    recent_max_images: int
    keyframe_window_sec: float
    keyframe_max_images: int


class KeyframeStore:
    def __init__(self, *, max_pinned_keyframes: int):
        self._max = int(max_pinned_keyframes)
        self._items: List[FramePacket] = []

    def add(self, pkt: FramePacket) -> None:
        self._items.append(pkt)
        if len(self._items) > self._max:
            self._items = self._items[-self._max :]

    def within(self, window_sec: float, now: float) -> List[FramePacket]:
        cutoff = now - float(window_sec)
        return [p for p in self._items if p.ts >= cutoff]


def _uniform_sample(items: List[FramePacket], k: int) -> List[FramePacket]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return items
    # Uniformly sample by index
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    out = []
    last = None
    for idx in idxs:
        if last == idx:
            continue
        out.append(items[int(idx)])
        last = idx
    return out[:k]


class FrameSelector:
    def __init__(self, *, ring: RingBuffer, keyframes: KeyframeStore, cfg: SelectionConfig):
        self.ring = ring
        self.keyframes = keyframes
        self.cfg = cfg

    def select(self) -> Tuple[List[FramePacket], List[str]]:
        now = time.time()

        recent = self.ring.within(self.cfg.recent_window_sec, now=now)
        recent_sel = _uniform_sample(recent, self.cfg.recent_max_images)

        kfs = self.keyframes.within(self.cfg.keyframe_window_sec, now=now)
        # Prefer most recent keyframes
        kfs_sorted = sorted(kfs, key=lambda p: p.ts, reverse=True)
        kf_sel = list(reversed(kfs_sorted[: self.cfg.keyframe_max_images]))  # keep chronological-ish

        combined: List[FramePacket] = []
        seen = set()
        for pkt in recent_sel + kf_sel:
            if pkt.frame_id in seen:
                continue
            combined.append(pkt)
            seen.add(pkt.frame_id)

        if len(combined) > self.cfg.max_images_per_turn:
            combined = combined[-self.cfg.max_images_per_turn :]

        return combined, [p.frame_id for p in combined]

