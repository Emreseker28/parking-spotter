"""Small SORT-style tracker (greedy IoU + centroid matching, time-based ageing).

Detections arrive at an irregular 1-4 fps (the duty-cycle limiter changes the rate),
so ageing is in seconds, not frames. A track is "confirmed" after ``min_hits``
detections, which filters one-frame flicker (insects, IR noise).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .guard_geometry import Box, center, iou


@dataclass(frozen=True)
class Det:
    cls: int
    name: str
    score: float
    box: Box


@dataclass
class Track:
    id: int
    cls: int
    name: str
    box: Box
    first_ts: float
    last_ts: float
    hits: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=64))   # (ts, box)

    def dwell(self) -> float:
        return self.last_ts - self.first_ts


class Tracker:
    def __init__(self, iou_thr: float = 0.15, max_age: float = 3.0, min_hits: int = 2,
                 centroid_frac: float = 0.6):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.centroid_frac = centroid_frac
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def confirmed(self) -> list[Track]:
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def has_tracks(self) -> bool:
        return bool(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def _score(self, t: Track, d: Det) -> float:
        """Match quality in [0, 1]: IoU, else closeness of centres (fast movers)."""
        i = iou(t.box, d.box)
        if i >= self.iou_thr:
            return 0.5 + i / 2
        (tx, ty), (dx, dy) = center(t.box), center(d.box)
        size = max(t.box[2] - t.box[0], t.box[3] - t.box[1], 1.0)
        dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5 / size
        return (1.0 - dist / self.centroid_frac) * 0.5 if dist < self.centroid_frac else 0.0

    def update(self, dets: list[Det], ts: float) -> list[Track]:
        """Match this frame's detections to tracks; returns the confirmed tracks."""
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(dets):
                if t.cls != d.cls:
                    continue
                s = self._score(t, d)
                if s > 0:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self._tracks[ti], dets[di]
            t.box, t.last_ts, t.hits = d.box, ts, t.hits + 1
            t.history.append((ts, d.box))
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            t = Track(self._next_id, d.cls, d.name, d.box, ts, ts)
            t.history.append((ts, d.box))
            self._next_id += 1
            self._tracks.append(t)
        self._tracks = [t for t in self._tracks if ts - t.last_ts <= self.max_age]
        return self.confirmed()
