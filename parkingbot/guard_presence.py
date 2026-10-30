"""Is the guarded car still in its zone? (substream, no YOLO.)

An edge signature of the car zone is taken when GUARD arms and compared every few
seconds (edge NCC with a small slide tolerance, like the stability gate). A car that
drives away changes the zone completely (low similarity); a parked car stays similar.
Several consecutive low checks raise a *suspicion*; the engine then confirms with
main-stream reads before anything is announced, so exposure shifts, an IR switch-over
or a person standing in front never raise CAR_GONE on their own.
"""

from __future__ import annotations

import cv2
import numpy as np

from .guard_geometry import Box


def signature(frame: np.ndarray, box: Box, width: int = 64) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h = max(8, int(round(gray.shape[0] * width / gray.shape[1])))
    gray = cv2.resize(gray, (width, h), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.Canny(gray, 50, 150)


def similarity(ref: np.ndarray, cur: np.ndarray, pad: int = 4) -> float:
    if ref.shape != cur.shape:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    if not ref.any() or not cur.any():
        return 0.0
    padded = cv2.copyMakeBorder(cur, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    return float(cv2.matchTemplate(padded, ref, cv2.TM_CCOEFF_NORMED).max())


class PresenceMonitor:
    def __init__(self, box: Box, check_s: float = 5.0, low_sim: float = 0.30,
                 low_checks: int = 3, refresh_s: float = 60.0, high_sim: float = 0.60):
        self.box = box
        self.check_s = check_s
        self.low_sim = low_sim
        self.low_checks = low_checks
        self.refresh_s = refresh_s
        self.high_sim = high_sim
        self.ref: np.ndarray | None = None
        self.sim: float | None = None
        self.low_since: float | None = None
        self._low = 0
        self._last_check = 0.0
        self._last_refresh = 0.0

    def reseed(self, frame: np.ndarray, ts: float) -> None:
        self.ref = signature(frame, self.box)
        self._last_refresh = ts
        self._low, self.low_since = 0, None

    def update(self, frame: np.ndarray, ts: float) -> bool:
        """Call for every frame; it only does work every ``check_s``. Returns True the
        moment ``low_checks`` consecutive low-similarity checks have accumulated."""
        if self.ref is None:
            self.reseed(frame, ts)
            return False
        if ts - self._last_check < self.check_s:
            return False
        self._last_check = ts
        cur = signature(frame, self.box)
        if cur is None:
            return False
        self.sim = similarity(self.ref, cur)
        if self.sim >= self.low_sim:
            self._low, self.low_since = 0, None
            if self.sim >= self.high_sim and ts - self._last_refresh >= self.refresh_s:
                self.ref, self._last_refresh = cur, ts          # follow slow lighting drift
            return False
        if self._low == 0:
            self.low_since = ts
        self._low += 1
        return self._low == self.low_checks

    @property
    def is_low(self) -> bool:
        return self._low >= self.low_checks
