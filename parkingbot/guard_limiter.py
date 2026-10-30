"""YOLO duty-cycle limiter (load control).

Tracks the fraction of wall time the detector was busy over a rolling window. Above
``limit`` it degrades in steps -- raise the motion threshold, then lower the YOLO
rate (4 -> 2 -> 1 fps) -- and recovers one step after ``recover_after`` seconds of
calm. Rules stay active at every step; only the detection rate drops. Every step
change is returned so the caller can log it.
"""

from __future__ import annotations

from collections import deque

# step -> (max detections per second, motion-threshold scale)
STEPS = [(4.0, 1.0), (4.0, 1.5), (2.0, 1.5), (1.0, 1.5)]


class DutyLimiter:
    def __init__(self, window: float = 60.0, limit: float = 0.40,
                 recover_after: float = 300.0, base_fps: float = 4.0, eval_every: float = 5.0):
        self.window = window
        self.limit = limit
        self.recover_after = recover_after
        self.eval_every = eval_every
        self.base_fps = base_fps
        self.step = 0
        self._busy: deque[tuple[float, float]] = deque()   # (end_ts, busy_seconds)
        self._calm_since: float | None = None
        self._last_eval = 0.0
        self.changes = 0

    def record(self, busy_s: float, now: float) -> None:
        self._busy.append((now, busy_s))

    def duty(self, now: float) -> float:
        while self._busy and self._busy[0][0] < now - self.window:
            self._busy.popleft()
        return min(1.0, sum(b for _, b in self._busy) / self.window)

    @property
    def fps(self) -> float:
        return min(self.base_fps, STEPS[self.step][0])

    @property
    def gate_scale(self) -> float:
        return STEPS[self.step][1]

    def evaluate(self, now: float) -> str | None:
        """Re-evaluate (at most every ``eval_every`` s). Returns a log line when the
        step changed, else None."""
        if now - self._last_eval < self.eval_every:
            return None
        self._last_eval = now
        d = self.duty(now)
        if d > self.limit and self.step < len(STEPS) - 1:
            self.step += 1
            self.changes += 1
            self._calm_since = None
            return (f"limiter: duty {d:.0%} > {self.limit:.0%} -> step {self.step} "
                    f"({self.fps:g} fps, motion threshold x{self.gate_scale:g})")
        if d <= self.limit * 0.5 and self.step > 0:
            if self._calm_since is None:
                self._calm_since = now
            elif now - self._calm_since >= self.recover_after:
                self.step -= 1
                self.changes += 1
                self._calm_since = now
                return (f"limiter: calm, duty {d:.0%} -> step {self.step} "
                        f"({self.fps:g} fps, motion threshold x{self.gate_scale:g})")
        else:
            self._calm_since = None
        return None
