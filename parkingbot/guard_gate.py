"""GUARD stage-1 gates: a cheap in-host motion gate and the XVR event gate.

Either gate only says "something may be happening"; YOLO on the crop still verifies
and the tracker/rules decide severity. The gates are OR-ed (the XVR event delay and
night reliability were not measured in Phase 0, so neither is trusted alone).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    motion: bool
    global_change: bool
    frac: float


class MotionGate:
    """Perimeter crop -> ~160x90 gray -> diff against a running-average background.

    - more than ``global_frac`` of the crop changing at once (headlight sweep, auto
      exposure, IR switch-over) is rejected and the background re-seeded;
    - a 3x3 opening removes 1-2 px speckle (insects/rain near the IR LEDs);
    - the background adapts slowly while active (``alpha_active``) so a person standing
      still is not absorbed before the tracker has seen them, yet a permanent scene
      change cannot keep the gate firing forever.
    """

    def __init__(self, perimeter_bbox, down=(160, 90), diff_thr: int = 25,
                 min_frac: float = 0.004, global_frac: float = 0.40, alpha: float = 0.05,
                 alpha_active: float = 0.01):
        self.bbox = tuple(int(round(v)) for v in perimeter_bbox)
        self.down = down
        self.diff_thr = diff_thr
        self.min_frac = min_frac
        self.global_frac = global_frac
        self.alpha = alpha
        self.alpha_active = alpha_active
        self.scale = 1.0           # >1 raises the threshold (duty limiter)
        self._bg: np.ndarray | None = None
        self._kernel = np.ones((3, 3), np.uint8)

    def update(self, frame: np.ndarray, active: bool = False) -> GateResult:
        x1, y1, x2, y2 = self.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return GateResult(False, False, 0.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        small = cv2.resize(gray, self.down, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._bg is None:
            self._bg = small
            return GateResult(False, False, 0.0)
        mask = (np.abs(small - self._bg) > self.diff_thr).astype(np.uint8)
        frac_raw = float(mask.mean())
        if frac_raw > self.global_frac:
            self._bg = small                  # exposure/IR change: re-seed, ignore
            return GateResult(False, True, frac_raw)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        frac = float(mask.mean())
        a = self.alpha_active if active else self.alpha
        self._bg = (1 - a) * self._bg + a * small
        return GateResult(frac > self.min_frac * self.scale, False, frac)


# ----------------------------------------------------------------- XVR events
RELEVANT_CODES = {"SmartMotionHuman", "SmartMotionVehicle", "VideoMotion"}   # all that the XVR emits for us
_EVT = re.compile(r"Code=(\w+);action=(\w+);index=(\d+)")


def parse_events(buf: str) -> tuple[list[tuple[str, str, int]], str]:
    """Pull (code, action, index) tuples out of the multipart event stream. Returns
    them plus the unparsed tail to carry into the next chunk."""
    events, last = [], 0
    for m in _EVT.finditer(buf):
        events.append((m.group(1), m.group(2), int(m.group(3))))
        last = m.end()
    return events, buf[last:][-400:]


class XvrEventGate:
    """Listens to ``eventManager.cgi?action=attach`` (Dahua, digest auth) and records
    when an event for our channel last arrived. Reconnects with backoff."""

    def __init__(self, nvr_ip: str, user: str, password: str, channel: int,
                 clock=None, backoff_max: float = 60.0, codes=("SmartMotionHuman",)):
        import time
        self._url = f"http://{nvr_ip}/cgi-bin/eventManager.cgi?action=attach&codes=[All]"
        self._user, self._pass = user, password
        self.index = channel - 1                 # event index is the 0-based channel
        self.codes = set(codes)
        self._clock = clock or time.time
        self.backoff_max = backoff_max
        self.last_event_ts: float = 0.0
        self.last_code = ""
        self.count = 0
        self.connected = False
        self.reconnects = 0

    def feed(self, chunk: str, carry: str = "") -> str:
        events, rest = parse_events(carry + chunk)
        for code, _action, idx in events:
            if code in self.codes and idx == self.index:
                self.last_event_ts = self._clock()
                self.last_code = code
                self.count += 1
        return rest

    def hint(self, now: float, tail: float) -> bool:
        return self.last_event_ts > 0 and now - self.last_event_ts <= tail

    async def run(self) -> None:
        import httpx
        backoff = 1.0
        while True:
            try:
                auth = httpx.DigestAuth(self._user, self._pass)
                async with httpx.AsyncClient(timeout=httpx.Timeout(15, read=60)) as client:
                    async with client.stream("GET", self._url, auth=auth) as r:
                        r.raise_for_status()
                        self.connected, backoff, carry = True, 1.0, ""
                        log.info("xvr events: attached (channel index %d)", self.index)
                        async for chunk in r.aiter_text():
                            carry = self.feed(chunk, carry)
            except asyncio.CancelledError:
                self.connected = False
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                log.warning("xvr events: %s; reconnecting in %.0fs", type(exc).__name__, backoff)
            self.connected = False
            self.reconnects += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.backoff_max)
