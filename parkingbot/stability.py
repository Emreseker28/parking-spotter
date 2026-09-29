"""Stability ("no motion") gate — plan Phase 2.

Targets the exact failure the plan exists for: a car is present but YOLOv8n misses
it (dark / top-down car on a grayscale IR night frame), so the spot reads FREE and
the bot cries "BOŞ". A car that *leaves* changes the ROI a lot; a car that stays
(but is missed) leaves the ROI essentially unchanged. This gate compares the
current ROI crop against the last confirmed-occupied reference crop and, when they
match, overrides the FREE verdict back to OCCUPIED (`stable_override`).

Cost: one crop + one matchTemplate per pending FREE verdict — essentially zero
CPU/RAM on the steady path (no model inference).

Comparison is edge-based normalized cross-correlation (Canny + matchTemplate
TM_CCOEFF_NORMED) on contrast-normalized (CLAHE), polygon-masked crops, so slow
lighting drift (dusk, IR switch) does not break it. The reference refreshes
whenever YOLO reports OCCUPIED with high coverage, and stale references (older
than max_ref_age_hours) are ignored.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .config import Settings, Spot
from .detection import SpotResult, roi_bbox

log = logging.getLogger(__name__)


class StabilityGate:
    def __init__(self, settings: Settings):
        self._s = settings
        st = settings.stability
        self._enabled = bool(st.get("enabled", False))
        self._threshold = float(st.get("similarity_threshold", 0.85))
        self._max_ref_age = float(st.get("max_ref_age_hours", 12)) * 3600
        self._ref_width = int(st.get("ref_width", 160))
        self._min_cov = float(st.get("min_coverage", 0.5))
        self._pad = float(settings.detection.get("crop_pad", 0.15))
        # spot.name -> {"crop": edge map, "ts": float}
        self._refs: dict[str, dict] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ crop
    def _crop(self, frame: np.ndarray, spot: Spot) -> np.ndarray | None:
        """Padded ROI crop -> CLAHE-normalized Canny edges, masked to the polygon
        interior (eroded, so the boundary edge does not dominate the comparison)."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi_bbox(spot.polygon, (h, w), self._pad)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = self._ref_width / gray.shape[1]
        nw, nh = self._ref_width, max(1, int(round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(gray, 50, 150)
        mask = np.zeros((nh, nw), dtype=np.uint8)
        poly = np.array(spot.polygon, dtype=np.float32)
        sx, sy = nw / (x1 - x0), nh / (y1 - y0)
        scaled = np.stack([(poly[:, 0] - x0) * sx, (poly[:, 1] - y0) * sy], axis=1)
        cv2.fillPoly(mask, [scaled.astype(np.int32)], 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
        return cv2.bitwise_and(edges, mask)

    # ------------------------------------------------------------- similarity
    def _similarity(self, ref: np.ndarray, cur: np.ndarray) -> float:
        """Edge-based NCC in [-1, 1]; 8px slide tolerance for alignment drift."""
        if ref.shape != cur.shape:
            cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]),
                             interpolation=cv2.INTER_AREA)
        pad = 8
        cur_p = cv2.copyMakeBorder(cur, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        return float(cv2.matchTemplate(cur_p, ref, cv2.TM_CCOEFF_NORMED).max())

    # ------------------------------------------------------------- reference
    def _refresh_ref(self, r: SpotResult) -> None:
        crop = self._crop(r.frame, r.spot)
        if crop is not None:
            self._refs[r.spot.name] = {"crop": crop, "ts": time.time()}

    def _get_ref(self, spot_name: str) -> dict | None:
        ref = self._refs.get(spot_name)
        if ref is None:
            return None
        if time.time() - ref["ts"] > self._max_ref_age:
            del self._refs[spot_name]
            return None
        return ref

    # ------------------------------------------------------------------ rule
    def check(self, r: SpotResult, confirmed: str | None) -> bool:
        """Apply the gate. Returns True when a FREE verdict is overridden to OCCUPIED.

        Also refreshes the reference when YOLO reports OCCUPIED with high coverage
        (so slow lighting drift is tracked). Never raises.
        """
        if not self._enabled or r.frame is None:
            return False
        try:
            if r.occupied:
                if r.coverage >= self._min_cov:
                    self._refresh_ref(r)
                return False
            if confirmed != "occupied":
                return False
            ref = self._get_ref(r.spot.name)
            if ref is None:
                return False
            cur = self._crop(r.frame, r.spot)
            if cur is None:
                return False
            sim = self._similarity(ref["crop"], cur)
            if sim >= self._threshold:
                log.info(
                    "stable_override: %r FREE->OCCUPIED (sim %.2f >= %.2f)",
                    r.spot.name, sim, self._threshold,
                )
                return True
            return False
        except Exception as exc:  # noqa: BLE001 - a gate must never crash the loop
            log.warning("stability gate error for %r: %s", r.spot.name, exc)
            return False
