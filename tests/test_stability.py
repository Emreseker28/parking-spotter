"""Tests for parkingbot.stability.StabilityGate (the 'no motion' gate)."""

from __future__ import annotations

import time
import unittest

import numpy as np

from parkingbot.config import Spot
from parkingbot.detection import SpotResult

SPOT_B = Spot(name="ev önü", channel=3, polygon=[(20, 150), (540, 175), (545, 375), (30, 360)])


def _frame(occupied: bool, w: int = 960, h: int = 576, noise: int = 0) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[200:280, 100:500] = 40
    if occupied:
        img[150:350, 80:480] = 160  # the "car" block inside the SPOT_B ROI
    if noise:
        rng = np.random.default_rng(42)
        img = np.clip(img.astype(np.int16) + rng.integers(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def _result(spot: Spot, frame: np.ndarray, occupied: bool, cov: float) -> SpotResult:
    return SpotResult(spot=spot, occupied=occupied, coverage=cov, detections=[], frame=frame)


class StabilityGateTest(unittest.TestCase):
    def _gate(self, **overrides):
        from parkingbot.config import load_settings
        from parkingbot.stability import StabilityGate

        s = load_settings()
        cfg = {
            "enabled": True, "similarity_threshold": 0.85,
            "max_ref_age_hours": 12, "ref_width": 160, "min_coverage": 0.5,
        }
        cfg.update(overrides)
        s.raw["stability"] = cfg
        return StabilityGate(s)

    # ------------------------------------------------------------- crop/sim
    def test_crop_shape_and_type(self):
        gate = self._gate()
        crop = gate._crop(_frame(True), SPOT_B)
        self.assertEqual(crop.shape[1], 160)  # ref_width
        self.assertEqual(crop.dtype, np.uint8)
        self.assertGreater(crop.shape[0], 0)

    def test_similarity_self_is_perfect(self):
        gate = self._gate()
        crop = gate._crop(_frame(True), SPOT_B)
        self.assertAlmostEqual(gate._similarity(crop, crop), 1.0, places=3)

    def test_similarity_departure_is_low(self):
        gate = self._gate()
        ref = gate._crop(_frame(True), SPOT_B)
        cur = gate._crop(_frame(False), SPOT_B)
        self.assertLess(gate._similarity(ref, cur), 0.5)

    def test_similarity_noisy_same_car_stays_high(self):
        gate = self._gate()
        ref = gate._crop(_frame(True), SPOT_B)
        cur = gate._crop(_frame(True, noise=25), SPOT_B)
        self.assertGreater(gate._similarity(ref, cur), 0.6)

    # ------------------------------------------------------------------ rule
    def test_disabled_gate_never_overrides(self):
        gate = self._gate(enabled=False)
        self.assertFalse(gate.check(_result(SPOT_B, _frame(True), False, 0.0), "occupied"))

    def test_occupied_reading_refreshes_ref_and_does_not_override(self):
        gate = self._gate()
        r = _result(SPOT_B, _frame(True), True, 0.75)
        self.assertFalse(gate.check(r, None))
        self.assertIsNotNone(gate._get_ref(SPOT_B.name))

    def test_free_confirmed_occupied_high_sim_overrides(self):
        gate = self._gate()
        gate.check(_result(SPOT_B, _frame(True), True, 0.75), None)  # set ref
        self.assertTrue(gate.check(_result(SPOT_B, _frame(True), False, 0.0), "occupied"))

    def test_free_confirmed_occupied_low_sim_does_not_override(self):
        gate = self._gate()
        gate.check(_result(SPOT_B, _frame(True), True, 0.75), None)  # set ref (car present)
        # Departure: empty frame -> low sim -> no override.
        self.assertFalse(gate.check(_result(SPOT_B, _frame(False), False, 0.0), "occupied"))

    def test_free_without_confirmed_occupied_does_not_override(self):
        gate = self._gate()
        gate.check(_result(SPOT_B, _frame(True), True, 0.75), None)
        self.assertFalse(gate.check(_result(SPOT_B, _frame(True), False, 0.0), None))
        self.assertFalse(gate.check(_result(SPOT_B, _frame(True), False, 0.0), "free"))

    def test_stale_ref_does_not_override(self):
        gate = self._gate()
        gate.check(_result(SPOT_B, _frame(True), True, 0.75), None)
        ref = gate._get_ref(SPOT_B.name)
        ref["ts"] = time.time() - 13 * 3600  # older than max_ref_age_hours=12
        self.assertFalse(gate.check(_result(SPOT_B, _frame(True), False, 0.0), "occupied"))

    def test_low_coverage_occupied_does_not_refresh_ref(self):
        gate = self._gate()
        gate.check(_result(SPOT_B, _frame(True), True, 0.10), None)  # cov < min_coverage
        self.assertIsNone(gate._get_ref(SPOT_B.name))


if __name__ == "__main__":
    unittest.main()
