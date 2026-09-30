"""Tests for the tier-2 detector session, RAM guard, and conf override."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parkingbot.config import Spot
from parkingbot.detection import Detector

SPOT_B = Spot(name="ev önü", channel=3, polygon=[(20, 150), (540, 175), (545, 375), (30, 360)])


def _detector(tmp: Path, tier2_path: str = "/nonexistent/yolov8s.onnx", **t2):
    from parkingbot.config import load_settings

    s = load_settings()
    cfg = {
        "enabled": True, "model_path": tier2_path,
        "conf_threshold": 0.15, "min_available_mb": 2500,
    }
    cfg.update(t2)
    s.raw["detection_tier2"] = cfg
    return Detector(s)


class MemAvailableTest(unittest.TestCase):
    def test_returns_positive_float(self):
        from parkingbot.bot import ParkingBot  # noqa: F401 - ensure import works

        # _mem_available_mb lives on ParkingBot; test it via a constructed bot.
        from parkingbot.config import load_settings
        from parkingbot.bot import ParkingBot

        b = ParkingBot(load_settings())
        mem = b._mem_available_mb()
        self.assertIsInstance(mem, float)
        self.assertGreater(mem, 0)


class Tier2SessionTest(unittest.TestCase):
    def test_session_is_lazy(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _detector(Path(tmp))
            self.assertIsNone(d._tier2_session)

    def test_missing_model_returns_none_and_stays_unloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _detector(Path(tmp))
            img = np.zeros((576, 960, 3), dtype=np.uint8)
            result = d.check_spot_tier2(img, SPOT_B)
            self.assertIsNone(result)
            self.assertIsNone(d._tier2_session)

    def test_unload_releases_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _detector(Path(tmp))
            d._tier2_session = object()  # simulate a loaded session
            d.unload_tier2()
            self.assertIsNone(d._tier2_session)

    def test_ensure_tier2_returns_session(self):
        """Regression: _ensure_tier2 must RETURN the session (a missing return made
        check_spot_tier2 crash with 'NoneType' object has no attribute 'run')."""
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "yolov8s.onnx"
            model.write_bytes(b"dummy")  # path must exist; content is mocked
            d = _detector(Path(tmp), tier2_path=str(model))
            fake_session = MagicMock()
            input_mock = MagicMock()
            input_mock.name = "images"
            fake_session.get_inputs.return_value = [input_mock]
            with patch("onnxruntime.InferenceSession", return_value=fake_session):
                result = d._ensure_tier2()
            self.assertIs(result, fake_session)
            self.assertIs(d._tier2_session, fake_session)
            self.assertEqual(d._tier2_input_name, "images")


class PostprocessConfOverrideTest(unittest.TestCase):
    def test_conf_override_filters_detections(self):
        # Build a fake YOLOv8 output: 2 boxes, one high-conf car, one low-conf car.
        # preds shape (8400, 84); we set box 0 = car @ 0.9, box 1 = car @ 0.1.
        preds = np.zeros((8400, 84), dtype=np.float32)
        # box 0: cx=320, cy=320, w=100, h=100 (letterboxed coords), car class score 0.9
        preds[0, 0:4] = [320, 320, 100, 100]
        preds[0, 4 + 2] = 0.9  # class 2 = car
        # box 1: car @ 0.1
        preds[1, 0:4] = [100, 100, 50, 50]
        preds[1, 4 + 2] = 0.1
        output = preds.T[None]  # (1, 84, 8400)

        with tempfile.TemporaryDirectory() as tmp:
            d = _detector(Path(tmp))
            # scale=1, pad=0 -> letterboxed coords == original coords
            dets_normal = d._postprocess(output, scale=1.0, pad_x=0, pad_y=0)
            self.assertEqual(len(dets_normal), 1)  # only the 0.9 box (conf_threshold 0.25)
            dets_low = d._postprocess(output, scale=1.0, pad_x=0, pad_y=0, conf_override=0.05)
            self.assertEqual(len(dets_low), 2)  # both boxes with the lower tier-2 conf


if __name__ == "__main__":
    unittest.main()
