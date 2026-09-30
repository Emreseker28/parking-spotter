"""Tests for the HTTP snapshot capture path and the ROI size safety check."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock

import cv2
import numpy as np

from parkingbot.capture import CaptureWorker


def _jpeg(w: int, h: int) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _worker(method: str = "auto") -> CaptureWorker:
    from parkingbot.config import load_settings

    s = load_settings()
    s.raw["capture"] = {
        "method": method, "viewer_subtype": 1, "detect_subtype": 0,
        "jpeg_quality": 2, "snapshot_timeout_seconds": 10,
    }
    return CaptureWorker(s)


class JpegSizeTest(unittest.TestCase):
    def test_size_wh(self):
        w = _worker()
        self.assertEqual(w._jpeg_size_wh(_jpeg(352, 288)), (352, 288))
        self.assertEqual(w._jpeg_size_wh(_jpeg(960, 1080)), (960, 1080))

    def test_non_jpeg_returns_none(self):
        w = _worker()
        self.assertIsNone(w._jpeg_size_wh(b"not a jpeg"))
        self.assertIsNone(w._jpeg_size_wh(b""))


class GrabMethodTest(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()

    def _grab(self, worker, *args, **kwargs):
        return self.loop.run_until_complete(worker.grab(*args, **kwargs))

    def test_auto_uses_snapshot_when_size_matches(self):
        w = _worker("auto")
        w._grab_snapshot = AsyncMock(return_value=_jpeg(960, 1080))
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        jpeg = self._grab(w, 1, 0, expected_size=(960, 1080))
        self.assertEqual(jpeg, _jpeg(960, 1080))
        w._grab_ffmpeg.assert_not_called()

    def test_auto_falls_back_on_size_mismatch(self):
        # Snapshot returns 352x288 but detection expects 960x1080 -> ffmpeg.
        w = _worker("auto")
        w._grab_snapshot = AsyncMock(return_value=_jpeg(352, 288))
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        jpeg = self._grab(w, 1, 0, expected_size=(960, 1080))
        self.assertEqual(jpeg, b"ffmpeg")
        w._grab_ffmpeg.assert_awaited_once()

    def test_auto_viewer_uses_snapshot_without_size_check(self):
        # No expected_size (viewer) -> snapshot used as-is, even at 352x288.
        w = _worker("auto")
        w._grab_snapshot = AsyncMock(return_value=_jpeg(352, 288))
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        jpeg = self._grab(w, 1, 1)
        self.assertEqual(jpeg, _jpeg(352, 288))
        w._grab_ffmpeg.assert_not_called()

    def test_auto_falls_back_on_snapshot_error(self):
        w = _worker("auto")
        w._grab_snapshot = AsyncMock(return_value=None)
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        jpeg = self._grab(w, 1, 1)
        self.assertEqual(jpeg, b"ffmpeg")

    def test_auto_skips_snapshot_on_known_mismatch(self):
        # First detection grab tries the snapshot, caches the mismatching size.
        w = _worker("auto")
        w._grab_snapshot = AsyncMock(return_value=_jpeg(352, 288))
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        self._grab(w, 1, 0, expected_size=(960, 1080))
        self.assertEqual(w._snapshot_size[1], (352, 288))
        # Second grab: known mismatch -> snapshot NOT called again.
        self._grab(w, 1, 0, expected_size=(960, 1080))
        self.assertEqual(w._grab_snapshot.await_count, 1)  # still only the first
        w._grab_ffmpeg.assert_awaited()

    def test_snapshot_mode_raises_on_failure(self):
        w = _worker("snapshot")
        w._grab_snapshot = AsyncMock(return_value=None)
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        with self.assertRaises(Exception):
            self._grab(w, 1, 1)
        w._grab_ffmpeg.assert_not_called()

    def test_rtsp_mode_always_ffmpeg(self):
        w = _worker("rtsp")
        w._grab_snapshot = AsyncMock(return_value=_jpeg(352, 288))
        w._grab_ffmpeg = AsyncMock(return_value=b"ffmpeg")
        jpeg = self._grab(w, 1, 0, expected_size=(960, 1080))
        self.assertEqual(jpeg, b"ffmpeg")
        w._grab_snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
