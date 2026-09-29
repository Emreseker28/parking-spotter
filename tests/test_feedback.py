"""Tests for parkingbot.feedback (crops, retention, callback parsing, accuracy report)."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from parkingbot.config import Spot
from parkingbot.feedback import (
    build_accuracy_report,
    parse_feedback_callback,
    prune_frames,
    save_spot_crop,
)

SPOT_A = Spot(name="Ön (sarı ev)", channel=1, polygon=[(285, 425), (510, 435), (505, 815), (295, 805)])
SPOT_B = Spot(name="ev önü", channel=3, polygon=[(20, 150), (540, 175), (545, 375), (30, 360)])


def _frame(h: int = 576, w: int = 960) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[100:400, 100:500] = (128, 128, 128)  # bright block inside both ROIs
    return img


class SaveSpotCropTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_saves_roi_crop_with_expected_name(self):
        fid = save_spot_crop(_frame(), SPOT_B, 1, False, 1759176825, self.dir, pad=0.15, quality=85)
        self.assertEqual(fid, "1759176825")
        expected = self.dir / "1759176825_ev_onu_free.jpg"
        self.assertTrue(expected.exists())
        img = cv2.imread(str(expected))
        self.assertIsNotNone(img)
        # Padded ROI bbox of SPOT_B in a 960x576 frame: x 0..623, y 117..408.
        self.assertEqual(img.shape[:2], (291, 623))

    def test_occupied_verdict_in_filename(self):
        save_spot_crop(_frame(), SPOT_A, 0, True, 1759176826, self.dir, pad=0.15, quality=85)
        self.assertTrue((self.dir / "1759176826_on_sari_ev_occupied.jpg").exists())

    def test_idempotent_second_call_returns_same_frame_id(self):
        first = save_spot_crop(_frame(), SPOT_B, 1, False, 1759176825, self.dir, 0.15, 85)
        second = save_spot_crop(_frame(), SPOT_B, 1, False, 1759176825, self.dir, 0.15, 85)
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.dir.glob("*.jpg"))), 1)

    def test_no_temp_files_left(self):
        save_spot_crop(_frame(), SPOT_B, 1, False, 1759176825, self.dir, 0.15, 85)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])


class PruneFramesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make(self, n: int):
        for i in range(n):
            (self.dir / f"{1759176800 + i:010d}_ev_onu_free.jpg").write_bytes(b"x")

    def test_keeps_newest_up_to_cap(self):
        self._make(5)
        prune_frames(self.dir, max_files=3)
        remaining = sorted(p.name for p in self.dir.glob("*.jpg"))
        self.assertEqual(len(remaining), 3)
        # Oldest two deleted, newest three kept.
        self.assertEqual(remaining[0], "1759176802_ev_onu_free.jpg")

    def test_no_prune_under_cap(self):
        self._make(2)
        prune_frames(self.dir, max_files=5)
        self.assertEqual(len(list(self.dir.glob("*.jpg"))), 2)


class ParseCallbackTest(unittest.TestCase):
    def test_valid_ok(self):
        self.assertEqual(
            parse_feedback_callback("fb:1:free:1759176825:ok"),
            (1, "free", "1759176825", "ok"),
        )

    def test_valid_no_occupied(self):
        self.assertEqual(
            parse_feedback_callback("fb:0:occupied:1759176825:no"),
            (0, "occupied", "1759176825", "no"),
        )

    def test_invalid_prefix(self):
        self.assertIsNone(parse_feedback_callback("cam:1"))

    def test_invalid_response(self):
        self.assertIsNone(parse_feedback_callback("fb:1:free:1759176825:maybe"))

    def test_invalid_verdict(self):
        self.assertIsNone(parse_feedback_callback("fb:1:empty:1759176825:ok"))

    def test_non_numeric_spot_idx(self):
        self.assertIsNone(parse_feedback_callback("fb:x:free:1759176825:ok"))

    def test_missing_fields(self):
        self.assertIsNone(parse_feedback_callback("fb:1:free"))


class BuildAccuracyReportTest(unittest.TestCase):
    def _rec(self, ts, spot, spot_idx, bot_occupied, label, hour, context):
        return {
            "ts": ts, "type": "feedback", "spot": spot, "spot_idx": spot_idx,
            "bot_occupied": bot_occupied, "coverage": 0.5, "label": label,
            "frame_id": str(int(ts)), "hour": hour, "context": context,
        }

    def test_empty_when_no_records(self):
        self.assertEqual(build_accuracy_report(7, []), "")

    def test_precision_wrong_per_spot_and_day_night_split(self):
        now = time.time()
        recs = [
            # BOŞ alerts (context=alert, bot_occupied=False)
            self._rec(now - 100, "ev önü", 1, False, "correct", 10, "alert"),   # day, correct
            self._rec(now - 200, "ev önü", 1, False, "correct", 11, "alert"),   # day, correct
            self._rec(now - 300, "ev önü", 1, False, "wrong", 22, "alert"),     # night, wrong
            # /parking feedback (context=parking)
            self._rec(now - 400, "Ön (sarı ev)", 0, True, "correct", 14, "parking"),
            self._rec(now - 500, "Ön (sarı ev)", 0, True, "wrong", 21, "parking"),  # night, wrong
        ]
        report = build_accuracy_report(7, recs)
        # BOŞ alert precision: 2/3 correct
        self.assertIn("BOŞ alert isabeti: *67%* (2/3)", report)
        # Wrong count: 2 (one alert, one parking)
        self.assertIn("Yanlış işaretleme: *2*", report)
        # Per-spot: Ön 1/2, ev önü 2/3
        self.assertIn("Ön (sarı ev): 50% (1/2)", report)
        self.assertIn("ev önü: 67% (2/3)", report)
        # Day/night split of BOŞ alerts: day 2/2 (100%), night 0/1 (0%)
        self.assertIn("Gece (19-06): 1 yanlış / 1 BOŞ alerti → %0", report)
        self.assertIn("Gündüz: 0 yanlış / 2 BOŞ alerti → %100", report)

    def test_old_records_excluded_by_days(self):
        now = time.time()
        recs = [
            self._rec(now - 10 * 86400, "ev önü", 1, False, "wrong", 10, "alert"),
        ]
        self.assertEqual(build_accuracy_report(7, recs), "")

    def test_no_bos_alerts_line(self):
        now = time.time()
        recs = [self._rec(now - 100, "Ön (sarı ev)", 0, True, "correct", 14, "parking")]
        report = build_accuracy_report(7, recs)
        self.assertIn("BOŞ alerti: yok", report)


class FeedbackHandlerTest(unittest.TestCase):
    """The callback handler end-to-end with a mocked Telegram update."""

    def _bot(self, tmp: Path):
        from parkingbot.config import load_settings
        from parkingbot.bot import ParkingBot

        s = load_settings()
        s.raw["feedback"] = {
            "enabled": True, "max_frames": 100,
            "frames_dir": str(tmp / "frames"), "jpeg_quality": 85,
            "log_path": str(tmp / "feedback.jsonl"),
        }
        return ParkingBot(s)

    def _press(self, bot, callback_data: str):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        query = MagicMock()
        query.data = callback_data
        query.answer = AsyncMock()
        update = MagicMock()
        update.callback_query = query
        update.effective_chat = MagicMock()
        update.effective_chat.id = bot.settings.secrets.allowed_chat_id
        asyncio.run(bot.on_feedback_button(update, None))
        return query

    def test_wrong_press_records_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self._bot(Path(tmp))
            bot._frame_index["1759176825:1"] = {
                "coverage": 0.0, "context": "alert", "occupied": False,
            }
            query = self._press(bot, "fb:1:free:1759176825:no")
            recs = bot._read_jsonl(bot._feedback_log)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["label"], "wrong")
            self.assertEqual(recs[0]["spot"], "ev önü")
            self.assertEqual(recs[0]["bot_occupied"], False)
            self.assertEqual(recs[0]["context"], "alert")
            self.assertEqual(recs[0]["frame_id"], "1759176825")
            query.answer.assert_awaited_once()

    def test_correct_press_records_correct_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self._bot(Path(tmp))
            bot._frame_index["1759176825:0"] = {
                "coverage": 0.62, "context": "parking", "occupied": True,
            }
            self._press(bot, "fb:0:occupied:1759176825:ok")
            recs = bot._read_jsonl(bot._feedback_log)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["label"], "correct")
            self.assertEqual(recs[0]["spot"], "Ön (sarı ev)")

    def test_invalid_callback_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = self._bot(Path(tmp))
            query = self._press(bot, "fb:9:free:1759176825:no")  # spot_idx out of range
            self.assertEqual(bot._read_jsonl(bot._feedback_log), [])
            query.answer.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
