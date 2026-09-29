"""Tests for parkingbot.jsonl.append_jsonl (append + size-based rotation)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from parkingbot.jsonl import append_jsonl


class AppendJsonlTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "log.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _lines(self) -> list[dict]:
        text = self.path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def test_creates_file_and_appends_one_record(self):
        append_jsonl(self.path, {"a": 1})
        self.assertEqual(self._lines(), [{"a": 1}])

    def test_appends_multiple_records_in_order(self):
        append_jsonl(self.path, {"n": 1})
        append_jsonl(self.path, {"n": 2})
        append_jsonl(self.path, {"n": 3})
        self.assertEqual(self._lines(), [{"n": 1}, {"n": 2}, {"n": 3}])

    def test_creates_parent_directories(self):
        nested = self.path.parent / "deep" / "dir" / "log.jsonl"
        append_jsonl(nested, {"ok": True})
        self.assertTrue(nested.exists())

    def test_unicode_round_trips(self):
        append_jsonl(self.path, {"spot": "Ön (sarı ev)", "state": "BOŞ"})
        self.assertEqual(self._lines(), [{"spot": "Ön (sarı ev)", "state": "BOŞ"}])


class RotationTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "log.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _rotated(self, name: str) -> bool:
        return (self.path.parent / name).exists()

    def test_rotates_when_next_line_would_exceed_max_bytes(self):
        # First record fits; second would exceed -> rotation before writing it.
        append_jsonl(self.path, {"pad": "x" * 100}, max_bytes=150, keep=3)
        append_jsonl(self.path, {"pad": "y" * 100}, max_bytes=150, keep=3)
        self.assertTrue(self._rotated("log.jsonl.1"))
        # The old file (now .1) holds the first record; the new file the second.
        old = json.loads((self.path.parent / "log.jsonl.1").read_text())
        new = json.loads(self.path.read_text())
        self.assertEqual(old, {"pad": "x" * 100})
        self.assertEqual(new, {"pad": "y" * 100})

    def test_no_rotation_while_under_limit(self):
        append_jsonl(self.path, {"a": 1}, max_bytes=10_000, keep=3)
        append_jsonl(self.path, {"a": 2}, max_bytes=10_000, keep=3)
        self.assertFalse(self._rotated("log.jsonl.1"))

    def test_keeps_at_most_keep_generations(self):
        # keep=3 -> after 4 rotations, .1/.2/.3 exist but no .4.
        for i in range(5):
            append_jsonl(self.path, {"pad": "x" * 100, "i": i}, max_bytes=150, keep=3)
        self.assertTrue(self._rotated("log.jsonl.1"))
        self.assertTrue(self._rotated("log.jsonl.2"))
        self.assertTrue(self._rotated("log.jsonl.3"))
        self.assertFalse(self._rotated("log.jsonl.4"))

    def test_rotation_preserves_all_records_across_generations(self):
        records = [{"pad": "x" * 100, "i": i} for i in range(4)]
        for rec in records:
            append_jsonl(self.path, rec, max_bytes=150, keep=3)
        seen = []
        for name in ("log.jsonl.3", "log.jsonl.2", "log.jsonl.1", "log.jsonl"):
            p = self.path.parent / name
            if p.exists():
                seen.append(json.loads(p.read_text()))
        self.assertEqual(seen, records)


if __name__ == "__main__":
    unittest.main()
