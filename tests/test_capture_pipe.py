"""The ffmpeg grab must go to a pipe, never a file (read-only rootfs, no-disk rule)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from parkingbot.capture import CaptureError, CaptureWorker
from parkingbot.config import load_settings


class FakeProc:
    def __init__(self, out=b"", err=b"", rc=0):
        self._r = (out, err)
        self.returncode = rc

    async def communicate(self):
        return self._r


class PipeGrabTest(unittest.TestCase):
    def _grab(self, proc):
        captured = {}

        async def fake_exec(*cmd, **kw):
            captured["cmd"], captured["kw"] = list(cmd), kw
            return proc

        w = CaptureWorker(load_settings())
        with patch("asyncio.create_subprocess_exec", fake_exec):
            try:
                data = asyncio.run(w._grab_ffmpeg(3, 0))
            except CaptureError as exc:
                return captured, exc
        return captured, data

    def test_jpeg_comes_from_stdout_pipe_and_no_file_arg(self):
        cap, data = self._grab(FakeProc(out=b"\xff\xd8jpeg"))
        self.assertEqual(data, b"\xff\xd8jpeg")
        self.assertEqual(cap["cmd"][-1], "pipe:1")
        self.assertIn("image2pipe", cap["cmd"])
        self.assertFalse(any(str(a).endswith((".jpg", ".jpeg", ".png")) for a in cap["cmd"]))
        self.assertNotIn("-y", cap["cmd"])
        import asyncio.subprocess as sp
        self.assertEqual(cap["kw"]["stdout"], sp.PIPE)

    def test_empty_output_or_nonzero_exit_is_a_capture_error(self):
        _, exc = self._grab(FakeProc(out=b"", err=b"boom\n", rc=1))
        self.assertIsInstance(exc, CaptureError)
        self.assertIn("boom", str(exc))
        _, exc = self._grab(FakeProc(out=b"", rc=0))
        self.assertIsInstance(exc, CaptureError)


if __name__ == "__main__":
    unittest.main()
