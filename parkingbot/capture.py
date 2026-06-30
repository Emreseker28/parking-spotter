"""Snapshot capture via the ffmpeg binary (subprocess).

All captures are serialized behind a single asyncio lock: budget XVRs cap
simultaneous RTSP connections and this recorder has wedged under load before
(plan §7). The viewer, /parking, and the push loop therefore never stack
connections.

We grab a single JPEG with the ffmpeg binary rather than cv2.VideoCapture
because the stream is HEVC/H.265, which OpenCV's bundled FFmpeg often fails to
decode (plan §3).
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from urllib.parse import quote

from .config import Settings

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """A snapshot grab failed (timeout, auth, offline channel, ...)."""


class CaptureWorker:
    """Serialized ffmpeg snapshot grabber."""

    def __init__(self, settings: Settings):
        self._s = settings
        self._lock = asyncio.Lock()

    def _rtsp_url(self, channel: int, subtype: int) -> str:
        sec = self._s.secrets
        rtsp = self._s.rtsp
        # URL-encode credentials so special chars in the password don't break
        # the inline user:pass@host URL (plan §12).
        user = quote(sec.rtsp_user, safe="")
        password = quote(sec.rtsp_pass, safe="")
        port = rtsp.get("port", 554)
        path = rtsp["path_template"].format(channel=channel, subtype=subtype)
        return f"rtsp://{user}:{password}@{sec.nvr_ip}:{port}{path}"

    def _redacted_url(self, channel: int, subtype: int) -> str:
        sec = self._s.secrets
        rtsp = self._s.rtsp
        port = rtsp.get("port", 554)
        path = rtsp["path_template"].format(channel=channel, subtype=subtype)
        return f"rtsp://***:***@{sec.nvr_ip}:{port}{path}"

    async def grab(self, channel: int, subtype: int) -> bytes:
        """Grab one JPEG frame. Returns raw JPEG bytes. Serialized globally."""
        rtsp = self._s.rtsp
        cap = self._s.capture
        timeout = float(rtsp.get("timeout_seconds", 30))
        transport = rtsp.get("transport", "tcp")
        q = str(cap.get("jpeg_quality", 2))

        async with self._lock:
            with tempfile.TemporaryDirectory(prefix="parkbot_") as tmp:
                out = Path(tmp) / "frame.jpg"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-rtsp_transport", transport,
                    # ffmpeg-level socket I/O timeout (microseconds) as a backstop.
                    # ffmpeg 6.x renamed the old -stimeout to -timeout for RTSP.
                    "-timeout", str(int(timeout * 1_000_000)),
                    "-i", self._rtsp_url(channel, subtype),
                    "-frames:v", "1",
                    "-update", "1",  # required by ffmpeg 6.1 image2 muxer (plan §3)
                    "-q:v", q,
                    str(out),
                ]
                log.info(
                    "ffmpeg grab ch=%s subtype=%s url=%s",
                    channel, subtype, self._redacted_url(channel, subtype),
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    # +10s wall-clock guard around the ffmpeg-level timeout.
                    _, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout + 10
                    )
                except asyncio.TimeoutError as exc:
                    proc.kill()
                    await proc.wait()
                    raise CaptureError(
                        f"ffmpeg timed out grabbing channel {channel}"
                    ) from exc

                if proc.returncode != 0 or not out.exists():
                    tail = (stderr or b"").decode(errors="replace").strip().splitlines()
                    detail = tail[-1] if tail else "no stderr"
                    raise CaptureError(
                        f"ffmpeg failed for channel {channel} "
                        f"(rc={proc.returncode}): {detail}"
                    )

                data = out.read_bytes()
                if not data:
                    raise CaptureError(f"ffmpeg produced an empty frame for ch {channel}")
                log.info("grabbed ch=%s subtype=%s bytes=%d", channel, subtype, len(data))
                return data
