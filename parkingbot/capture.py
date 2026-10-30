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
        # channel -> (w, h) snapshot frame size, discovered on first use. Lets the
        # detection path skip the snapshot once its size is known to mismatch the
        # ROI polygon coords (avoids wasting an HTTP grab on every poll).
        self._snapshot_size: dict[int, tuple[int, int]] = {}

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

    async def grab(self, channel: int, subtype: int,
                  expected_size: tuple[int, int] | None = None) -> bytes:
        """Grab one JPEG frame. Returns raw JPEG bytes. Serialized globally.

        expected_size: (width, height) the caller expects. When given (the
        detection path, where ROI polygons are in exact pixel coords), a snapshot
        frame of a different size is rejected and we fall back to ffmpeg. None
        (the viewer) accepts any valid JPEG.
        """
        method = str(self._s.capture.get("method", "auto"))
        async with self._lock:
            if method in ("snapshot", "auto"):
                # For detection (auto), skip the snapshot when its size is already
                # known to mismatch the expected ROI-polygon size — no point
                # wasting an HTTP grab on every poll.
                known = self._snapshot_size.get(channel)
                skip = (
                    method == "auto"
                    and expected_size is not None
                    and known is not None
                    and known != tuple(expected_size)
                )
                if not skip:
                    jpeg = await self._grab_snapshot(channel)
                    if jpeg is not None:
                        if expected_size is not None:
                            size = self._jpeg_size_wh(jpeg)
                            self._snapshot_size[channel] = size
                            if size != tuple(expected_size):
                                log.warning(
                                    "snapshot ch=%s size %s != expected %s, "
                                    "falling back to ffmpeg",
                                    channel, size, tuple(expected_size),
                                )
                                jpeg = None
                        if jpeg is not None:
                            log.info("snapshot grab ch=%s bytes=%d", channel, len(jpeg))
                            return jpeg
                    elif method == "snapshot":
                        raise CaptureError(f"snapshot failed for channel {channel}")
            return await self._grab_ffmpeg(channel, subtype)

    async def _grab_snapshot(self, channel: int) -> bytes | None:
        """Grab a frame via the HTTP snapshot API (no RTSP session, no HEVC
        first-keyframe wait). Returns the JPEG bytes, or None on any failure."""
        sec = self._s.secrets
        cap = self._s.capture
        timeout = float(cap.get("snapshot_timeout_seconds", 10))
        url = f"http://{sec.nvr_ip}/cgi-bin/snapshot.cgi?channel={channel}"
        try:
            import httpx

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    url, auth=httpx.DigestAuth(sec.rtsp_user, sec.rtsp_pass)
                )
                resp.raise_for_status()
                data = resp.content
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot grab failed for ch=%s: %s", channel, exc)
            return None
        if not data or data[:2] != b"\xff\xd8":  # JPEG SOI marker
            log.warning("snapshot returned a non-JPEG for ch=%s", channel)
            return None
        return data

    @staticmethod
    def _jpeg_size_wh(jpeg: bytes):
        """(width, height) of a JPEG, or None if it can't be decoded."""
        import cv2
        import numpy as np

        if not jpeg:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        if arr.size == 0:
            return None
        try:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except cv2.error:
            return None
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)

    async def _grab_ffmpeg(self, channel: int, subtype: int) -> bytes:
        """Grab one JPEG frame via ffmpeg (RTSP). Serialized globally.

        The JPEG goes to a pipe (stdout), never a file: the container runs with a
        read-only rootfs and the Spot Watch / Guard modes must never put an image
        on disk.
        """
        rtsp = self._s.rtsp
        cap = self._s.capture
        timeout = float(rtsp.get("timeout_seconds", 30))
        transport = rtsp.get("transport", "tcp")
        q = str(cap.get("jpeg_quality", 2))

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "error",
            "-rtsp_transport", transport,
            # ffmpeg-level socket I/O timeout (microseconds) as a backstop.
            # ffmpeg 6.x renamed the old -stimeout to -timeout for RTSP.
            "-timeout", str(int(timeout * 1_000_000)),
            "-i", self._rtsp_url(channel, subtype),
            "-frames:v", "1",
            "-q:v", q,
            "-f", "image2pipe",
            "-c:v", "mjpeg",
            "pipe:1",
        ]
        log.info(
            "ffmpeg grab ch=%s subtype=%s url=%s",
            channel, subtype, self._redacted_url(channel, subtype),
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # +10s wall-clock guard around the ffmpeg-level timeout.
            data, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout + 10
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise CaptureError(
                f"ffmpeg timed out grabbing channel {channel}"
            ) from exc

        if proc.returncode != 0 or not data:
            tail = (stderr or b"").decode(errors="replace").strip().splitlines()
            detail = tail[-1] if tail else "no stderr"
            raise CaptureError(
                f"ffmpeg failed for channel {channel} "
                f"(rc={proc.returncode}): {detail}"
            )
        log.info("grabbed ch=%s subtype=%s bytes=%d", channel, subtype, len(data))
        return data
