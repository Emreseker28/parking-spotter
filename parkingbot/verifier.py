"""Second-opinion spot verification with a LOCAL vision model (ollama).

Why: at night YOLOv8n often fails to detect the dark / top-down parked cars, so a
spot reads FREE and the bot cries "BOŞ" (empty) when a car is actually there. We ask
a local VLM (e.g. llava) to look at the cropped spot and say whether it's occupied.

Two callers, one method (`verify_spot`):
  - the push loop's GATE: don't send a FREE alert unless the VLM agrees it's empty;
  - the push loop's AUDITOR: log bot-vs-VLM agreement every poll for a credibility score.

Everything stays on-box (front-of-house frames never leave the network) and costs
nothing per call. The weights live on /mnt/models via ollama's OLLAMA_MODELS.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings, Spot
from .detection import roi_bbox

log = logging.getLogger(__name__)

# Kept deliberately explicit about night/IR and top-down cars — those are exactly the
# cases where YOLO fails and we need the VLM to still call the spot occupied.
_PROMPT = (
    "You are a parking-spot inspector. This is a cropped image of ONE parking spot at "
    "the front of a house. The image may be daytime, nighttime, or grayscale infrared. "
    "Decide whether a vehicle (car, van, SUV, truck, or motorcycle) is currently parked "
    "in this spot. A vehicle counts as parked even if it is dark, in shadow, low-contrast, "
    "or seen from a steep top-down angle. An EMPTY spot shows bare ground/pavement with no "
    "vehicle. Reply with ONLY a JSON object and nothing else: "
    '{"occupied": true or false, "confidence": a number from 0.0 to 1.0, "reason": "a few words"}'
)


@dataclass
class VerifyResult:
    ok: bool                    # did we get a usable answer from the VLM?
    occupied: bool | None       # VLM verdict (None if not ok)
    confidence: float
    reason: str


class Verifier:
    """Calls a local ollama vision model to judge whether a spot is occupied."""

    def __init__(self, settings: Settings):
        self._s = settings
        v = settings.verify
        self._enabled = bool(v.get("enabled", False))
        # OLLAMA_URL env wins (handy for Docker -> host.docker.internal).
        raw_url = os.environ.get("OLLAMA_URL") or v.get("ollama_url", "http://localhost:11434")
        self._url = raw_url.rstrip("/")
        self._model = str(v.get("model", "llava:7b"))
        self._timeout = float(v.get("timeout_seconds", 60))
        self._pad = float(v.get("crop_pad", 0.15))

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ crop
    def _crop_b64(self, img: np.ndarray, spot: Spot) -> str:
        """Crop the padded ROI (same framing the detector uses) -> base64 JPEG."""
        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi_bbox(spot.polygon, (h, w), self._pad)
        crop = img[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else img
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("cv2.imencode failed on ROI crop")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ---------------------------------------------------------------- verify
    async def verify_spot(self, img: np.ndarray | None, spot: Spot) -> VerifyResult:
        """Ask the local VLM whether `spot` is occupied in `img`. Never raises."""
        if not self._enabled:
            return VerifyResult(False, None, 0.0, "verifier disabled")
        if img is None:
            return VerifyResult(False, None, 0.0, "no frame available")

        try:
            b64 = self._crop_b64(img, spot)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the loop
            log.warning("verify: crop failed for %r: %s", spot.name, exc)
            return VerifyResult(False, None, 0.0, f"crop error: {exc}")

        payload = {
            "model": self._model,
            "prompt": _PROMPT,
            "images": [b64],
            "stream": False,
            "format": "json",           # ask ollama to constrain output to JSON
            "options": {"temperature": 0},
        }
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 - connection refused, timeout, etc.
            log.warning("verify: ollama call failed for %r: %s", spot.name, exc)
            return VerifyResult(False, None, 0.0, f"ollama error: {exc}")

        return self._parse(str(data.get("response", "")), spot)

    @staticmethod
    def _parse(raw: str, spot: Spot) -> VerifyResult:
        text = raw.strip()
        try:
            obj = json.loads(text)
        except Exception:
            # Salvage a JSON object embedded in extra prose, if any.
            lo, hi = text.find("{"), text.rfind("}")
            if lo != -1 and hi > lo:
                try:
                    obj = json.loads(text[lo:hi + 1])
                except Exception:
                    obj = None
            else:
                obj = None
        if not isinstance(obj, dict):
            log.warning("verify: unparseable VLM reply for %r: %s", spot.name, text[:160])
            return VerifyResult(False, None, 0.0, f"unparseable: {text[:120]}")

        occ = obj.get("occupied")
        if isinstance(occ, str):
            occ = occ.strip().lower() in ("true", "yes", "1", "occupied", "occupied.")
        if not isinstance(occ, bool):
            return VerifyResult(False, None, 0.0, f"no 'occupied' field: {text[:120]}")

        try:
            conf = float(obj.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(obj.get("reason", ""))[:200]
        return VerifyResult(True, occ, conf, reason)
