"""Human feedback: inline ✅/❌ buttons, labeled ROI crops, feedback log (plan Phase 1).

Gives a real accuracy metric without the VLM and builds the dataset for Phase 7:
every alert and /parking result carries ✅ Doğru / ❌ Yanlış buttons; presses are
logged to feedback.jsonl and the judged ROI crop is saved to data/frames/ for
offline review and fine-tuning.

All new behavior sits behind `feedback.enabled` (default off until Phase 1 is
accepted). Telegram messages stay in Turkish.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Spot
from .detection import roi_bbox
from .jsonl import append_jsonl

log = logging.getLogger(__name__)

CB_PREFIX = "fb:"
VERDICTS = ("free", "occupied")
RESPONSES = ("ok", "no")

_TRANS = str.maketrans({"ö": "o", "ü": "u", "ı": "i", "ş": "s", "ç": "c", "ğ": "g"})


def sanitize_spot_name(name: str) -> str:
    """'Ön (sarı ev)' -> 'on_sari_ev' (ASCII-safe, filename-friendly)."""
    s = name.lower().translate(_TRANS)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "spot"


def save_spot_crop(frame: np.ndarray, spot: Spot, spot_idx: int, occupied: bool,
                   ts: float, frames_dir, pad: float, quality: int) -> str | None:
    """Save the padded ROI crop as <ts>_<spot>_<verdict>.jpg. Returns the frame id.

    Idempotent: if the crop for this frame already exists, returns its id without
    rewriting. Never raises — a failed save must not break the push loop.
    """
    fid = str(int(ts))
    fname = f"{fid}_{sanitize_spot_name(spot.name)}_{'occupied' if occupied else 'free'}.jpg"
    path = Path(frames_dir) / fname
    if path.exists():
        return fid
    try:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi_bbox(spot.polygon, (h, w), pad)
        crop = frame[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else frame
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            log.warning("feedback: imencode failed for %r", spot.name)
            return None
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, path)
        return fid
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the loop
        log.warning("feedback: crop save failed for %r: %s", spot.name, exc)
        return None


def prune_frames(frames_dir, max_files: int) -> None:
    """Keep at most `max_files` crops, deleting the oldest (filenames are ts-sorted)."""
    files = sorted(Path(frames_dir).glob("*.jpg"))
    for f in files[: max(0, len(files) - max_files)]:
        try:
            f.unlink()
        except OSError:
            pass


def parse_feedback_callback(data: str):
    """Parse 'fb:<spot_idx>:<bot_verdict>:<frame_id>:<response>' -> tuple, or None."""
    parts = (data or "").split(":")
    if len(parts) != 5 or parts[0] != CB_PREFIX[:-1]:
        return None
    _, idx_s, verdict, fid, response = parts
    if verdict not in VERDICTS or response not in RESPONSES:
        return None
    if not idx_s.isdigit() or not fid.isdigit():
        return None
    return int(idx_s), verdict, fid, response


def append_feedback(path, record: dict) -> None:
    """Append one feedback record to the feedback log (rotating JSONL)."""
    append_jsonl(path, record)


def build_accuracy_report(days: int, records: list[dict]) -> str:
    """Build the /accuracy report from feedback records. Pure; '' when no data.

    Reports BOŞ-alert precision (correct "BOŞ" / all "BOŞ" alerts), wrong count,
    per-spot breakdown, and a day/night split (night = 19:00-06:00 local).
    """
    cutoff = time.time() - days * 86400
    recs = [r for r in records
            if r.get("type") == "feedback" and float(r.get("ts", 0)) >= cutoff]
    if not recs:
        return ""
    wrong = sum(1 for r in recs if r.get("label") == "wrong")
    bos = [r for r in recs if r.get("context") == "alert" and not r.get("bot_occupied")]
    bos_correct = sum(1 for r in bos if r.get("label") == "correct")
    per_spot: dict[str, list[int]] = {}
    for r in recs:
        tally = per_spot.setdefault(r.get("spot", "?"), [0, 0])
        tally[1] += 1
        if r.get("label") == "correct":
            tally[0] += 1

    lines = [f"📊 *Doğruluk* (son {days} gün)"]
    if bos:
        pct = round(100 * bos_correct / len(bos))
        lines.append(f"• BOŞ alert isabeti: *{pct}%* ({bos_correct}/{len(bos)})")
    else:
        lines.append("• BOŞ alerti: yok")
    lines.append(f"• Yanlış işaretleme: *{wrong}*")
    for name, (c, t) in sorted(per_spot.items()):
        lines.append(f"• {name}: {round(100 * c / t)}% ({c}/{t})")
    if bos:
        for label, night in (("Gece (19-06)", True), ("Gündüz", False)):
            group = [r for r in bos
                     if (int(r.get("hour", 12)) >= 19 or int(r.get("hour", 12)) < 6) == night]
            if not group:
                continue
            w = sum(1 for r in group if r.get("label") == "wrong")
            pct = round(100 * (len(group) - w) / len(group))
            lines.append(f"• {label}: {w} yanlış / {len(group)} BOŞ alerti → %{pct}")
    return "\n".join(lines)
