"""Zero-model appearance features for "is this my car?" (plan section 5.3, option 1).

A car crop becomes one float32 vector made of independent, individually L2-normalised
blocks, so similarity can weight them and the weights can be tuned on real data:

  hist    HSV colour histogram (daylight colour; in IR frames it degenerates to tone)
  tone    grey-level histogram
  layout  small grey thumbnail, mean/std normalised (overall shape + light/dark pattern)
  edges   small blurred edge map (body outline, windows, roof rails)
  geom    box geometry relative to the frame (aspect, size, position): a weak prior (S3)

Nothing here keeps pixels: the vector cannot be inverted back into an image.
Conditions: IR/night frames are grayscale, so "night" is detected from the frame itself
(colour channels nearly equal) rather than from the clock.
"""

from __future__ import annotations

import cv2
import numpy as np

HIST_BINS = (8, 4, 4)
HIST_N = HIST_BINS[0] * HIST_BINS[1] * HIST_BINS[2]      # 128
TONE_N = 32
LAYOUT_W, LAYOUT_H = 32, 20
LAYOUT_N = LAYOUT_W * LAYOUT_H                            # 640
EDGE_W, EDGE_H = 32, 20
EDGE_N = EDGE_W * EDGE_H                                  # 640
GEOM_N = 5
BLOCKS = {"hist": (0, HIST_N)}
_o = HIST_N
for _name, _n in (("tone", TONE_N), ("layout", LAYOUT_N), ("edges", EDGE_N), ("geom", GEOM_N)):
    BLOCKS[_name] = (_o, _o + _n)
    _o += _n
DIM = _o

DEFAULT_WEIGHTS = {"hist": 0.20, "tone": 0.10, "layout": 0.40, "edges": 0.30}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-9 else v.astype(np.float32)


def colorfulness(img: np.ndarray) -> float:
    """~0 for IR/grayscale frames (B == R), larger for daylight colour."""
    return float(np.mean(np.abs(img[:, :, 0].astype(np.int16) - img[:, :, 2].astype(np.int16))))


def condition_of(img: np.ndarray, rain: bool = False, color_thr: float = 6.0) -> str:
    """'night' for IR/grayscale frames, 'rain' when the owner tagged it, else 'day'."""
    if rain:
        return "rain"
    return "night" if colorfulness(img) < color_thr else "day"


def features(crop: np.ndarray, box: tuple, frame_hw: tuple) -> np.ndarray:
    """Appearance vector of ``crop`` (BGR); ``box`` is its (x1, y1, x2, y2) in the frame of
    size ``frame_hw`` = (h, w). Uses the central ~80 % of the crop to drop background."""
    h, w = crop.shape[:2]
    mx, my = int(w * 0.10), int(h * 0.10)
    core = crop[my:max(my + 1, h - my), mx:max(mx + 1, w - mx)]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).flatten()
    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    tone = cv2.calcHist([gray], [0], None, [TONE_N], [0, 256]).flatten()
    lay = cv2.resize(gray, (LAYOUT_W, LAYOUT_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    lay = (lay - lay.mean()) / (lay.std() + 1e-6)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(cv2.resize(gray, (96, 60)))
    edges = cv2.GaussianBlur(cv2.Canny(eq, 50, 150), (5, 5), 0)
    edges = cv2.resize(edges, (EDGE_W, EDGE_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    fh, fw = frame_hw
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    geom = np.array([bw / bh, bw / fw, bh / fh, (box[0] + box[2]) / 2 / fw, (box[1] + box[3]) / 2 / fh],
                    dtype=np.float32)
    return np.concatenate([_unit(hist), _unit(tone), _unit(lay.flatten()), _unit(edges.flatten()), geom])


def block(vec: np.ndarray, name: str) -> np.ndarray:
    a, b = BLOCKS[name]
    return vec[a:b]


def geom_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 when the box shape/size/position agree, falling towards 0 as they diverge."""
    ga, gb = block(a, "geom"), block(b, "geom")
    rel = np.abs(ga - gb) / (np.abs(gb) + 0.05)
    return float(np.clip(1.0 - float(rel.mean()), 0.0, 1.0))


def similarity(a: np.ndarray, b: np.ndarray, weights: dict | None = None,
               geom_weight: float = 0.25) -> tuple[float, dict]:
    """Combined similarity in ~[0, 1] plus the per-block values (for logging/tuning)."""
    w = weights or DEFAULT_WEIGHTS
    parts = {k: float(np.dot(block(a, k), block(b, k))) for k in w}
    appearance = sum(w[k] * parts[k] for k in w) / sum(w.values())
    g = geom_similarity(a, b)
    parts["geom"] = g
    return float((1 - geom_weight) * appearance + geom_weight * g), parts
