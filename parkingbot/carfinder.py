"""Find the car inside a spot's polygon on a main-stream frame (shared by GUARD arming,
/enroll and arrival matching, so all three agree on what "the car" is).

Plain YOLOv8n misses some night / top-down cars (ch3), so it falls back to the tier-2
detector. Blocking: call it from a worker thread.
"""

from __future__ import annotations

import numpy as np

from .config import Spot
from .guard_geometry import Box, area, inter_area


def find_car(detector, img: np.ndarray, spot: Spot) -> Box | None:
    """Best car box (x1, y1, x2, y2 in main-stream pixels) inside ``spot``, or None."""
    full = detector.detect_array(img)
    r = detector.check_spot(img, spot, full)
    dets = list(r.detections) if r.occupied else []
    if not dets:
        t2 = detector.check_spot_tier2(img, spot)
        if t2 is not None and t2.occupied:
            dets = list(t2.detections)
    if not dets:
        return None
    xs = [p[0] for p in spot.polygon]
    ys = [p[1] for p in spot.polygon]
    pb = (min(xs), min(ys), max(xs), max(ys))
    best = max(dets, key=lambda d: inter_area(d.box, pb))
    if inter_area(best.box, pb) < 0.5 * area(best.box):
        return None                      # the best box is mostly outside the spot
    return tuple(float(v) for v in best.box)
