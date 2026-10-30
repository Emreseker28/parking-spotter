"""Box/zone geometry for GUARD (pure functions, no I/O).

Boxes are (x1, y1, x2, y2). GUARD runs on the substream (352x288 on this XVR) while
spot polygons and the arming reads are in main-stream pixels, so zones are built in
main coordinates and scaled (anisotropically: ch1 is a portrait 960x1080 frame
squashed into 352x288) into substream coordinates.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter_area(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    i = inter_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def scale_box(b: Box, sx: float, sy: float) -> Box:
    return (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)


def clamp_box(b: Box, w: float, h: float) -> Box:
    return (min(max(b[0], 0.0), w), min(max(b[1], 0.0), h),
            min(max(b[2], 0.0), w), min(max(b[3], 0.0), h))


def median_box(boxes: list[Box]) -> Box:
    """Element-wise median: robust to a bad read among the ~5 arming reads."""
    if not boxes:
        raise ValueError("no boxes")
    return tuple(statistics.median(b[i] for b in boxes) for i in range(4))  # type: ignore[return-value]


def dilate_box(b: Box, frac_w: float, frac_h: float, w: float, h: float) -> Box:
    """Grow a box by ``frac_w`` of its width and ``frac_h`` of its height in total
    (half per side), clamped to the frame. Defaults are +40 % / +25 %."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    return clamp_box((b[0] - bw * frac_w / 2, b[1] - bh * frac_h / 2,
                      b[2] + bw * frac_w / 2, b[3] + bh * frac_h / 2), w, h)


def box_poly(b: Box) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def poly_bbox(poly) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_poly(pt: tuple[float, float], poly) -> bool:
    contour = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


@dataclass(frozen=True)
class Zones:
    """GUARD zones in substream pixels."""
    frame_size: tuple[int, int]          # (w, h) of the substream frame
    car: Box
    perimeter: list[tuple[float, float]]  # polygon around the car (rect by default)

    @property
    def perimeter_bbox(self) -> Box:
        return poly_bbox(self.perimeter)

    def in_perimeter(self, pt: tuple[float, float]) -> bool:
        return point_in_poly(pt, self.perimeter)

    def detect_window(self, margin: float = 0.15) -> tuple[int, int, int, int]:
        """Crop window for the detector: the perimeter bbox plus a margin, as ints."""
        w, h = self.frame_size
        x1, y1, x2, y2 = self.perimeter_bbox
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        c = clamp_box((x1 - mx, y1 - my, x2 + mx, y2 + my), w, h)
        return (int(c[0]), int(c[1]), max(int(c[0]) + 8, int(c[2])), max(int(c[1]) + 8, int(c[3])))


def build_zones(car_boxes_main: list[Box], main_size: tuple[int, int],
                sub_size: tuple[int, int], dil_w: float = 0.40, dil_h: float = 0.25,
                perimeter_override_main=None) -> Zones:
    """Car zone = median of the arming reads (main-stream pixels), perimeter = the
    car zone dilated (or a manual polygon), both scaled into substream pixels."""
    sx, sy = sub_size[0] / main_size[0], sub_size[1] / main_size[1]
    car_main = median_box(car_boxes_main)
    car = clamp_box(scale_box(car_main, sx, sy), *sub_size)
    if perimeter_override_main:
        perim = [(x * sx, y * sy) for x, y in perimeter_override_main]
    else:
        perim = box_poly(dilate_box(car, dil_w, dil_h, *sub_size))
    return Zones(frame_size=tuple(sub_size), car=car, perimeter=perim)
