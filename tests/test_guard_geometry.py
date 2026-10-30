"""guard_geometry: boxes, dilation, median, anisotropic main->substream zones."""

from __future__ import annotations

import unittest

from parkingbot.guard_geometry import (area, build_zones, dilate_box, inter_area, iou,
                                       median_box, point_in_poly, scale_box)


class BoxTest(unittest.TestCase):
    def test_iou_and_overlap(self):
        a, b = (0, 0, 10, 10), (5, 5, 15, 15)
        self.assertEqual(inter_area(a, b), 25)
        self.assertAlmostEqual(iou(a, b), 25 / 175)
        self.assertEqual(iou(a, (20, 20, 30, 30)), 0.0)
        self.assertEqual(area((5, 5, 1, 1)), 0.0)

    def test_median_ignores_one_bad_read(self):
        good = (100, 100, 200, 180)
        boxes = [good, good, (102, 99, 201, 182), (0, 0, 350, 280), good]
        m = median_box(boxes)
        self.assertEqual(m, good)

    def test_dilate_is_plus_40_25_percent_total_and_clamped(self):
        b = dilate_box((100, 100, 200, 180), 0.40, 0.25, 352, 288)
        self.assertAlmostEqual(b[2] - b[0], 140)           # +40 % of 100
        self.assertAlmostEqual(b[3] - b[1], 100)           # +25 % of 80
        edge = dilate_box((0, 0, 100, 100), 0.4, 0.25, 352, 288)
        self.assertEqual(edge[:2], (0, 0))                 # never leaves the frame


class ZonesTest(unittest.TestCase):
    def test_main_to_substream_scaling_is_anisotropic(self):
        # ch1: portrait 960x1080 squashed into 352x288
        z = build_zones([(285, 425, 510, 815)], (960, 1080), (352, 288))
        sx, sy = 352 / 960, 288 / 1080
        self.assertAlmostEqual(z.car[0], 285 * sx)
        self.assertAlmostEqual(z.car[3], 815 * sy)
        pb = z.perimeter_bbox
        self.assertLess(pb[0], z.car[0])
        self.assertGreater(pb[2], z.car[2])
        self.assertTrue(z.in_perimeter(((z.car[0] + z.car[2]) / 2, (z.car[1] + z.car[3]) / 2)))
        self.assertFalse(z.in_perimeter((1, 1)))

    def test_manual_perimeter_override(self):
        poly = [(0, 0), (480, 0), (480, 288), (0, 288)]
        z = build_zones([(100, 100, 300, 200)], (960, 576), (352, 288), perimeter_override_main=poly)
        self.assertTrue(z.in_perimeter((100, 100)))
        self.assertFalse(z.in_perimeter((300, 100)))       # x = 480 main -> 176.0 sub

    def test_detect_window_is_inside_frame_and_ints(self):
        z = build_zones([(10, 10, 300, 200)], (960, 576), (352, 288))
        x1, y1, x2, y2 = z.detect_window()
        self.assertTrue(all(isinstance(v, int) for v in (x1, y1, x2, y2)))
        self.assertTrue(0 <= x1 < x2 <= 352 and 0 <= y1 < y2 <= 288)

    def test_point_in_poly(self):
        self.assertTrue(point_in_poly((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)]))
        self.assertFalse(point_in_poly((15, 5), [(0, 0), (10, 0), (10, 10), (0, 10)]))

    def test_scale_box(self):
        self.assertEqual(scale_box((10, 20, 30, 40), 2, 0.5), (20, 10, 60, 20))


if __name__ == "__main__":
    unittest.main()
