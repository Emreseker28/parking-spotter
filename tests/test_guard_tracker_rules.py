"""Tracker + rules engine: dwell, loiter/contact/multi thresholds, night, honest cond_ts."""

from __future__ import annotations

import unittest

from parkingbot.guard_geometry import Zones, box_poly, dilate_box
from parkingbot.guard_rules import ALERT, WARNING, RuleConfig, RulesEngine
from parkingbot.guard_tracker import Det, Tracker

CAR = (100.0, 100.0, 200.0, 180.0)
ZONES = Zones((352, 288), CAR, [(70, 80), (230, 80), (230, 200), (70, 200)])


def person(x, y, w=20, h=50):
    return Det(0, "person", 0.8, (x, y, x + w, y + h))


# Inside the perimeter, in the strip left of the car box (no overlap with the car).
BESIDE = person(78, 120, 15, 50)


class TrackerTest(unittest.TestCase):
    def test_id_persists_and_confirms_after_min_hits(self):
        t = Tracker(min_hits=2)
        self.assertEqual(t.update([person(10, 10)], 0.0), [])           # 1 hit: not confirmed
        tr = t.update([person(12, 10)], 0.3)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0].id, 1)
        self.assertEqual(t.update([person(14, 11)], 0.6)[0].id, 1)

    def test_fast_mover_matched_by_centroid(self):
        t = Tracker(min_hits=1)
        t.update([person(10, 10)], 0.0)
        tr = t.update([person(25, 10)], 0.5)           # no box overlap, centres close
        self.assertEqual(tr[0].id, 1)

    def test_ages_out_in_seconds_not_frames(self):
        t = Tracker(max_age=3.0, min_hits=1)
        t.update([person(10, 10)], 0.0)
        self.assertTrue(t.has_tracks())
        t.update([], 2.9)
        self.assertTrue(t.has_tracks())
        t.update([], 3.2)
        self.assertFalse(t.has_tracks())

    def test_classes_never_cross_match_and_two_people_get_two_ids(self):
        t = Tracker(min_hits=1)
        t.update([person(10, 10), Det(2, "car", 0.9, (12, 12, 40, 40))], 0.0)
        tr = t.update([person(11, 10), person(200, 100), Det(2, "car", 0.9, (13, 12, 41, 40))], 0.3)
        self.assertEqual(len({x.id for x in tr}), 3)


def run(rules, steps, night=False):
    """steps: list of (ts, [Det]); returns all events."""
    t = Tracker(min_hits=2)
    events = []
    for ts, dets in steps:
        persons = [x for x in t.update(dets, ts) if x.cls == 0]
        events += rules.update(persons, ZONES, ts, night)
    return events


def stay(det, start, end, dt=0.3):
    n = int((end - start) / dt) + 1
    return [(round(start + i * dt, 3), det if isinstance(det, list) else [det]) for i in range(n)]


class RulesTest(unittest.TestCase):
    def test_loiter_warns_once_after_10s_inside_perimeter(self):
        ev = run(RulesEngine(), stay(BESIDE, 0, 14))
        self.assertEqual([e.rule for e in ev], ["LOITER"])
        self.assertEqual(ev[0].severity, WARNING)
        self.assertGreaterEqual(ev[0].seconds, 10.0)

    def test_loiter_condition_time_is_when_it_became_true(self):
        ev = run(RulesEngine(), stay(BESIDE, 0, 14))
        # first confirmed frame is t=0.3; condition true at 0.3 + 10, noticed on a later frame
        self.assertAlmostEqual(ev[0].cond_ts, 10.3, places=3)

    def test_no_loiter_when_just_passing_or_outside(self):
        far = run(RulesEngine(), stay(person(300, 100), 0, 30))        # pavement, outside perimeter
        self.assertEqual(far, [])
        walking = [(i * 0.3, [person(78, 85 + i * 6, 15, 50)]) for i in range(30)]    # crosses in ~6 s
        self.assertEqual([e.rule for e in run(RulesEngine(), walking) if e.rule == "LOITER"], [])

    def test_leaving_perimeter_resets_the_dwell(self):
        steps = stay(BESIDE, 0, 8) + stay(person(300, 140), 8.3, 11.0) + stay(BESIDE, 11.3, 17)
        self.assertEqual(run(RulesEngine(), steps), [])               # never 10 s in one go

    def test_night_lowers_thresholds(self):
        steps = stay(BESIDE, 0, 8)
        self.assertEqual(run(RulesEngine(), steps, night=False), [])
        ev = run(RulesEngine(), steps, night=True)
        self.assertEqual([e.rule for e in ev], ["LOITER"])            # 6 s at night

    def test_contact_alerts_after_3s_overlap(self):
        p = person(150, 110, 30, 60)                                  # inside the car box
        ev = [e for e in run(RulesEngine(), stay(p, 0, 6)) if e.rule == "CONTACT"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].severity, ALERT)
        self.assertGreaterEqual(ev[0].seconds, 3.0)

    def test_no_contact_when_standing_next_to_the_car(self):
        p = person(205, 110, 20, 50)                                  # beside, not overlapping
        self.assertEqual([e for e in run(RulesEngine(), stay(p, 0, 8)) if e.rule == "CONTACT"], [])

    def test_contact_tolerates_a_short_detection_gap(self):
        p = person(150, 110, 30, 60)
        steps = stay(p, 0, 1.5) + stay(p, 2.4, 6)                     # 0.9 s hole < 1.0 s gap
        self.assertEqual(len([e for e in run(RulesEngine(), steps) if e.rule == "CONTACT"]), 1)

    def test_multi_needs_two_people_inside_perimeter(self):
        a, b = BESIDE, person(160, 120)
        ev = [e for e in run(RulesEngine(), stay([a, b], 0, 5)) if e.rule == "MULTI"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].persons, 2)
        self.assertEqual([e for e in run(RulesEngine(), stay([a, person(300, 140)], 0, 5))
                          if e.rule == "MULTI"], [])

    def test_custom_thresholds_are_respected(self):
        r = RulesEngine(RuleConfig(loiter_s=2.0))
        ev = run(r, stay(BESIDE, 0, 4))
        self.assertEqual([e.rule for e in ev], ["LOITER"])


if __name__ == "__main__":
    unittest.main()


# ---- regressions from REAL false alarms on ch1 (2026-10-30 22:39 and 22:44) --------------------
# Substream pixels. Car zone and perimeter as armed in production (perimeter now +16 % / +10 %).
REAL_CAR = (116.0, 117.0, 185.0, 207.0)
REAL_ZONES = Zones((352, 288), REAL_CAR, box_poly(dilate_box(REAL_CAR, 0.16, 0.10, 352, 288)))

# person detections (clip seconds, box) of the pedestrian who walked down the lane beside the car
PED_BESIDE = [(28.7, (100, 113, 124, 178)), (28.9, (98, 114, 123, 179)), (29.0, (97, 116, 122, 186)),
              (29.1, (96, 121, 121, 192)), (29.3, (96, 123, 120, 193)), (29.4, (95, 123, 120, 193)),
              (29.6, (94, 126, 120, 201)), (30.0, (92, 135, 118, 211)), (30.1, (93, 139, 119, 217)),
              (30.3, (93, 144, 123, 224)), (30.4, (94, 148, 123, 226)),
              (31.0, (104, 159, 142, 233)),                      # ONE detection, then nothing for 2 s
              (33.0, (104, 89, 127, 136))]
# pedestrian who walked behind (up the street from) the car, left to right
PED_BEHIND = [(36.0 + i * 0.12, b) for i, b in enumerate([
    (87, 101, 99, 170), (87, 103, 97, 170), (87, 103, 100, 170), (87, 101, 101, 165), (88, 99, 103, 162),
    (88, 98, 104, 162), (88, 98, 104, 163), (89, 95, 104, 161), (90, 92, 105, 153), (91, 90, 106, 152),
    (92, 90, 106, 152), (93, 90, 110, 151), (95, 90, 111, 149), (97, 90, 113, 147), (98, 89, 114, 145),
    (101, 89, 116, 144), (101, 89, 116, 140), (102, 89, 118, 138), (104, 89, 121, 138), (106, 89, 120, 138),
    (107, 89, 122, 136), (108, 89, 123, 135), (109, 89, 125, 133), (110, 89, 126, 135)])]


def replay(series, end, night=True, fps=4.0, cfg=None):
    rules, tr, events = RulesEngine(cfg or RuleConfig()), Tracker(), []
    t = series[0][0]
    by_t = {round(ts, 2): b for ts, b in series}
    while t <= end:
        b = by_t.get(round(t, 2))
        persons = [x for x in tr.update([Det(0, "person", 0.7, tuple(map(float, b)))] if b else [], t)
                   if x.cls == 0]
        events += rules.update(persons, REAL_ZONES, t, night, fps=fps)
        t = round(t + 0.1, 2)
    return events


class RealPedestrianRegressionTest(unittest.TestCase):
    def test_pedestrian_walking_down_the_lane_beside_the_car_raises_nothing(self):
        self.assertEqual(replay(PED_BESIDE, 36.0), [])

    def test_pedestrian_walking_behind_the_car_raises_nothing(self):
        self.assertEqual(replay(PED_BEHIND, 44.0), [])

    def test_a_single_detection_followed_by_coasting_is_not_contact(self):
        inside = (130.0, 130.0, 160.0, 200.0)                         # centre well inside the car box
        ev = replay([(10.0, inside)], 14.0)
        self.assertEqual([e for e in ev if e.rule == "CONTACT"], [])

    def test_someone_actually_at_the_car_still_raises_contact(self):
        inside = (130.0, 130.0, 160.0, 200.0)
        series = [(10.0 + i * 0.25, inside) for i in range(16)]       # 4 s of repeated detections
        ev = [e for e in replay(series, 15.0) if e.rule == "CONTACT"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].severity, ALERT)

    def test_contact_still_works_when_the_limiter_drops_yolo_to_1_fps(self):
        inside = (130.0, 130.0, 160.0, 200.0)
        series = [(10.0 + i, inside) for i in range(6)]               # one detection per second
        ev = [e for e in replay(series, 17.0, fps=1.0) if e.rule == "CONTACT"]
        self.assertEqual(len(ev), 1)

    def test_centre_must_be_at_the_car_not_just_overlapping_its_box(self):
        edge = (90.0, 120.0, 124.0, 190.0)                            # overlaps the car box edge ~20 %,
        series = [(10.0 + i * 0.25, edge) for i in range(20)]         # centre x=107 is outside it
        self.assertEqual([e for e in replay(series, 15.0) if e.rule == "CONTACT"], [])
        wide = RuleConfig(contact_center_margin=100.0)                # the old behaviour
        self.assertEqual(len([e for e in replay(series, 15.0, cfg=wide) if e.rule == "CONTACT"]), 1)

    def test_default_perimeter_margin_is_narrow(self):
        from parkingbot.config import load_settings
        from parkingbot.guard import GuardConfig
        c = GuardConfig.from_settings(load_settings())
        self.assertLessEqual(c.dil_w, 0.20)
        self.assertLessEqual(c.dil_h, 0.12)
        self.assertEqual(c.rules.contact_min_hits, 3)
        self.assertGreater(c.rules.contact_center_margin, 0.0)
