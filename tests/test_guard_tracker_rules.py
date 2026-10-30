"""Tracker + rules engine: dwell, loiter/contact/multi thresholds, night, honest cond_ts."""

from __future__ import annotations

import unittest

from parkingbot.guard_geometry import Zones
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
