"""appearance features, CarStore (vectors only), Matcher policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from parkingbot import appearance as ap
from parkingbot.car_store import CarStore
from parkingbot.matcher import MatchConfig, Matcher
from parkingbot.modes import StateStore


def car_img(kind: str, seed: int = 0, gray: bool = False):
    rng = np.random.default_rng(seed)
    img = np.full((220, 420, 3), 60, np.uint8)
    if kind == "owner":
        img[:] = (205, 200, 195)
        for x in range(0, 420, 60):
            cv2.rectangle(img, (x, 40), (x + 25, 110), (120, 120, 125), -1)       # windows
        cv2.rectangle(img, (30, 150), (390, 180), (90, 90, 95), -1)
    else:
        img[:] = (40, 45, 150)
        for i in range(10):
            cv2.circle(img, (40 + i * 38, 60 + (i % 3) * 40), 14, (200, 220, 240), -1)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    if gray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([g, g, g])
    return img


def vec(kind, seed=0, gray=False, box=(100, 150, 520, 370)):
    return ap.features(car_img(kind, seed, gray), box, (576, 960))


class AppearanceTest(unittest.TestCase):
    def test_vector_is_fixed_size_numbers_not_pixels(self):
        v = vec("owner")
        self.assertEqual(v.shape, (ap.DIM,))
        self.assertEqual(v.dtype, np.float32)
        self.assertLess(v.nbytes, 8000)                    # a 420x220 crop is 277 KB of pixels

    def test_same_car_scores_high_other_car_low(self):
        same, _ = ap.similarity(vec("owner", 1), vec("owner", 2))
        diff, parts = ap.similarity(vec("owner", 1), vec("other", 2))
        self.assertGreater(same, 0.95)
        self.assertLess(diff, 0.7)
        self.assertGreater(same - diff, 0.3)
        self.assertEqual(set(parts), {"hist", "tone", "layout", "edges", "geom"})

    def test_geometry_prior_penalises_a_different_box(self):
        a = vec("owner", box=(100, 150, 520, 370))
        b = vec("owner", box=(300, 100, 380, 500))        # thin sliver in another place
        self.assertGreater(ap.geom_similarity(a, a), 0.99)
        self.assertLess(ap.geom_similarity(a, b), 0.5)

    def test_condition_comes_from_the_frame_not_the_clock(self):
        self.assertEqual(ap.condition_of(car_img("owner")), "day")
        self.assertEqual(ap.condition_of(car_img("owner", gray=True)), "night")
        self.assertEqual(ap.condition_of(car_img("owner"), rain=True), "rain")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "s.db"
        self.ss = StateStore(self.path)
        self.cs = CarStore(self.ss.db)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_counts_and_filters(self):
        v = vec("owner")
        self.cs.add("A", "day", v, 1)
        self.cs.add("A", "night", v, 1)
        self.cs.add("A", "day", vec("other"), 0, source="confirm")
        self.cs.add("B", "day", v, 1)
        r = self.cs.refs("A", label=1, cond="day")
        self.assertEqual(len(r), 1)
        np.testing.assert_allclose(r[0]["vec"], v)
        c = self.cs.counts("A")
        self.assertEqual((c["day"]["pos"], c["day"]["neg"], c["night"]["pos"]), (1, 1, 1))
        self.assertEqual(self.cs.counts("B")["day"]["pos"], 1)

    def test_persists_across_reopen_and_delete(self):
        self.cs.add("A", "day", vec("owner"), 1)
        again = CarStore(StateStore(self.path).db)
        self.assertEqual(len(again.refs("A")), 1)
        self.assertEqual(again.delete_spot("A"), 1)
        self.assertEqual(again.refs("A"), [])

    def test_only_vectors_are_stored_never_images(self):
        self.cs.add("A", "day", vec("owner"), 1, meta={"box": [1, 2, 3, 4]})
        rows = self.ss.db.execute("SELECT vec, meta FROM car_refs").fetchall()
        for blob, meta in rows:
            self.assertEqual(len(blob), ap.DIM * 4)                  # exactly the feature vector
            self.assertLess(len(meta), 100)
        cols = [r[1] for r in self.ss.db.execute("PRAGMA table_info(car_refs)")]
        self.assertNotIn("image", cols)
        self.assertNotIn("crop", cols)


class MatcherTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cs = CarStore(StateStore(Path(self._tmp.name) / "s.db").db)
        self.m = Matcher(self.cs, MatchConfig())

    def tearDown(self):
        self._tmp.cleanup()

    def enroll(self, n=8, cond="day", gray=False, spot="A"):
        for i in range(n):
            self.cs.add(spot, cond, vec("owner", i, gray), 1)

    def test_no_references_always_asks(self):
        r = self.m.score("A", vec("owner"), "day")
        self.assertEqual(r.decision, "ask")
        self.assertIn("enroll", r.reason)

    def test_enrolled_car_is_recognised_automatically(self):
        self.enroll()
        r = self.m.score("A", vec("owner", 99), "day")
        self.assertEqual(r.decision, "auto")
        self.assertGreater(r.score, 0.92)

    def test_other_car_is_dismissed_silently(self):
        self.enroll()
        r = self.m.score("A", vec("other", 5), "day")
        self.assertEqual(r.decision, "other")
        self.assertLess(r.score, 0.75)

    def test_too_few_references_never_auto(self):
        self.enroll(n=3)
        r = self.m.score("A", vec("owner", 99), "day")
        self.assertEqual(r.decision, "ask")
        self.assertIn("references", r.reason)

    def test_night_query_with_only_day_references_asks_never_auto_or_other(self):
        self.enroll(cond="day")
        night_owner = vec("owner", 50, gray=True)
        r = self.m.score("A", night_owner, "night")
        self.assertEqual(r.refs_used, "other")
        self.assertNotEqual(r.decision, "auto")
        self.assertEqual(r.decision, "ask")                          # same car, different light
        far = self.m.score("A", vec("other", 7, gray=True), "night")
        self.assertEqual(far.decision, "other")                      # clearly different still dismissed

    def test_condition_matched_references_are_preferred(self):
        self.enroll(cond="day")
        self.enroll(cond="night", gray=True)
        r = self.m.score("A", vec("owner", 60, gray=True), "night")
        self.assertEqual(r.refs_used, "night")
        self.assertEqual(r.decision, "auto")

    def test_known_other_car_pulls_a_borderline_score_down(self):
        self.enroll()
        v = vec("owner", 77)
        self.cs.add("A", "day", v, label=0, source="confirm")        # same look labelled NOT mine
        r = self.m.score("A", v, "day")
        self.assertNotEqual(r.decision, "auto")
        self.assertIn("known other car", r.reason)

    def test_references_are_per_spot(self):
        self.enroll(spot="A")
        r = self.m.score("B", vec("owner", 1), "day")
        self.assertEqual(r.decision, "ask")                          # nothing enrolled for B

    def test_learn_adds_labelled_samples_that_change_the_next_match(self):
        self.enroll(n=3)                                             # too few -> always ask
        self.assertEqual(self.m.score("A", vec("owner", 90), "day").decision, "ask")
        for i in range(3):
            self.m.learn("A", vec("owner", 100 + i), "day", mine=True)
        self.assertEqual(self.m.score("A", vec("owner", 91), "day").decision, "auto")
        self.m.learn("A", vec("other", 3), "day", mine=False)
        self.assertEqual(self.cs.counts("A")["day"]["neg"], 1)


if __name__ == "__main__":
    unittest.main()
