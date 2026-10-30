"""GUARD rules engine: tracks -> events (INFO / WARNING / ALERT).

Phase 2 rules: LOITER, CONTACT, MULTI. (CROUCH, VEHICLE_STOP, IMPACT, CAR_GONE are
later phases.) All thresholds come from config; ``night`` scales the durations down
(night sensitivity is higher).

``cond_ts`` on an event is the moment the rule's condition BECAME true (e.g. the
person's arrival + the loiter threshold), not when this code noticed, so the
end-to-end latency (condition met -> Telegram message sent) is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .guard_geometry import Zones, area, center, dilate_box, inter_area
from .guard_tracker import Track

INFO, WARNING, ALERT = "INFO", "WARNING", "ALERT"


@dataclass(frozen=True)
class GuardEvent:
    rule: str            # LOITER | CONTACT | MULTI
    severity: str
    cond_ts: float
    track_id: int | None = None
    seconds: float = 0.0     # how long the condition has held
    persons: int = 0


@dataclass
class RuleConfig:
    loiter_s: float = 10.0
    contact_s: float = 3.0
    multi_s: float = 2.0
    contact_overlap: float = 0.10     # share of the person's box inside the car box
    # ...AND the person's CENTRE must be inside the car box grown by this fraction per side.
    # From the real ch1 false alarms: a passer-by on the street has a box that overlaps the
    # car's image box by 15-30 % (camera angle) but a centre clearly outside it.
    contact_center_margin: float = 0.06
    # CONTACT must be made of real detections, not coasting: the track has to have been seen
    # within the last ``obs`` seconds and re-detected at least ``min_hits`` times during the
    # contact. (Real ch1 false alarm: ONE detection + ~2 s of coasting on a stale box counted
    # as "touching the car for 2 s".) Both scale with the current YOLO rate.
    contact_min_hits: int = 3
    contact_gap_s: float = 1.0        # tolerated detection gap inside a contact episode
    night_factor: float = 0.6         # durations x this at night
    multi_rearm_s: float = 10.0
    stale_s: float = 2.0              # an unseen track this long no longer counts as present


@dataclass
class _TrackState:
    in_since: float | None = None
    loiter_fired: bool = False
    contact_since: float | None = None
    contact_last: float | None = None
    contact_fired: bool = False
    contact_hits0: int = 0


def contact_s_for(cfg: "RuleConfig", k: float) -> float:
    return cfg.contact_s * k


class RulesEngine:
    def __init__(self, cfg: RuleConfig | None = None):
        self.cfg = cfg or RuleConfig()
        self._st: dict[int, _TrackState] = {}
        self._multi_since: float | None = None
        self._multi_fired = False
        self._multi_clear_since: float | None = None

    def reset(self) -> None:
        self._st.clear()
        self._multi_since = None
        self._multi_fired = False
        self._multi_clear_since = None

    def update(self, persons: list[Track], zones: Zones, now: float,
               night: bool = False, fps: float = 4.0) -> list[GuardEvent]:
        k = self.cfg.night_factor if night else 1.0
        obs_s = max(0.8, 1.6 / max(fps, 0.1))                    # "seen recently" window
        min_hits = max(2, min(self.cfg.contact_min_hits, int(contact_s_for(self.cfg, k) * fps)))
        loiter_s, contact_s, multi_s = (self.cfg.loiter_s * k, self.cfg.contact_s * k,
                                        self.cfg.multi_s * k)
        events: list[GuardEvent] = []
        alive = {t.id for t in persons}
        for tid in [t for t in self._st if t not in alive]:
            del self._st[tid]

        inside: list[Track] = []
        for t in persons:
            st = self._st.setdefault(t.id, _TrackState())
            if now - t.last_ts > self.cfg.stale_s:
                # coasting track (not detected recently): not present, dwell restarts
                st.in_since = None
                st.loiter_fired = False
                continue
            in_perim = zones.in_perimeter(center(t.box))
            if in_perim:
                inside.append(t)
                if st.in_since is None:
                    st.in_since = now
                held = now - st.in_since
                if held >= loiter_s and not st.loiter_fired:
                    st.loiter_fired = True
                    events.append(GuardEvent("LOITER", WARNING, st.in_since + loiter_s,
                                             t.id, held, len(persons)))
            else:
                st.in_since = None       # left the perimeter: the dwell starts over
                st.loiter_fired = False

            # CONTACT: the person's box overlaps the car box for a sustained time.
            pa = area(t.box)
            overlap = inter_area(t.box, zones.car) / pa if pa > 0 else 0.0
            m = self.cfg.contact_center_margin
            at_car = zones.contains_car(center(t.box), m)
            observed = now - t.last_ts <= obs_s
            if overlap >= self.cfg.contact_overlap and at_car and observed:
                if st.contact_since is None:
                    st.contact_since = now
                    st.contact_hits0 = t.hits
                st.contact_last = now
                held = now - st.contact_since
                if held >= contact_s and (t.hits - st.contact_hits0) >= min_hits and not st.contact_fired:
                    st.contact_fired = True
                    events.append(GuardEvent("CONTACT", ALERT, st.contact_since + contact_s,
                                             t.id, held, len(persons)))
            elif st.contact_last is not None and now - st.contact_last > self.cfg.contact_gap_s:
                st.contact_since = st.contact_last = None
                st.contact_fired = False

        # MULTI: two or more persons inside the perimeter at the same time.
        if len(inside) >= 2:
            self._multi_clear_since = None
            if self._multi_since is None:
                self._multi_since = now
            held = now - self._multi_since
            if held >= multi_s and not self._multi_fired:
                self._multi_fired = True
                events.append(GuardEvent("MULTI", WARNING, self._multi_since + multi_s,
                                         None, held, len(inside)))
        else:
            self._multi_since = None
            if self._multi_fired:
                if self._multi_clear_since is None:
                    self._multi_clear_since = now
                if now - self._multi_clear_since >= self.cfg.multi_rearm_s:
                    self._multi_fired = False
        return events
