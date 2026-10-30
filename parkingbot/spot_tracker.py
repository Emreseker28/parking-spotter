"""Per-spot SPOT_WATCH state machine (pure logic, no I/O, injectable clock).

Feed it one occupancy reading at a time. It returns the events the caller should
announce. All timing (the ~10 s confirm re-check, the 60 s cadence) is the
caller's job; this class only counts *consecutive* readings.

Alert rules (plan section 3):
- a spot counts as free only after ``confirm_reads`` consecutive free readings;
- alert once on the free transition (or "free right now" when the mode is entered
  and the spot is already free);
- remind every ``remind_every`` seconds, at most ``remind_max`` times, while free;
- say "taken again" when it fills after we alerted;
- while muted (pause) state keeps tracking but nothing is announced, so the alert
  fires on the first reading after unmuting if the spot is still free.
"""

from __future__ import annotations

from dataclasses import dataclass

FREE = "free"
OCCUPIED = "occupied"


@dataclass(frozen=True)
class TrackerEvent:
    kind: str            # "free_alert" | "reminder" | "taken"
    immediate: bool = False   # free_alert: no occupied->free transition was seen
    n: int = 0                # reminder: 1-based reminder number


class SpotTracker:
    def __init__(self, name: str, confirm_reads: int = 2,
                 remind_every: float = 600.0, remind_max: int = 3):
        self.name = name
        self.confirm_reads = max(1, int(confirm_reads))
        self.remind_every = float(remind_every)
        self.remind_max = int(remind_max)
        self.confirmed: str | None = None
        self.pending_free = 0
        self.alerted = False
        self.reminders = 0
        self.last_alert_ts = 0.0
        self.free_since: float | None = None
        self.saw_occupied = False   # an occupied reading was seen this watch session
        self.just_arrived = False   # set on a confirmed free -> occupied flip; the engine clears it
        self.quiet_next_free = False   # swallow the next free alert (the owner just drove off)

    @property
    def needs_confirm(self) -> bool:
        """True while a free reading awaits its confirming re-check."""
        return self.confirmed != FREE and 0 < self.pending_free < self.confirm_reads

    def on_enter(self) -> None:
        """Mode (re)entered: re-arm so an already-free spot alerts."""
        self.alerted = False
        self.reminders = 0
        self.pending_free = 0
        self.saw_occupied = False

    def veto_free(self) -> None:
        """A gate (VLM) rejected the free verdict: go back to occupied."""
        self.confirmed = OCCUPIED
        self.pending_free = 0
        self.alerted = False
        self.reminders = 0
        self.free_since = None

    def observe(self, occupied: bool, now: float, muted: bool = False) -> list[TrackerEvent]:
        events: list[TrackerEvent] = []
        if occupied:
            was = self.confirmed
            self.pending_free = 0
            self.saw_occupied = True
            self.confirmed = OCCUPIED
            if was == FREE:
                self.just_arrived = True
            if was == FREE and self.alerted and not muted:
                events.append(TrackerEvent("taken"))
            if was != OCCUPIED:
                self.alerted = False
                self.reminders = 0
                self.free_since = None
            return events

        if self.confirmed != FREE:
            self.pending_free += 1
            if self.pending_free < self.confirm_reads:
                return events
            self.confirmed = FREE
            self.free_since = now
            self.pending_free = 0
        if muted:
            return events
        if not self.alerted:
            self.alerted = True
            self.last_alert_ts = now
            self.reminders = 0
            if self.quiet_next_free:         # they know: they just left it
                self.quiet_next_free = False
            else:
                events.append(TrackerEvent("free_alert", immediate=not self.saw_occupied))
        elif self.reminders < self.remind_max and now - self.last_alert_ts >= self.remind_every:
            self.reminders += 1
            self.last_alert_ts = now
            events.append(TrackerEvent("reminder", n=self.reminders))
        return events

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "confirmed": self.confirmed, "alerted": self.alerted,
            "reminders": self.reminders, "last_alert_ts": self.last_alert_ts,
            "free_since": self.free_since, "saw_occupied": self.saw_occupied,
        }

    def restore(self, d: dict) -> None:
        conf = d.get("confirmed")
        self.confirmed = conf if conf in (FREE, OCCUPIED) else None
        self.alerted = bool(d.get("alerted", False))
        self.reminders = int(d.get("reminders", 0))
        self.last_alert_ts = float(d.get("last_alert_ts", 0.0))
        fs = d.get("free_since")
        self.free_since = float(fs) if fs is not None else None
        self.saw_occupied = bool(d.get("saw_occupied", False))
        self.pending_free = 0
