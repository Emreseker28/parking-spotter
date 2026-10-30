"""Owner-car matcher: score one appearance vector against the enrolled references.

Policy (plan section 5.4), deliberately conservative until real data says otherwise:
  score >= t_hi                 -> "auto"  (switch to GUARD, tell the owner, offer an undo)
  t_lo <= score < t_hi          -> "ask"   ("Is this your car?" [Yes] / [No])
  score <  t_lo                 -> "other" (someone else's car; stay in SPOT_WATCH)
and it NEVER decides "auto" without enough references for that spot (it asks instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import appearance as ap
from .car_store import CarStore


@dataclass
class MatchConfig:
    t_hi: float = 0.92
    t_lo: float = 0.75
    top_k: int = 3
    min_refs: int = 5                  # fewer positives than this for the spot -> always ask
    cross_t_lo: float = 0.55           # no same-condition refs: IR vs daylight owner scores ~0.70,
                                       # other cars <= 0.49 (30 h of real crops) -> below this = other
    neg_pull: float = 0.15             # score x(1 - this) if a known NOT-mine car looks as close
    weights: dict = field(default_factory=lambda: dict(ap.DEFAULT_WEIGHTS))
    geom_weight: float = 0.25


@dataclass
class MatchResult:
    score: float
    decision: str                      # auto | ask | other
    cond: str                          # condition of the query
    refs_used: str                     # which condition's references were used
    n_pos: int
    n_neg: int
    best_neg: float
    reason: str
    parts: dict = field(default_factory=dict)


class Matcher:
    def __init__(self, store: CarStore, cfg: MatchConfig | None = None):
        self.store = store
        self.cfg = cfg or MatchConfig()

    def _sims(self, vec, refs):
        return [ap.similarity(vec, r["vec"], self.cfg.weights, self.cfg.geom_weight) for r in refs]

    def score(self, spot: str, vec: np.ndarray, cond: str) -> MatchResult:
        c = self.cfg
        pos_all = self.store.refs(spot, label=1)
        neg_all = self.store.refs(spot, label=0)
        pos = [r for r in pos_all if r["cond"] == cond]
        used = cond
        if not pos:
            pos, used = pos_all, "other"      # IR vs daylight: weaker evidence, never auto
        if not pos:
            return MatchResult(0.0, "ask", cond, "none", 0, len(neg_all), 0.0,
                               "no references for this spot: run /enroll")
        sims = self._sims(vec, pos)
        ranked = sorted(sims, key=lambda t: -t[0])[: c.top_k]
        score = float(np.mean([s for s, _ in ranked]))
        parts = ranked[0][1]
        neg_same = [r for r in neg_all if r["cond"] == cond] or neg_all
        best_neg = max((s for s, _ in self._sims(vec, neg_same)), default=0.0)
        reason = ""
        if neg_same and best_neg >= score:
            score *= 1.0 - c.neg_pull
            reason = f"looks as much like a known other car ({best_neg:.2f}); "
        n_pos_cond = len([r for r in pos_all if r["cond"] == cond])
        if used == "other":
            # Different lighting than every reference: the score is weaker evidence, so
            # never switch silently and never dismiss the car unless it is clearly different.
            decision = "ask" if score >= c.cross_t_lo else "other"
            if decision == "ask":
                reason += f"no {cond} references yet (cross-condition score); "
        elif score >= c.t_hi:
            decision = "auto"
        elif score >= c.t_lo:
            decision = "ask"
        else:
            decision = "other"
        if decision == "auto" and len(pos_all) < c.min_refs:
            decision, reason = "ask", reason + f"only {len(pos_all)} references (< {c.min_refs}); "
        return MatchResult(score, decision, cond, used, n_pos_cond, len(neg_all), best_neg,
                           reason.strip() or "ok", parts)

    def learn(self, spot: str, vec: np.ndarray, cond: str, mine: bool, source: str = "confirm",
              meta: dict | None = None) -> int:
        """A confirmation / correction becomes a labelled sample (improves the next match)."""
        return self.store.add(spot, cond, vec, label=1 if mine else 0, source=source, meta=meta)
