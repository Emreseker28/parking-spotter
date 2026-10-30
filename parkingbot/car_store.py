"""Owner-car reference store (SQLite, VECTORS ONLY).

Each row is one appearance vector of a car crop plus tags. No crop, frame or other
image data is ever stored; the vector cannot be turned back into a picture.

Keyed by spot (the owner's car parks in a similar pose in the same spot) and by
condition (day / night-IR / rain), because IR frames and daylight frames of the same
car look very different. ``label`` is 1 for "this is the owner's car" and 0 for a
confirmed other car (a negative from a "No, not mine" answer).
"""

from __future__ import annotations

import json
import time

import numpy as np

CONDITIONS = ("day", "night", "rain")


class CarStore:
    def __init__(self, db):
        self._db = db
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS car_refs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, spot TEXT NOT NULL, cond TEXT NOT NULL, "
            "label INTEGER NOT NULL, source TEXT NOT NULL, vec BLOB NOT NULL, "
            "meta TEXT NOT NULL DEFAULT '{}', ts REAL NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS car_refs_spot ON car_refs(spot, cond, label)")
        self._db.commit()

    def add(self, spot: str, cond: str, vec: np.ndarray, label: int = 1, source: str = "enroll",
            meta: dict | None = None, ts: float | None = None) -> int:
        v = np.asarray(vec, dtype=np.float32)
        cur = self._db.execute(
            "INSERT INTO car_refs(spot, cond, label, source, vec, meta, ts) VALUES(?,?,?,?,?,?,?)",
            (spot, cond, int(label), source, v.tobytes(), json.dumps(meta or {}),
             time.time() if ts is None else ts))
        self._db.commit()
        return int(cur.lastrowid)

    def refs(self, spot: str, label: int | None = None, cond: str | None = None) -> list[dict]:
        q, args = "SELECT id, cond, label, source, vec, meta, ts FROM car_refs WHERE spot=?", [spot]
        if label is not None:
            q += " AND label=?"
            args.append(int(label))
        if cond is not None:
            q += " AND cond=?"
            args.append(cond)
        rows = self._db.execute(q + " ORDER BY id", args).fetchall()
        return [{"id": r[0], "cond": r[1], "label": r[2], "source": r[3],
                 "vec": np.frombuffer(r[4], dtype=np.float32), "meta": json.loads(r[5]), "ts": r[6]}
                for r in rows]

    def counts(self, spot: str) -> dict[str, dict[str, int]]:
        out = {c: {"pos": 0, "neg": 0} for c in CONDITIONS}
        for cond, label, n in self._db.execute(
                "SELECT cond, label, COUNT(*) FROM car_refs WHERE spot=? GROUP BY cond, label", (spot,)):
            out.setdefault(cond, {"pos": 0, "neg": 0})["pos" if label else "neg"] = n
        return out

    def delete_spot(self, spot: str, source: str | None = None) -> int:
        if source:
            cur = self._db.execute("DELETE FROM car_refs WHERE spot=? AND source=?", (spot, source))
        else:
            cur = self._db.execute("DELETE FROM car_refs WHERE spot=?", (spot,))
        self._db.commit()
        return cur.rowcount
