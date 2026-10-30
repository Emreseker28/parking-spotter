"""Live check of enrollment + matching against the real XVR (console only, temp DB in tmpfs).
  docker run --rm --read-only --tmpfs /state ... python /probes/auto_live_check.py"""
import asyncio, logging, time
from pathlib import Path
from parkingbot.config import load_settings
from parkingbot.capture import CaptureWorker
from parkingbot.detection import Detector
from parkingbot.modes import StateStore, ModeController
from parkingbot.car_store import CarStore
from parkingbot.matcher import Matcher
from parkingbot.autoswitch import AutoSwitch, AutoConfig
from parkingbot.spotwatch import Outbox
logging.basicConfig(level=logging.WARNING)
s = load_settings(); store = StateStore(Path("/state/s.db")); ctl = ModeController(store, guard_available=True); ctl.verify()
cs = CarStore(store.db); cfg = AutoConfig.from_settings(s); m = Matcher(cs, cfg.match)
class N:
    async def send_text(self, *a, **k): pass
    async def send_photo(self, *a, **k): pass
    async def send_media_group(self, *a, **k): pass
class G:
    async def arm(self, i): return True, "ok"
    async def disarm(self, r=""): pass
auto = AutoSwitch(s, ctl, m, Detector(s), CaptureWorker(s), G(), Outbox(N()), cfg=cfg)
A = next(i for i, sp in enumerate(s.spots) if sp.channel == 1); B = next(i for i, sp in enumerate(s.spots) if sp.channel == 3)
async def main():
    for idx in (A, B):
        t = time.perf_counter(); ok, msg = await auto.enroll(idx)
        print(f"ENROLL {s.spots[idx].name!r}: ok={ok} in {time.perf_counter()-t:.0f}s :: {msg.splitlines()[0]}")
        print("   counts:", cs.counts(s.spots[idx].name))
    print("\nSELF MATCH (fresh reads vs own references):")
    reads = {}
    for idx in (A, B):
        for k in range(3):
            r = await auto._read_car(s.spots[idx])
            if r is None: print("   no car read for", s.spots[idx].name); continue
            vec, cond, box = r; reads.setdefault(idx, []).append((vec, cond))
            res = m.score(s.spots[idx].name, vec, cond)
            print(f"   {s.spots[idx].name!r:<16} cond={cond:<5} score={res.score:.3f} decision={res.decision:<5} refs_used={res.refs_used} n_pos={res.n_pos}")
    print("\nCROSS-SPOT (the OTHER spot's car scored against this spot's references = a different car):")
    for q, ref in ((B, A), (A, B)):
        for vec, cond in reads.get(q, [])[:2]:
            res = m.score(s.spots[ref].name, vec, cond)
            print(f"   car of {s.spots[q].name!r} vs refs of {s.spots[ref].name!r}: score={res.score:.3f} decision={res.decision} ({res.reason})")
    print("\n", auto.status_text())
asyncio.run(main())
