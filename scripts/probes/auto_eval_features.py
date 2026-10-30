import pickle, numpy as np, collections, itertools, sys
from parkingbot import appearance as ap
data = pickle.load(open(sys.argv[1], "rb"))
def hour(t): m, d = t.split(); return int(d[:2]) + (0 if m == "10-30" else -24) * 0 + (24 if m == "10-30" else 0) / 1 * 0 if False else (int(d[:2]) + (24 if m=="10-30" else 0))
for d in data:
    d["cond"] = ap.condition_of(d["crop"]); d["vec"] = ap.features(d["crop"], d["box"], d["frame_hw"])
    mo, hm = d["t"].split(); h, mi = map(int, hm.split(":")); d["abs_min"] = ((0 if mo == "10-29" else 1) * 24 + h) * 60 + mi
    d["label"] = ("own" if d["in_spot"] >= 0.5 else ("other" if d["in_spot"] < 0.2 else "edge"))
cnt = collections.Counter((d["ch"], d["label"], d["cond"]) for d in data)
for k in sorted(cnt): print(k, cnt[k])
def sims(q, refs, w):
    return sorted((ap.similarity(q["vec"], r["vec"], w)[0] for r in refs), reverse=True)
def run(ch, w, label="own"):
    pos = [d for d in data if d["ch"] == ch and d["label"] == label]
    neg = [d for d in data if d["ch"] == ch and d["label"] == "other"]
    res = {}
    for cond in ("day", "night"):
        P = [d for d in pos if d["cond"] == cond]
        ps, ns = [], []
        for q in P:
            refs = [r for r in P if abs(r["abs_min"] - q["abs_min"]) > 120]
            if len(refs) >= 3: ps.append(np.mean(sims(q, refs, w)[:3]))
        for q in [d for d in neg if d["cond"] == cond]:
            if len(P) >= 3: ns.append(np.mean(sims(q, P, w)[:3]))
        res[cond] = (ps, ns)
    return res
def show(ch, w, name):
    print(f"\n== ch{ch} weights {name}")
    for cond, (ps, ns) in run(ch, w).items():
        if ps: print(f"  {cond:<5} owner: n={len(ps)} mean={np.mean(ps):.3f} min={np.min(ps):.3f} | other cars: n={len(ns)} " + (f"mean={np.mean(ns):.3f} max={np.max(ns):.3f}" if ns else "none") + (f" | GAP min(owner)-max(other)={np.min(ps)-np.max(ns):+.3f}" if ns else ""))
W = {"default": ap.DEFAULT_WEIGHTS}
for b in ("hist", "tone", "layout", "edges"): W["only " + b] = {b: 1.0}
for name, w in W.items():
    show(1, w, name)
print("\n########## ch3 (own = van in 'ev onu' polygon)")
for name in ("default", "only hist", "only layout", "only edges"): show(3, W[name], name)

print("\n########## ch1 own crops: per-crop score vs same-condition refs, with box size")
pos = [d for d in data if d["ch"] == 1 and d["label"] == "own"]
rows = []
for q in pos:
    refs = [r for r in pos if r["cond"] == q["cond"] and abs(r["abs_min"] - q["abs_min"]) > 120]
    if len(refs) >= 3:
        rows.append((np.mean(sims(q, refs, ap.DEFAULT_WEIGHTS)[:3]), q))
rows.sort(key=lambda t: t[0])
for sc, q in rows[:8] + rows[-3:]:
    b = q["box"]; print(f"  {sc:.3f} {q['cond']:<5} {q['t']} box={b} w={b[2]-b[0]} h={b[3]-b[1]} in_spot={q['in_spot']:.2f}")
ws = np.array([q["box"][2]-q["box"][0] for _, q in rows]); hs = np.array([q["box"][3]-q["box"][1] for _, q in rows])
print(f"  box width  median={np.median(ws):.0f} min={ws.min()} max={ws.max()} | height median={np.median(hs):.0f} min={hs.min()} max={hs.max()}")

print("\n########## cross-condition (query condition != reference condition), ch1, junk boxes excluded")
def sane(d):
    b = d["box"]; w, h = b[2]-b[0], b[3]-b[1]; return 150 <= w <= 260 and 280 <= h <= 420
own = [d for d in data if d["ch"] == 1 and d["label"] == "own" and sane(d)]
oth = [d for d in data if d["ch"] == 1 and d["label"] == "other"]
for qc, rc in (("night", "day"), ("day", "night")):
    R = [d for d in own if d["cond"] == rc]
    O = [np.mean(sims(q, R, ap.DEFAULT_WEIGHTS)[:3]) for q in own if q["cond"] == qc]
    N = [np.mean(sims(q, R, ap.DEFAULT_WEIGHTS)[:3]) for q in oth if q["cond"] == qc]
    print(f"  query {qc:<5} vs {rc} refs: owner mean={np.mean(O):.3f} min={np.min(O):.3f} | others " + (f"mean={np.mean(N):.3f} max={np.max(N):.3f}" if N else "n/a"))
print("\n########## sane-box rule: how many ch1 'own' crops would be rejected as unreliable reads")
allown = [d for d in data if d["ch"] == 1 and d["label"] == "own"]
print("  rejected", sum(1 for d in allown if not sane(d)), "of", len(allown))
