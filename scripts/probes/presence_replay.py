"""Run the PresenceMonitor over recorded substream frames spanning dusk (day -> IR switch)."""
import subprocess, sys, datetime as dt
from urllib.parse import quote
import cv2, numpy as np
from parkingbot.config import load_settings
from parkingbot.guard_presence import PresenceMonitor
S = load_settings(); sec = S.secrets
def grab(ch, t):
    s = t.strftime("%Y_%m_%d_%H_%M_%S"); e = (t + dt.timedelta(seconds=15)).strftime("%Y_%m_%d_%H_%M_%S")
    url = f"rtsp://{quote(sec.rtsp_user,safe='')}:{quote(sec.rtsp_pass,safe='')}@{sec.nvr_ip}:554/cam/playback?channel={ch}&subtype=1&starttime={s}&endtime={e}"
    r = subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-rtsp_transport","tcp","-i",url,"-frames:v","1","-f","image2pipe","-c:v","mjpeg","pipe:1"],capture_output=True,timeout=60)
    return cv2.imdecode(np.frombuffer(r.stdout,np.uint8),cv2.IMREAD_COLOR) if r.stdout else None
ZONES = {1: (116, 117, 186, 209), 3: (3, 66, 163, 188)}
start = dt.datetime(2026, 10, 29, 16, 30, 0); n = int(sys.argv[1]); step = int(sys.argv[2])
mons = {ch: PresenceMonitor(z, check_s=0, low_sim=0.30, low_checks=3, refresh_s=0, high_sim=0.60) for ch, z in ZONES.items()}
fixed = {}
print(f"{'time':<7}{'ch1 sim':>9}{'ch1 vs 16:30':>14}{'ch3 sim':>9}{'ch3 vs 16:30':>14}  colour(ch1)")
sus = {1: 0, 3: 0}; mins = {1: 1.0, 3: 1.0}
from parkingbot.guard_presence import signature, similarity
for k in range(n):
    t = start + dt.timedelta(minutes=step * k); row = [t.strftime("%H:%M")]; col = ""
    for ch in (1, 3):
        f = grab(ch, t)
        if f is None: row += ["-", "-"]; continue
        if ch == 1: col = "IR" if float(np.mean(np.abs(f[:,:,0].astype(int)-f[:,:,2].astype(int)))) < 6 else "colour"
        m = mons[ch]; trig = m.update(f, float(k))
        sig = signature(f, ZONES[ch])
        if ch not in fixed: fixed[ch] = sig
        row += [f"{m.sim:.2f}" if m.sim is not None else "seed", f"{similarity(fixed[ch], sig):.2f}"]
        if m.sim is not None: mins[ch] = min(mins[ch], m.sim)
        sus[ch] += 1 if trig else 0
    print(f"{row[0]:<7}{row[1]:>9}{row[2]:>14}{row[3]:>9}{row[4]:>14}  {col}", flush=True)
print("suspicions raised (3 consecutive low checks):", sus, "| min sim with refresh:", {c: round(v, 2) for c, v in mins.items()})
