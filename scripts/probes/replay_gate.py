"""Phase 0 item 11 proxy: replay XVR playback (substream, 7fps, 352x288) through a prototype motion gate.
Frames live only in the ffmpeg pipe / RAM. usage: replay_gate.py channel start 'YYYY_mm_dd_HH_MM_SS' seconds"""
import subprocess, sys, os, numpy as np, cv2, json
from urllib.parse import quote
env = dict(l.strip().split("=",1) for l in open("/home/likewhosloss/parking-spotter/.env") if "=" in l and not l.startswith("#"))
ch, start, secs = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
import datetime as dt
t0 = dt.datetime.strptime(start, "%Y_%m_%d_%H_%M_%S"); end = (t0 + dt.timedelta(seconds=secs)).strftime("%Y_%m_%d_%H_%M_%S")
url = f"rtsp://{quote(env['RTSP_USER'],safe='')}:{quote(env['RTSP_PASS'],safe='')}@{env['NVR_IP']}:554/cam/playback?channel={ch}&subtype=1&starttime={start}&endtime={end}"
W, H = 352, 288
# perimeter crop in substream coords: spot polygon bbox scaled from the mainstream, dilated 40%/25%
poly = {3: ([(20,150),(540,175),(545,375),(30,360)], 960, 576), 1: ([(285,425),(510,435),(505,815),(295,805)], 960, 1080)}[ch]
xs = [p[0]*W/poly[1] for p in poly[0]]; ys = [p[1]*H/poly[2] for p in poly[0]]
x0,x1,y0,y1 = min(xs),max(xs),min(ys),max(ys); dw,dh = (x1-x0)*0.4/2, (y1-y0)*0.25/2
x0,x1,y0,y1 = [int(v) for v in (max(0,x0-dw),min(W,x1+dw),max(0,y0-dh),min(H,y1+dh))]
p = subprocess.Popen(["ffmpeg","-nostdin","-loglevel","error","-rtsp_transport","tcp","-i",url,"-t",str(secs),"-fps_mode","passthrough","-f","rawvideo","-pix_fmt","gray","pipe:1"], stdout=subprocess.PIPE)
bg = None; n = 0; motion = []; glob = 0; FR = W*H
while True:
    buf = p.stdout.read(FR)
    if len(buf) < FR: break
    g = np.frombuffer(buf, np.uint8).reshape(H, W)
    small = cv2.resize(g[y0:y1, x0:x1], (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    n += 1
    if bg is None: bg = small.copy(); motion.append(0); continue
    d = np.abs(small - bg) > 25; frac = d.mean()
    if frac > 0.40: glob += 1; motion.append(0)            # global change (IR switch, exposure): rejected
    else: motion.append(1 if frac > 0.004 else 0)
    bg = 0.95*bg + 0.05*small
fps = n / secs if secs else 0
m = np.array(motion); tail = int(10*7)                      # 10 s tail at 7 fps
act = np.zeros_like(m); last = -10**9
for i, v in enumerate(m):
    if v: last = i
    act[i] = 1 if i - last <= tail else 0
print(json.dumps(dict(ch=ch, start=start, secs=secs, frames=n, fps=round(fps,2), motion_frames=int(m.sum()), global_rejects=glob,
    active_frac_with_tail=round(float(act.mean()),3))))
