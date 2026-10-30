"""Collect car crops from recorded main-stream frames (scratch dev data only; RAM -> one npz)."""
import subprocess, sys, datetime as dt, pickle
from pathlib import Path
from urllib.parse import quote
import cv2, numpy as np
from parkingbot.config import load_settings, Spot
from parkingbot.detection import Detector
S = load_settings(); D = Detector(S); D._tier2_path = Path("/mnt/models/yolov8s.onnx"); D._tier2_conf = 0.25
sec = S.secrets
def grab(ch, t):
    s = t.strftime("%Y_%m_%d_%H_%M_%S"); e = (t + dt.timedelta(seconds=20)).strftime("%Y_%m_%d_%H_%M_%S")
    url = f"rtsp://{quote(sec.rtsp_user,safe='')}:{quote(sec.rtsp_pass,safe='')}@{sec.nvr_ip}:554/cam/playback?channel={ch}&subtype=0&starttime={s}&endtime={e}"
    r = subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-rtsp_transport","tcp","-i",url,"-frames:v","1","-f","image2pipe","-c:v","mjpeg","pipe:1"],capture_output=True,timeout=60)
    return cv2.imdecode(np.frombuffer(r.stdout,np.uint8),cv2.IMREAD_COLOR) if r.stdout else None
SIZES = {1: (960,1080), 3: (960,576)}
def boxes(img, ch):
    w, h = SIZES[ch]
    whole = Spot("w", ch, [(0,0),(w,0),(w,h),(0,h)], (w,h))
    out = {}
    for d in D.detect_array(img): out[tuple(int(v) for v in d.box)] = d.score          # plain detector
    r = D.check_spot_tier2(img, whole)                                                   # tier-2 whole frame (night)
    if r is not None:
        for d in r.detections:
            b = tuple(int(v) for v in d.box)
            if not any(iou(b, o) > 0.5 for o in out): out[b] = d.score
    return list(out)
def iou(a, b):
    iw = max(0, min(a[2],b[2]) - max(a[0],b[0])); ih = max(0, min(a[3],b[3]) - max(a[1],b[1]))
    i = iw*ih; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u else 0
spots = {s.channel: s for s in S.spots}
data = []
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=3) - dt.timedelta(minutes=5)
n, step = int(sys.argv[1]), int(sys.argv[2])
for k in range(n):
    t = now - dt.timedelta(minutes=step * k)
    for ch in (1, 3):
        img = grab(ch, t)
        if img is None: continue
        gray = float(np.mean(np.abs(img[:,:,0].astype(int) - img[:,:,2].astype(int))))    # ~0 for IR/grayscale frames
        sp = spots[ch]; xs=[p[0] for p in sp.polygon]; ys=[p[1] for p in sp.polygon]; pb=(min(xs),min(ys),max(xs),max(ys))
        for b in boxes(img, ch):
            ov = max(0, min(b[2],pb[2]) - max(b[0],pb[0])) * max(0, min(b[3],pb[3]) - max(b[1],pb[1])) / max(1,(b[2]-b[0])*(b[3]-b[1]))
            x1,y1,x2,y2 = [max(0,v) for v in b]
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 30 or crop.shape[1] < 30: continue
            data.append(dict(ch=ch, t=t.strftime("%m-%d %H:%M"), box=b, in_spot=ov, color=gray, crop=crop.copy(), frame_hw=img.shape[:2]))
    print(k, t.strftime("%m-%d %H:%M"), len(data), flush=True)
pickle.dump(data, open(sys.argv[3], "wb"))
print("saved", len(data))
