"""Phase 0 probe 1: codec/res/fps/bitrate of each channel+subtype. Creds never printed."""
import json, subprocess, sys, os
from urllib.parse import quote
env = dict(l.strip().split("=",1) for l in open("/home/likewhosloss/parking-spotter/.env") if "=" in l and not l.startswith("#"))
def url(ch, sub):
    return f"rtsp://{quote(env['RTSP_USER'],safe='')}:{quote(env['RTSP_PASS'],safe='')}@{env['NVR_IP']}:554/cam/realmonitor?channel={ch}&subtype={sub}"
for ch in [int(a) for a in sys.argv[1].split(",")]:
    for sub in [int(a) for a in sys.argv[2].split(",")]:
        r = subprocess.run(["ffprobe","-v","error","-rtsp_transport","tcp","-timeout","15000000","-show_streams","-of","json",url(ch,sub)],capture_output=True,text=True,timeout=40)
        if r.returncode: print(ch,sub,"FAIL",r.stderr.replace(env['RTSP_PASS'],'***')[-200:]); continue
        for s in json.loads(r.stdout)["streams"]:
            if s["codec_type"]=="video":
                print(f"ch{ch} sub{sub}: {s['codec_name']} {s.get('profile','')} {s['width']}x{s['height']} avg_fps={s.get('avg_frame_rate')} r_fps={s.get('r_frame_rate')} bitrate={s.get('bit_rate','n/a')} pix={s.get('pix_fmt')}")
