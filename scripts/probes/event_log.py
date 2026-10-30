"""Log XVR event codes (code/action/index + host receive time). Text only. usage: event_log.py seconds out.tsv"""
import httpx, time, re, sys
env = dict(l.strip().split("=",1) for l in open("/home/likewhosloss/parking-spotter/.env") if "=" in l and not l.startswith("#"))
auth = httpx.DigestAuth(env["RTSP_USER"], env["RTSP_PASS"]); end = time.time()+int(sys.argv[1]); buf=""
with open(sys.argv[2],"a") as out, httpx.stream("GET", f"http://{env['NVR_IP']}/cgi-bin/eventManager.cgi?action=attach&codes=[All]", auth=auth, timeout=httpx.Timeout(15, read=30)) as r:
    for chunk in r.iter_text():
        buf += chunk; last = 0
        for m in re.finditer(r"Code=(\w+);action=(\w+);index=(\d+)", buf):
            out.write(f"{time.strftime('%H:%M:%S')}\t{m.group(1)}\t{m.group(2)}\t{m.group(3)}\n"); last = m.end()
        buf = buf[last:][-300:]; out.flush()
        if time.time() > end: break
