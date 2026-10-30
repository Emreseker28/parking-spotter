import httpx, sys
env = dict(l.strip().split("=",1) for l in open("/home/likewhosloss/parking-spotter/.env") if "=" in l and not l.startswith("#"))
auth = httpx.DigestAuth(env["RTSP_USER"], env["RTSP_PASS"])
base = f"http://{env['NVR_IP']}"
def get(path, **kw):
    try:
        r = httpx.get(base+path, auth=auth, timeout=10, **kw)
        return r.status_code, r.text
    except Exception as e:
        return None, repr(e)
for path in sys.argv[1:]:
    code, txt = get(path)
    print("==", path, code)
    lines = [l for l in txt.splitlines() if not any(k in l.lower() for k in ("password","passwd","secret","token"))]
    print("\n".join(lines[:int(80)]))
