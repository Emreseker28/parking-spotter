import sys, time, json, resource, numpy as np, cv2, openvino as ov
model, threads, frame = sys.argv[1], int(sys.argv[2]), sys.argv[3]
core = ov.Core(); m = core.read_model(model)
c = core.compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"})
H, W = c.input(0).shape[2], c.input(0).shape[3]
img = cv2.imread(frame); sc = min(W/img.shape[1], H/img.shape[0]); nw, nh = int(img.shape[1]*sc), int(img.shape[0]*sc)
cv = np.full((H, W, 3), 114, np.uint8); cv[:nh, :nw] = cv2.resize(img, (nw, nh))
x = (cv[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)/255).copy()
for _ in range(3): c(x)
ts = []
for _ in range(30):
    t = time.perf_counter(); c(x); ts.append((time.perf_counter()-t)*1000)
print(json.dumps(dict(model=model.split('/')[-1], size=f"{W}x{H}", threads=threads, engine="openvino", p50_ms=round(float(np.median(ts)),1), p95_ms=round(float(np.percentile(ts,95)),1), rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))))
