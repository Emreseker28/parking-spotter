"""Phase 0 probe 3: YOLO ONNX latency/RSS. usage: bench_ort.py model threads frame [provider] [runs]
Pinning is done by the caller (taskset). Prints one JSON line."""
import sys, json, time, os, resource
import numpy as np, cv2, onnxruntime as ort
model, threads, frame = sys.argv[1], int(sys.argv[2]), sys.argv[3]
provider = sys.argv[4] if len(sys.argv) > 4 else "cpu"
runs = int(sys.argv[5]) if len(sys.argv) > 5 else 25
so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
prov = {"cpu": ["CPUExecutionProvider"], "openvino": [("OpenVINOExecutionProvider", {"device_type": "CPU", "num_of_threads": threads}), "CPUExecutionProvider"]}[provider]
s = ort.InferenceSession(model, so, providers=prov)
inp = s.get_inputs()[0]; H, W = inp.shape[2], inp.shape[3]
img = cv2.imread(frame)
sc = min(W / img.shape[1], H / img.shape[0]); nw, nh = int(img.shape[1]*sc), int(img.shape[0]*sc)
canvas = np.full((H, W, 3), 114, np.uint8); canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
x = (canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255).copy()
for _ in range(3): s.run(None, {inp.name: x})
ts = []
for _ in range(runs):
    t = time.perf_counter(); out = s.run(None, {inp.name: x}); ts.append((time.perf_counter() - t) * 1000)
p = out[0][0].T  # (N, 84)
cls = p[:, 4:].argmax(1); conf = p[:, 4:].max(1)
dets = {n: int(((cls == c) & (conf > 0.25)).sum()) for n, c in (("person", 0), ("car", 2))}
print(json.dumps(dict(model=os.path.basename(model), size=f"{W}x{H}", threads=threads, provider=provider,
      p50_ms=round(float(np.median(ts)), 1), p95_ms=round(float(np.percentile(ts, 95)), 1),
      rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), raw_over_conf=dets)))
