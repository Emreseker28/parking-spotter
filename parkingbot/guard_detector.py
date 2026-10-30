"""GUARD detector: YOLOv8n ONNX on a crop window of the substream frame.

Separate from ``detection.Detector`` (which is specialised for vehicle occupancy of
a spot polygon): this one returns person / bicycle / motorcycle / car detections for
a window, runs on a small static-input model (320 by default), and is fully
in-memory. Frames are never written anywhere.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from .guard_tracker import Det

log = logging.getLogger(__name__)

# COCO ids we care about -> name
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}


class GuardDetector:
    def __init__(self, model_path: Path, threads: int = 1, conf: float = 0.30,
                 nms_iou: float = 0.45):
        self.model_path = Path(model_path)
        self.threads = threads
        self.conf = conf
        self.nms_iou = nms_iou
        self._session = None
        self._input = None
        self.size = 0
        self._lock = threading.Lock()

    def ensure(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if not self.model_path.exists():
                raise FileNotFoundError(f"GUARD model not found: {self.model_path}")
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.inter_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.model_path), so,
                                                 providers=["CPUExecutionProvider"])
            inp = self._session.get_inputs()[0]
            self._input, self.size = inp.name, int(inp.shape[2])
            log.info("guard detector loaded %s (input %d, threads=%d)",
                     self.model_path.name, self.size, self.threads)

    def detect(self, frame: np.ndarray, window: tuple[int, int, int, int],
               conf: float | None = None) -> list[Det]:
        """Detect inside ``window`` = (x1, y1, x2, y2) of ``frame``; boxes come back in
        full-frame pixel coordinates."""
        self.ensure()
        x1, y1, x2, y2 = window
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 8 or cw < 8:
            return []
        s = self.size
        sc = min(s / cw, s / ch)
        nw, nh = max(1, int(round(cw * sc))), max(1, int(round(ch * sc)))
        canvas = np.full((s, s, 3), 114, np.uint8)
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = (canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0).copy()
        out = self._session.run(None, {self._input: x})[0][0].T          # (N, 84)
        ids = list(CLASSES)
        scores = out[:, [4 + i for i in ids]]                              # (N, 4)
        best = scores.argmax(1)
        conf_v = scores[np.arange(len(scores)), best]
        keep = conf_v >= (self.conf if conf is None else conf)
        if not keep.any():
            return []
        boxes, conf_k, cls_k = out[keep, :4], conf_v[keep], best[keep]
        dets: list[Det] = []
        for ci in np.unique(cls_k):
            m = cls_k == ci
            b = boxes[m]
            xywh = [[float(cx - w / 2), float(cy - h / 2), float(w), float(h)]
                    for cx, cy, w, h in b]
            idx = cv2.dnn.NMSBoxes(xywh, conf_k[m].astype(float).tolist(), 0.0, self.nms_iou)
            for i in np.array(idx).flatten():
                cx, cy, w, h = b[i]
                bx1 = (cx - w / 2 - px) / sc + x1
                by1 = (cy - h / 2 - py) / sc + y1
                bx2 = (cx + w / 2 - px) / sc + x1
                by2 = (cy + h / 2 - py) / sc + y1
                cid = ids[int(ci)]
                dets.append(Det(cid, CLASSES[cid], float(conf_k[m][i]),
                                (max(bx1, 0.0), max(by1, 0.0), bx2, by2)))
        return dets
