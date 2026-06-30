"""Vehicle detection (YOLOv8n ONNX via onnxruntime) and per-spot ROI occupancy.

Occupancy logic (plan §6): run detection on a frame; a spot is OCCUPIED if any
vehicle box covers at least `roi_overlap_threshold` of the spot's ROI polygon.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT, Settings, Spot

log = logging.getLogger(__name__)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original pixels
    score: float
    cls: int


@dataclass
class SpotResult:
    spot: Spot
    occupied: bool
    coverage: float          # best vehicle coverage of the ROI [0..1]
    detections: list[Detection]


class Detector:
    def __init__(self, settings: Settings):
        self._s = settings
        det = settings.detection
        # Model location: PARKING_MODEL_PATH env wins (handy for non-Docker runs
        # pointing straight at /mnt/models), else config.yaml's model_path.
        # Relative paths resolve against the repo root, not the CWD.
        raw_path = os.environ.get("PARKING_MODEL_PATH") or det.get(
            "model_path", "models/yolov8n.onnx"
        )
        path = Path(raw_path)
        self._model_path = path if path.is_absolute() else (ROOT / path)
        self._input = int(det.get("input_size", 640))
        self._conf = float(det.get("conf_threshold", 0.30))
        self._nms_iou = float(det.get("nms_iou", 0.45))
        self._vehicle = set(det.get("vehicle_classes", [2, 3, 5, 7]))
        self._overlap_thr = float(det.get("roi_overlap_threshold", 0.15))
        # Per-spot ROI crop pass: some spots (extreme top-down / dark cars) are
        # invisible to the model on the full frame but clear once the ROI is
        # cropped out; others (clear near cars) only detect on the full frame.
        # We run BOTH and take the best coverage (plan §6 crop-and-upscale).
        self._crop_detect_enabled = bool(det.get("crop_detect", True))
        self._crop_pad = float(det.get("crop_pad", 0.15))
        self._crop_upscale = float(det.get("crop_upscale", 2.0))

        threads = int(settings.inference.get("omp_num_threads", 2))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))

        self._session = None  # lazy-loaded on first use
        self._input_name = None
        # cache of (h, w, spot.name) -> (roi_mask, roi_area)
        self._roi_cache: dict[tuple[int, int, str], tuple[np.ndarray, int]] = {}

    # ---------------------------------------------------------------- model
    def _ensure_session(self):
        if self._session is not None:
            return
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self._model_path}. Export it first:\n"
                f"  python scripts/export_model.py\n"
                f"(or mount a yolov8n.onnx into ./models). See README."
            )
        import onnxruntime as ort

        threads = int(self._s.inference.get("omp_num_threads", 2))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        log.info("Loaded ONNX model %s (threads=%d)", self._model_path, threads)

    # ----------------------------------------------------------- pre/post
    def _letterbox(self, img: np.ndarray):
        """Resize keeping aspect ratio, pad to a square `input`x`input`."""
        h, w = img.shape[:2]
        scale = min(self._input / h, self._input / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input, self._input, 3), 114, dtype=np.uint8)
        pad_y, pad_x = (self._input - nh) // 2, (self._input - nw) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _preprocess(self, img_bgr: np.ndarray):
        canvas, scale, pad_x, pad_y = self._letterbox(img_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]  # NCHW
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def _postprocess(self, output: np.ndarray, scale, pad_x, pad_y) -> list[Detection]:
        # YOLOv8 ONNX output: (1, 84, 8400) -> transpose to (8400, 84).
        preds = np.squeeze(output, 0).T  # (8400, 84)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        confs = class_scores[np.arange(class_scores.shape[0]), cls_ids]

        keep = (confs >= self._conf) & np.isin(cls_ids, list(self._vehicle))
        if not np.any(keep):
            return []
        boxes_xywh, confs, cls_ids = boxes_xywh[keep], confs[keep], cls_ids[keep]

        # cx,cy,w,h (letterboxed coords) -> x1,y1,x2,y2 in original image coords.
        cx, cy, w, h = boxes_xywh.T
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        # NMS via OpenCV (expects x,y,w,h boxes).
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(rects, confs.tolist(), self._conf, self._nms_iou)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()
        return [
            Detection(
                box=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                score=float(confs[i]),
                cls=int(cls_ids[i]),
            )
            for i in idxs
        ]

    @staticmethod
    def decode(jpeg_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode JPEG frame for detection.")
        return img

    def detect_array(self, img: np.ndarray) -> list[Detection]:
        """Run detection on a decoded BGR image; boxes in that image's coords."""
        self._ensure_session()
        tensor, scale, pad_x, pad_y = self._preprocess(img)
        output = self._session.run(None, {self._input_name: tensor})[0]
        return self._postprocess(output, scale, pad_x, pad_y)

    def detect(self, jpeg_bytes: bytes) -> tuple[list[Detection], tuple[int, int]]:
        """Return (vehicle detections, (height, width)) for a JPEG frame."""
        img = self.decode(jpeg_bytes)
        h, w = img.shape[:2]
        return self.detect_array(img), (h, w)

    # ----------------------------------------------------------- ROI logic
    def _roi_bbox(self, spot: Spot, shape: tuple[int, int], pad: float):
        h, w = shape
        xs = [p[0] for p in spot.polygon]
        ys = [p[1] for p in spot.polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pw, ph = int((x1 - x0) * pad), int((y1 - y0) * pad)
        return (max(0, x0 - pw), max(0, y0 - ph),
                min(w, x1 + pw), min(h, y1 + ph))

    def _crop_detect(self, img: np.ndarray, spot: Spot) -> list[Detection]:
        """Detect on the upscaled ROI crop; map boxes back to full-frame coords."""
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self._roi_bbox(spot, (h, w), self._crop_pad)
        if x1 <= x0 or y1 <= y0:
            return []
        crop = img[y0:y1, x0:x1]
        up = self._crop_upscale
        if up != 1.0:
            crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        mapped = []
        for d in self.detect_array(crop):
            bx1, by1, bx2, by2 = d.box
            mapped.append(Detection(
                box=(bx1 / up + x0, by1 / up + y0, bx2 / up + x0, by2 / up + y0),
                score=d.score, cls=d.cls,
            ))
        return mapped
    def _roi_mask(self, spot: Spot, shape: tuple[int, int]):
        h, w = shape
        key = (h, w, spot.name)
        cached = self._roi_cache.get(key)
        if cached is not None:
            return cached
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(spot.polygon, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 1)
        area = int(mask.sum())
        if area == 0:
            log.warning("ROI for spot %r has zero area in a %dx%d frame.", spot.name, w, h)
        self._roi_cache[key] = (mask, area)
        return mask, area

    def _coverage(self, mask: np.ndarray, area: int,
                  detections: list[Detection], shape: tuple[int, int]) -> float:
        h, w = shape
        best = 0.0
        if area > 0:
            for det in detections:
                x1, y1, x2, y2 = det.box
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inside = int(mask[iy1:iy2, ix1:ix2].sum())
                best = max(best, inside / area)
        return best

    def check_spot(self, img: np.ndarray, spot: Spot,
                   full_dets: list[Detection]) -> SpotResult:
        """Occupancy for one spot, combining full-frame and ROI-crop detections.

        `full_dets` is the full-frame detection list for the channel (computed
        once per channel and shared across its spots).
        """
        shape = img.shape[:2]
        mask, area = self._roi_mask(spot, shape)
        dets = list(full_dets)
        if self._crop_detect_enabled:
            dets += self._crop_detect(img, spot)
        cov = self._coverage(mask, area, dets, shape)
        return SpotResult(spot=spot, occupied=cov >= self._overlap_thr,
                          coverage=cov, detections=dets)
