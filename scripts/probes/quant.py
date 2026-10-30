import sys, glob, numpy as np, cv2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
src, dst, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
class R(CalibrationDataReader):
    def __init__(s):
        s.it = iter(sorted(glob.glob("probe_frames/*.jpg")))
    def get_next(s):
        f = next(s.it, None)
        if f is None: return None
        img = cv2.imread(f); sc = min(S/img.shape[1], S/img.shape[0]); nw, nh = int(img.shape[1]*sc), int(img.shape[0]*sc)
        cv = np.full((S, S, 3), 114, np.uint8); cv[:nh, :nw] = cv2.resize(img, (nw, nh))
        return {"images": (cv[:, :, ::-1].transpose(2,0,1)[None].astype(np.float32)/255).copy()}
quantize_static(src, dst, R(), quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
print("quantized", dst)
