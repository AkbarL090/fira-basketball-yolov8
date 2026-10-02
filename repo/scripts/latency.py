"""Ukur latensi per tahap pada perangkat target.
Pemakaian: python scripts/latency.py best.pt [best.onnx ...]
Memakai citra test asli (bukan citra kosong) agar waktu NMS/postprocess realistis."""
import glob, itertools, sys
import numpy as np
import cv2
from ultralytics import YOLO

imgs = [cv2.imread(p) for p in sorted(glob.glob("dataset_raw/test/images/*.jpg"))]
assert imgs, "tidak ada citra test"
N, WARM = 200, 20
print("| model | pre (ms) | inferensi (ms) | post (ms) | total (ms) | FPS |")
print("|---|---|---|---|---|---|")
for w in sys.argv[1:]:
    m = YOLO(w)
    cyc = itertools.cycle(imgs)
    for _ in range(WARM):
        m(next(cyc), imgsz=640, verbose=False)
    s = np.array([[r.speed[k] for k in ("preprocess", "inference", "postprocess")]
                  for r in (m(next(cyc), imgsz=640, verbose=False)[0] for _ in range(N))])
    mean = s.mean(0); tot = mean.sum()
    print(f"| {w} | {mean[0]:.1f} | {mean[1]:.1f} | {mean[2]:.1f} | {tot:.1f} | {1000/tot:.1f} |")
