"""Ukur latensi per tahap (praproses, inferensi, pascaproses) untuk kelima model slide 14.

  python scripts/latency.py --device cuda            # semua model, fp32
  python scripts/latency.py --device cuda --half     # fp16 (mendekati TensorRT FP16 di Jetson)
  python scripts/latency.py --models efficientnet_b0 resnet18

Latensi hanya bergantung pada arsitektur, bukan pada nilai bobot, sehingga model dibuat tanpa bobot
pretrained. Praproses memakai citra potongan asli dari dataset_raw (test). Tulis nama perangkat pada laporan.
"""
import argparse, csv, os, statistics, time

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode
from torchvision import models, transforms as T

MODELS = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"]
SLIDE = {"mobilenet_v3_small": "Raspberry Pi, tanpa GPU", "mobilenet_v3_large": "Edge kelas menengah",
         "efficientnet_b0": "Jetson", "resnet18": "Praktikum", "resnet50": "Workstation GPU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--data", default="dataset_raw")
    ap.add_argument("--out", default="results/latensi.csv")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_grad_enabled(False)

    rows = [r for r in csv.DictReader(open(os.path.join(a.data, "metadata.csv"))) if r["split"] == "test"]
    imgs = [Image.open(os.path.join(a.data, r["nama_file"])).convert("RGB") for r in rows]
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    dtype = torch.float16 if a.half else torch.float32
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print(f"perangkat: {name} | presisi: {'fp16' if a.half else 'fp32'} | torch {torch.__version__}\n")

    out = []
    for n in a.models:
        m = models.get_model(n, weights=None, num_classes=4).eval().to(dev, dtype)
        x0 = tf(imgs[0]).unsqueeze(0)
        with FlopCounterMode(display=False) as fc:
            m(x0.to(dev, dtype))
        gmacs = fc.get_total_flops() / 2 / 1e9                 # konvensi torchvision (multiply-add)
        params = sum(p.numel() for p in m.parameters()) / 1e6

        pre, inf, post = [], [], []
        for i in range(a.warmup + a.n):
            img = imgs[i % len(imgs)]
            t0 = time.perf_counter()
            x = tf(img).unsqueeze(0).to(dev, dtype); sync()
            t1 = time.perf_counter()
            y = m(x); sync()
            t2 = time.perf_counter()
            k = int(torch.softmax(y.float(), 1).argmax(1).item())
            t3 = time.perf_counter()
            if i >= a.warmup:
                pre.append((t1 - t0) * 1e3); inf.append((t2 - t1) * 1e3); post.append((t3 - t2) * 1e3)
        mp, mi, mo = (statistics.mean(v) for v in (pre, inf, post))
        tot = mp + mi + mo
        out.append(dict(model=n, params_juta=round(params, 2), gmacs=round(gmacs, 2), pre_ms=round(mp, 2),
                        inferensi_ms=round(mi, 2), inferensi_p95_ms=round(float(np.percentile(inf, 95)), 2),
                        post_ms=round(mo, 2), total_ms=round(tot, 2), fps=round(1000 / tot, 1),
                        perangkat=name, presisi="fp16" if a.half else "fp32"))

    print("| Model | Param (juta) | GMACs | Pre (ms) | Inferensi (ms) | p95 inf. (ms) | Post (ms) | Total (ms) | FPS |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['model']} | {r['params_juta']} | {r['gmacs']} | {r['pre_ms']} | {r['inferensi_ms']} | "
              f"{r['inferensi_p95_ms']} | {r['post_ms']} | {r['total_ms']} | {r['fps']} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\ntersimpan:", a.out)


if __name__ == "__main__":
    main()
