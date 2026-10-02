"""Perbandingan 4 mode transfer learning pada YOLOv8s. Dijalankan di Google Colab (GPU).
Jangan menulis API key di file ini; simpan di Colab Secrets (userdata)."""
import shutil
import pandas as pd
from ultralytics import YOLO

DATA = "data.yaml"
OUT = "results"
common = dict(data=DATA, epochs=100, imgsz=640, batch=16, device=0, seed=0,
              optimizer="AdamW", lr0=0.00143, project=OUT, exist_ok=True)

modes = {
    "full":    (YOLO("yolov8s.pt"),   dict()),                     # semua layer dilatih
    "partial": (YOLO("yolov8s.pt"),   dict(freeze=7)),             # blok akhir backbone ikut dilatih
    "feature": (YOLO("yolov8s.pt"),   dict(freeze=10)),            # backbone beku
    "scratch": (YOLO("yolov8s.yaml"), dict(pretrained=False)),     # tanpa pretrained
}

rows = []
for name, (model, kw) in modes.items():
    model.train(name=name, **common, **kw)
    best = YOLO(f"{OUT}/{name}/weights/best.pt")
    test = best.val(data=DATA, split="test", verbose=False)
    df = pd.read_csv(f"{OUT}/{name}/results.csv")
    df.columns = df.columns.str.strip()
    c = df["metrics/mAP50(B)"].values
    first = next((i + 1 for i, v in enumerate(c) if v >= 0.9), None)
    stable = next((i + 1 for i in range(len(c)) if (c[i:] >= 0.9).all()), None)
    rows.append(dict(mode=name, val_mAP50=c.max(), test_mAP50=test.box.map50,
                     test_mAP50_95=test.box.map, epoch_pertama_0_9=first,
                     epoch_stabil_0_9=stable, waktu_s=df["time"].iloc[-1]))

pd.DataFrame(rows).round(3).to_csv(f"{OUT}/ringkasan.csv", index=False)
shutil.make_archive("results_all", "zip", OUT)
