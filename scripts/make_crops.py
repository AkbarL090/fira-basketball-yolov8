"""Membuat dataset klasifikasi (dataset_raw/<kelas>/...) dari frame beranotasi di sumber_frame/.

Setiap kotak anotasi dipotong menjadi satu citra (dengan sedikit konteks di sekelilingnya).
Kelas `latar` diambil dari area tanpa objek. Split train/valid/test dibuat berdasarkan BLOK
frame berurutan dengan jeda antar blok, bukan acak, agar frame bertetangga tidak bocor
ke split lain (data leakage, slide 22).
"""
import csv, os, random, re, shutil
from PIL import Image

KELAS = ["backboard", "ball", "rim"]          # urutan id pada label sumber
SRC, OUT = "sumber_frame", "dataset_raw"
PAD = 0.15                                    # konteks tambahan: 15% lebar/tinggi kotak
FR = dict(train=0.70, valid=0.15, test=0.15)  # proporsi blok
GAP = 4                                       # frame dibuang di batas antar blok
TANGGAL, CAHAYA, SESI = "2026-09-24", "netral", "BRAIL-24Sep"
random.seed(0)

frames = sorted(((int(re.match(r"(\d+)_", f).group(1)), f) for f in os.listdir(f"{SRC}/images")))
N = len(frames)
a = int(N * FR["train"]); b = a + int(N * FR["valid"])
split_of = {}
for i, (n, f) in enumerate(frames):
    if i < a - GAP // 2: split_of[f] = "train"
    elif a + GAP // 2 <= i < b - GAP // 2: split_of[f] = "valid"
    elif i >= b + GAP // 2: split_of[f] = "test"       # selain itu: frame jeda, tidak dipakai

def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    return inter / ((p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - inter + 1e-9)

shutil.rmtree(OUT, ignore_errors=True)
for k in KELAS + ["latar"]:
    os.makedirs(f"{OUT}/{k}")
rows, skipped = [], 0
urut = {k: 0 for k in KELAS + ["latar"]}                 # penomoran per kelas (slide 19)
tgl = TANGGAL.replace("-", "")
def nama(k):
    urut[k] += 1
    return f"{k}/{k}_{tgl}_{CAHAYA}_{urut[k]:03d}.jpg"
for n, f in frames:
    if f not in split_of:
        skipped += 1; continue
    img = Image.open(f"{SRC}/images/{f}").convert("RGB"); W, H = img.size
    boxes = []
    for line in open(f"{SRC}/labels/{f.rsplit('.', 1)[0]}.txt"):
        p = line.split()
        if len(p) < 5: continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        boxes.append((c, (cx - w/2) * W, (cy - h/2) * H, (cx + w/2) * W, (cy + h/2) * H))
    for i, (c, x1, y1, x2, y2) in enumerate(boxes):
        px, py = max(4, PAD * (x2 - x1)), max(4, PAD * (y2 - y1))
        box = (max(0, round(x1 - px)), max(0, round(y1 - py)), min(W, round(x2 + px)), min(H, round(y2 + py)))
        name = nama(KELAS[c])
        img.crop(box).save(f"{OUT}/{name}", quality=95)
        rows.append([name, KELAS[c], split_of[f], n, f, "%d,%d,%d,%d" % box])
    # satu potongan latar per frame, ukuran mengikuti objek pada frame itu
    for _ in range(60):
        _, x1, y1, x2, y2 = random.choice(boxes)
        w = int((x2 - x1) * random.uniform(0.8, 1.6)) + 8; h = int((y2 - y1) * random.uniform(0.8, 1.6)) + 8
        if w >= W or h >= H: continue
        bx = random.randint(0, W - w); by = random.randint(0, H - h); box = (bx, by, bx + w, by + h)
        if all(iou(box, (q[1], q[2], q[3], q[4])) < 0.02 for q in boxes):
            name = nama("latar")
            img.crop(box).save(f"{OUT}/{name}", quality=95)
            rows.append([name, "latar", split_of[f], n, f, "%d,%d,%d,%d" % box]); break

with open(f"{OUT}/metadata.csv", "w", newline="") as o:
    w = csv.writer(o)
    w.writerow(["nama_file", "kelas", "split", "frame", "sumber_frame", "bbox_xyxy", "tanggal", "kondisi_cahaya", "sesi", "kamera"])
    for r in rows: w.writerow(r + [TANGGAL, CAHAYA, SESI, "e-con See3CAM_CU135 (kepala robot)"])

print("frame dipakai:", N - skipped, "| frame jeda dibuang:", skipped)
import collections
cnt = collections.Counter((r[1], r[2]) for r in rows)
print("%-10s %6s %6s %6s %6s" % ("kelas", "train", "valid", "test", "total"))
for k in KELAS + ["latar"]:
    t = [cnt[(k, s)] for s in ("train", "valid", "test")]
    print("%-10s %6d %6d %6d %6d" % (k, *t, sum(t)))
