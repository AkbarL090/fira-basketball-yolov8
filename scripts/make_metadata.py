"""Membuat dataset_raw/metadata.csv. Kolom tanggal, kondisi_cahaya, sesi diisi manual."""
import csv, glob, os
NAMES = ["backboard", "ball", "rim"]
ROOT = "dataset_raw"
rows = []
for split in ["train", "valid", "test"]:
    for f in sorted(glob.glob(f"{ROOT}/{split}/labels/*.txt")):
        lines = [l.split() for l in open(f) if l.strip()]
        cls = sorted({NAMES[int(l[0])] for l in lines})
        rows.append([os.path.basename(f).replace(".txt", ".jpg"), split,
                     ";".join(cls), len(lines), "", "", ""])
with open(f"{ROOT}/metadata.csv", "w", newline="") as o:
    w = csv.writer(o)
    w.writerow(["nama_file", "split", "kelas", "jumlah_kotak", "tanggal", "kondisi_cahaya", "sesi"])
    w.writerows(rows)
print(len(rows), "baris ditulis ke", f"{ROOT}/metadata.csv")
