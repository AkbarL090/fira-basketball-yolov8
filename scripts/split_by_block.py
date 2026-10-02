"""Split ulang berbasis blok frame berurutan dengan jeda, agar frame bertetangga
tidak terbagi ke train dan val/test (data leakage, slide 22).
Hasil: dataset_session/ dan data_session.yaml. Penomoran frame diambil dari awalan nama file."""
import glob, os, re, shutil
SRC, DST = "dataset_raw", "dataset_session"
FR = dict(train=0.70, valid=0.15, test=0.15)   # proporsi blok
GAP = 4                                        # frame dibuang di batas antar blok

items = []
for split in ["train", "valid", "test"]:
    for img in glob.glob(f"{SRC}/{split}/images/*.jpg"):
        n = int(re.match(r"(\d+)_", os.path.basename(img)).group(1))
        lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        items.append((n, img, lbl))
items.sort()
N = len(items)
a = int(N * FR["train"]); b = a + int(N * FR["valid"])
blocks = {"train": items[:a - GAP // 2],
          "valid": items[a + GAP // 2: b - GAP // 2],
          "test":  items[b + GAP // 2:]}
shutil.rmtree(DST, ignore_errors=True)
for split, rows in blocks.items():
    os.makedirs(f"{DST}/{split}/images"); os.makedirs(f"{DST}/{split}/labels")
    for _, img, lbl in rows:
        shutil.copy(img, f"{DST}/{split}/images/"); shutil.copy(lbl, f"{DST}/{split}/labels/")
    print(split, len(rows), "citra, frame", rows[0][0], "-", rows[-1][0])
open("data_session.yaml", "w").write(
    f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    "names:\n  0: backboard\n  1: ball\n  2: rim\n")
