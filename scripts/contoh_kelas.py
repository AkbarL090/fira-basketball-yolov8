"""Membuat docs/img/contoh_kelas.png: 6 contoh potongan per kelas (untuk dokumen desain)."""
import csv, os, random
from PIL import Image, ImageDraw

random.seed(1)
rows = list(csv.DictReader(open("dataset_raw/metadata.csv")))
S, N, L = 120, 6, 90
sheet = Image.new("RGB", (L + N * S, 4 * S), "white")
d = ImageDraw.Draw(sheet)
for r, k in enumerate(["backboard", "ball", "rim", "latar"]):
    fs = random.sample([x["nama_file"] for x in rows if x["kelas"] == k and x["split"] == "train"], N)
    d.text((6, r * S + S // 2 - 5), k, fill="black")
    for c, f in enumerate(fs):
        im = Image.open(os.path.join("dataset_raw", f)).convert("RGB"); im.thumbnail((S - 6, S - 6))
        sheet.paste(im, (L + c * S + 3, r * S + 3))
os.makedirs("docs/img", exist_ok=True)
sheet.save("docs/img/contoh_kelas.png")
