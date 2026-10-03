# Transfer Learning untuk Klasifikasi Objek Basket (backboard, ball, rim)

Tugas RET503 Computer Vision and Deep Learning, Pertemuan 3 (Transfer Learning dan Fine-Tuning).
Proyek: persepsi robot humanoid untuk **FIRA HuroCup 2027, kategori Basketball** (Barelang FC, Politeknik Negeri Batam).

**Model:** EfficientNet-B0 pretrained ImageNet (torchvision), salah satu dari lima model pada slide 14.
**Mode:** `feature`, `partial`, dan `scratch` dibandingkan (slide 21); mode terbaik ditetapkan di bagian Hasil.

## Isi repositori

```
├── README.md
├── docs/desain.md              # dokumen desain awal (template slide 18)
├── dataset_raw/                # dataset klasifikasi: <kelas>/*.jpg + metadata.csv
├── sumber_frame/               # frame asli + label YOLO (sumber potongan)
├── colab/jalankan_di_colab.ipynb
├── scripts/
│   ├── make_crops.py           # frame beranotasi -> dataset_raw (split blok anti-leakage)
│   ├── train.py                # 5 model x 3 mode
│   ├── report.py               # tabel + grafik per epoch + confusion matrix
│   ├── latency.py              # latensi pre/inferensi/post untuk 5 model
│   └── contoh_kelas.py
├── results/                    # history.csv, summary.json, tabel, grafik
└── requirements.txt
```

## Data

Frame diambil dari kamera robot (e-con See3CAM_CU135 di kepala humanoid) pada **24 September 2026 di lab BRAIL**, cahaya netral, di-capture tiap 0,5 detik saat robot dijalankan: **181 frame dari satu rekaman**. Frame dianotasi di Roboflow (tiga kotak: backboard, ball, rim) dalam format YOLO untuk proyek detektor berikutnya. Untuk tugas ini, setiap kotak dipotong (dengan konteks 15%) menjadi satu citra klasifikasi, dan satu potongan `latar` diambil dari area tanpa objek pada tiap frame.

| Kelas | Train | Valid | Test | Total |
|---|---|---|---|---|
| backboard | 109 | 20 | 25 | 154 |
| ball (bola tenis) | 115 | 15 | 26 | 156 |
| rim | 118 | 21 | 25 | 164 |
| latar | 124 | 23 | 26 | 173 |

Setiap kelas melebihi syarat minimal 50 citra. Metadata per citra (`nama_file`, `kelas`, `tanggal`, `kondisi_cahaya`, plus split, nomor frame, bbox, sesi, kamera) ada di `dataset_raw/metadata.csv`.

**Pencegahan data leakage (slide 22).** Frame berurutan sangat mirip, sehingga split acak membuat test terlihat terlalu bagus. Di sini frame diurutkan lalu dibagi menjadi tiga **blok berurutan** (70/15/15%) dengan jeda 4 frame di batas blok (8 frame dibuang), sehingga tidak ada frame train yang bertetangga langsung dengan frame valid atau test. Cara ini belum menggantikan pemisahan per sesi: seluruh data tetap berasal dari satu rekaman.

## Metode

Praproses seragam (slide 13): ubah ke 224 × 224, RGB, normalisasi mean/std ImageNet. Augmentasi train: RandomResizedCrop (skala 0,6-1,0), HorizontalFlip, ColorJitter. Adam, CosineAnnealingLR, 10 epoch, batch 32. Layer yang beku tetap berada pada mode `eval()` untuk BatchNorm (slide 11).

| Mode | Bobot awal | Yang dilatih | Learning rate |
|---|---|---|---|
| `feature` | ImageNet | head baru saja (`classifier[1]`) | 1e-3 |
| `partial` | ImageNet | `features[7:]` (blok akhir) + head | 1e-4 / 1e-3 |
| `scratch` | acak | semua | 1e-3 |

Bagian blok akhir pada model lain: ResNet `layer4`, MobileNetV3-Small `features[9:]`, MobileNetV3-Large `features[13:]`. Mode dijalankan dengan 3 seed (0, 1, 2) karena set valid dan test kecil.

Yang dicatat (slide 21): akurasi val terbaik, waktu pelatihan, dan epoch saat akurasi val ≥ 90% (pertama kali, dan epoch setelah mana tidak pernah turun lagi).

## Cara menjalankan

```bash
pip install -r requirements.txt
python scripts/make_crops.py                                   # opsional: membuat ulang dataset_raw
python scripts/train.py --model efficientnet_b0 --modes feature partial scratch --seeds 0 1 2
python scripts/report.py --model efficientnet_b0
python scripts/latency.py --device cuda                        # tulis nama perangkat pada laporan
```

Seluruh langkah juga tersedia di `colab/jalankan_di_colab.ipynb` (GPU T4). Opsi `--model all` melatih kelima model sebagai pembanding tambahan.

## Hasil

[isi: tabel dari results/ringkasan.md]

[isi: gambar results/akurasi_per_epoch_efficientnet_b0.png dan results/confusion_efficientnet_b0.png]

## Latensi

[isi: tabel dari scripts/latency.py, sebutkan perangkat dan presisi]

Catatan penerapan: model direncanakan berjalan di Jetson Xavier NX (TensorRT FP16) pada robot; pengujian di robot menyusul. Jumlah parameter pada tabel latensi memakai head 4 kelas, sehingga lebih kecil dari angka slide 14 (head 1000 kelas).

## Analisis

[isi]

## Keterbatasan

- Seluruh data dari satu sesi, satu lokasi, dan cahaya netral; hasil belum membuktikan kinerja di venue lain.
- Test set hanya sekitar 100 potongan, sehingga selisih beberapa persen antar mode berada dalam derau.
- Potongan `ball` kecil (median ≈ 26 × 33 piksel) dan diperbesar ke 224 × 224.
- Potongan `latar` diambil dari adegan yang sama; kelas ini relatif mudah dibedakan lewat warna.
- Ini klasifikasi potongan, bukan deteksi. Menemukan lokasi objek di frame penuh adalah pekerjaan proyek detektor (YOLO) berikutnya.
