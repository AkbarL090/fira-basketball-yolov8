# Dokumen Desain Awal: Deteksi Objek Basket untuk FIRA HuroCup 2027

Mata kuliah RET503, Pertemuan 3. Bagian bertanda **[isi]** dilengkapi setelah pengukuran.

## 1. Tujuan dan misi robot
Robot humanoid pada kategori Basketball FIRA HuroCup 2027 harus mengenali **bola**, **ring (rim)**, dan **papan pantul (backboard)** dari kamera onboard sebagai dasar menentukan posisi, arah, dan keputusan lemparan. Keluaran perception: kelas, kotak pembatas, dan skor kepercayaan per frame.

## 2. Spesifikasi tugas
| Item | Isi |
|---|---|
| Tugas | Deteksi objek (3 kelas: `backboard`, `ball`, `rim`) |
| Masukan | Citra RGB dari kamera e-con See3CAM_CU135 (sensor 1080p; dipakai 1280×720 MJPG, 30 fps, dapat 60 fps), dipasang di kepala robot humanoid, tinggi ≈ 1 m dari lantai; diubah ke 640×640 (RGB, skala 0-1) |
| Keluaran | Kotak + kelas + skor; kemudian dipakai modul strategi |
| Kondisi lapangan | Jarak lempar 1-1,5 m; kamera ikut bergerak bersama kepala saat robot berjalan (motion blur, sudut berubah); bola bergerak; pencahayaan lab/hall |
| Target kinerja | mAP50 ≥ 0,90 pada test set yang dipisah per sesi; ≥ 15 FPS pada perangkat target (anggaran 67 ms per frame) |

## 3. Kandidat model dan alasan
| Kandidat | Parameter (≈) | Alasan |
|---|---|---|
| YOLOv8n | 3,2 juta | Paling ringan; kandidat bila YOLOv8s tidak mencapai 15 FPS |
| YOLOv8s | 11,2 juta | Akurasi lebih baik untuk objek kecil (bola jauh); sudah dilatih dan diekspor ONNX |

Keduanya tersedia bobot pretrained COCO, mendukung ekspor ONNX/TensorRT, dan satu keluarga sehingga pipeline data sama. Pemilihan akhir berdasarkan akurasi dan latensi hasil `scripts/latency.py`; perangkat komputasi robot direncanakan NVIDIA Jetson.

## 4. Strategi transfer learning
Bobot awal COCO (kelas `sports ball` mirip dengan bola basket). Empat strategi dibandingkan dengan setelan yang sama: `feature` (`freeze=10`), `partial` (`freeze=7`), `full` (tanpa freeze), dan `scratch` (tanpa pretrained). Satu LR (`lr0=0,00143`, AdamW), 100 epoch, seed 0. Hipotesis: dataset kecil (181 citra) sehingga mode pretrained lebih cepat konvergen dan `scratch` paling lambat.

## 5. Rencana data
- Sumber saat ini: 181 citra (ekspor Roboflow v1) anotasi sendiri, 3 kelas, semua kelas > 50 citra. Diambil 24 September 2026 di lab BRAIL, cahaya netral, dari kamera robot yang dijalankan dan di-capture tiap 0,5 detik (satu rekaman). Kelas `ball` adalah bola tenis.
- Kelemahan: seluruhnya satu rekaman (frame 1530-1710) dengan split acak, berisiko leakage.
- Rencana: split berbasis blok frame berjeda (`scripts/split_by_block.py`), lalu menambah rekaman dari sesi/lokasi/cahaya berbeda, khusus untuk test set.
- Metadata per citra (`dataset_raw/metadata.csv`): nama file, split, kelas, jumlah kotak, tanggal, kondisi cahaya, sesi.

## 6. Metrik evaluasi
mAP50 dan mAP50-95 (val dan test), epoch ke mAP50 ≥ 0,9, waktu latih, confusion matrix, serta latensi (pre/inferensi/post) dan FPS pada perangkat target.

## 7. Risiko dan mitigasi
| Risiko | Mitigasi |
|---|---|
| Data leakage (frame berurutan terbagi ke train dan val/test) | Split berbasis blok frame; test dari sesi baru |
| Bola kecil, jauh, atau blur saat robot bergerak | Tambah data kondisi jarak jauh dan blur; pertimbangkan `imgsz` lebih besar dan augmentasi blur |
| Perubahan pencahayaan dan latar venue berbeda | Rekam di beberapa lokasi dan jam; augmentasi HSV |
| Latensi melebihi 67 ms | Gunakan YOLOv8n; ekspor ke format teroptimasi (ONNX/TensorRT) |
| Rasio aspek citra latih berbeda dari kamera (resize stretch 640×480 pada ekspor Roboflow) | Ekspor ulang tanpa stretch (resize Fit) dan latih dengan `imgsz=640` |
| Dataset kecil, hasil tidak stabil antar run | Laporkan beberapa seed bila waktu cukup |
