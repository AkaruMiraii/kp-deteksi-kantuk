# Prototipe KP Deteksi Indikasi Kantuk

Prototipe lokal EAR/MAR dengan pengaturan ambang, pelacakan durasi, dan identitas sesi. Branch ini memuat source kandidat yang digunakan untuk paket executable portable. Branch utama tidak diganti otomatis.

## Jalur demonstrasi

Paket portable memakai mode threshold dan identitas sesi. Pemeriksaan keaslian tidak aktif. Pengaturan kapasitas beberapa wajah tersedia; pengujian yang dilaporkan memakai subjek bergantian dengan satu wajah per bingkai. Pengamatan identitas yang meragukan ditahan. MAR adalah pengamatan pendukung, bukan pemicu mandiri kantuk.

## Menjalankan source di Windows

```powershell
uv venv --python 3.11 .venv-poc
uv pip install --python .venv-poc\Scripts\python.exe -r requirements-poc.txt
.\.venv-poc\Scripts\python.exe kp_poc_entry.py
```

Launcher membuka kamera hanya setelah tombol Mulai Kamera ditekan. Q untuk keluar monitor, H untuk rincian, ? untuk bantuan. Parameter awal EAR 0,12, MAR 0,50, durasi penutupan mata 1,5 detik, pendaftaran identitas 180 detik. Identitas berlaku dalam RAM selama sesi. Log berada di outputs/multiperson pada source, atau _internal/outputs/multiperson pada paket portable.

## Model

Face Landmarker yang sudah ada dipertahankan. Jalur identitas memerlukan `models/sface.onnx`, berasal dari OpenCV Zoo: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface . Tinjau lisensi dan gunakan artefak model resmi sebelum menjalankan. Model tambahan tidak diunggah dalam perubahan ini. Referensi hash model dan versi runtime dapat dicatat dalam manifest paket lokal. Rekaman wajah, embedding, dan log pengguna tidak disertakan.

## Pengujian dan build

```powershell
.\.venv-poc\Scripts\python.exe -m unittest discover -s tests
uv pip install --python .venv-poc\Scripts\python.exe -r requirements-build.txt
.\.venv-poc\Scripts\python.exe -m PyInstaller --noconfirm KP_Deteksi_Kantuk.spec
Copy-Item models dist\KP_Deteksi_Kantuk\models -Recurse -Force
.\dist\KP_Deteksi_Kantuk\KP_Deteksi_Kantuk.exe --self-test
```

Self-test memuat model dan citra netral tanpa membuka kamera. Dependency anti-spoof tidak disertakan dalam paket mode threshold. Modul eksperimental lain bukan jaminan semua mode tersedia pada executable ini. Perintah run.py historis tetap dipertahankan, tetapi bukan entry point paket terbaru.

## Bukti dan batas verifikasi

Pada snapshot laptop: 171 eksekusi tes lulus, executable memuat model MediaPipe dan SFace, smoke video memproses 356 bingkai dengan exit 0 dan tanpa cleanup error. Launcher diperiksa visual. Webcam langsung pada executable belum diuji. Kelulusan tes bukan akurasi manusia atau keberhasilan pengenalan seluruh peserta. Indikator ini bukan diagnosis medis.

Source, tes, launcher, dan spec dikomit. build/, dist/, virtual environment, video, log pribadi, dan ZIP executable tidak dikomit. Model/artefak release hanya boleh didistribusikan setelah pemeriksaan lisensi. Program kerja asli dan laporan tidak diubah.
