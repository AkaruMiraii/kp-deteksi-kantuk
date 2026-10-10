# KP Deteksi Indikasi Kantuk

Prototipe lokal untuk satu wajah dengan OpenCV dan MediaPipe Face Landmarker. Fitur utama: pengukuran EAR kedua mata, MAR mulut, pelacakan durasi, keputusan indikasi kantuk berbasis mata, diagnostik, perekaman opt-in, replay lossless, anotasi interval, serta dashboard dan model kandidat menguap eksperimental.

## Struktur

```text
Project KP/
  scripts/               Modul deteksi, CLI, replay, anotasi, dan training
  tests/                 Unit test dan integration test
  models/                Face Landmarker dan koefisien smart_yawn.json
  docs/                  Spesifikasi, evaluasi, eksperimen, materi belajar
  README.md              Panduan utama
  DIAGNOSTICS.md         Panduan diagnostik terperinci
  requirements.txt       Dependency runtime yang dipin
  requirements-dev.txt   Dependency quality checks
  pyproject.toml         Konfigurasi lint dan formatter
  outputs/               Data lokal dan hasil sesi, tidak masuk Git
  .venv/                 Environment laptop, tidak masuk Git
```

Tidak ada rekaman partisipan, snapshot wajah, cache browser, atau environment Python yang diunggah ke repository. Materi `docs/learning-kp/` tetap disimpan karena terkait pembelajaran EAR dan proyek ini.

## Setup Windows

Gunakan Python 3.11. Environment laptop saat cleanup menggunakan Python 3.11.16, MediaPipe 1.0.1, OpenCV 5.0.0 dan NumPy 2.4.6. Pin paket berada di `requirements.txt`.

Untuk clone baru, jalankan dari folder proyek:

```powershell
uv venv --python 3.11 .venv
uv pip install --link-mode=copy --python .venv\Scripts\python.exe -r requirements.txt
uv pip check --python .venv\Scripts\python.exe
```

Jangan membuat ulang `.venv` yang sedang digunakan. Pilih `.venv\Scripts\python.exe` lewat Python: Select Interpreter di VS Code. Aktivasi environment dan perubahan execution policy tidak diperlukan.

## Menjalankan deteksi utama

```powershell
.\.venv\Scripts\python.exe -m scripts.landmark_diagnostics --eye-threshold 0.12 --eye-duration 1.5 --eye-max-gap 0.5 --mouth-threshold 0.5 --mouth-duration 2 --mouth-max-gap 0.5
```

Threshold di atas adalah parameter prototipe, bukan nilai universal tervalidasi. Tanpa parameter kanal mata, keputusan kantuk tidak aktif. Penutupan bilateral yang cukup lama memberi `indikasi_kantuk`. Bukaan mulut lama hanya observasi, bukan bukti kantuk. Geometri invalid atau wajah hilang menghasilkan `tidak_dapat_dinilai`.

Mode utama tidak menyimpan rekaman secara default. Tambahkan `--record` untuk log dan snapshot, atau `--record-video` untuk video raw beserta log. Baca [DIAGNOSTICS.md](DIAGNOSTICS.md) sebelum merekam.

Pemeriksaan webcam dan landmark sederhana:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_webcam
.\.venv\Scripts\python.exe -m scripts.check_landmarks
.\.venv\Scripts\python.exe -m scripts.check_landmarks_video
```

Perintah kamera dijalankan dari desktop laptop, bukan otomatis saat import modul.

## Dashboard eksperimental

```powershell
.\.venv\Scripts\python.exe -m scripts.smart_live --eye-alertness
.\.venv\Scripts\python.exe -m scripts.smart_live --input-image C:\path\image.png --eye-alertness
.\.venv\Scripts\python.exe -m scripts.smart_live --model outputs\training\smart_yawn_v1\smart_yawn.json --session outputs\diagnostics\NAMA_SESI
```

Default model kini `models/smart_yawn.json`, sehingga clone baru tidak bergantung pada sesi lokal di `outputs/`. Koefisiennya identik dengan model eksperimen yang sudah ada. `--model` tetap dapat menunjuk hasil training baru. Smart mode menulis CSV, metadata dan satu preview ke `outputs/smart_yawn/`, termasuk pada webcam. Jangan gunakan mode ini jika tidak ingin menyimpan preview.

Kandidat menguap tidak mengubah keputusan mata menjadi positif. Label training masih draft visual AI; skor bukan probabilitas terkalibrasi. Tier mata adaptif adalah tampilan eksperimental, bukan tahap tidur klinis. Lihat [smart yawn](docs/SMART_YAWN.md), [eye v2](docs/DATASETS_AND_EYE_V2.md), dan [protokol evaluasi](docs/evaluation-protocol.md).

## Pengujian dan clean code

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
uv tool run ruff@0.17.0 check scripts tests
uv tool run ruff@0.17.0 format scripts tests --check
```

`requirements-dev.txt` menyediakan versi Ruff yang sama jika ingin memasangnya ke environment proyek. Formatter dan lint mengecek import, unused names, syntax, serta konsistensi format. Test mencakup durasi, keputusan, geometri, log/snapshot, video raw, replay, dashboard, model causal, dan keamanan import CLI. Webcam nyata harus dijalankan dari sesi desktop; unit test dan replay gambar tidak mengklaim validasi akurasi klinis.

## Model dan data

`models/face_landmarker.task` adalah model MediaPipe yang sudah digunakan proyek. SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. Sumber upstream: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

Model smart yawn memakai koefisien JSON, bukan pickle/joblib. Data training, anotasi partisipan, hasil eksperimen, dan rekaman tetap lokal serta diabaikan Git. Model dan dokumentasi eksperimen tidak otomatis membuktikan akurasi pada subjek baru.
