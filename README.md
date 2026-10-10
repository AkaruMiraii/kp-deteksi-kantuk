# KP Deteksi Kantuk: Proof of Concept

Duplikat runtime-only dari proyek lengkap. Tampilan dashboard dan parameter deteksi sama. Tidak berisi dataset, anotasi, materi belajar, laporan training, alat training, unit test, rekaman lama, cache atau riwayat Git. Model inferensi tetap diperlukan dan disertakan: Face Landmarker dan koefisien JSON kandidat menguap. Koefisien bukan data training mentah.

## Setup pada clone/folder baru

Gunakan PowerShell dari folder ini:

```powershell
uv venv --python 3.11 .venv
uv pip install --link-mode=copy --python .venv\Scripts\python.exe -r requirements.txt
```

## Dashboard dan perekaman

```powershell
.\.venv\Scripts\python.exe run.py --eye-alertness
.\.venv\Scripts\python.exe run.py --eye-alertness --record-video
```

Kalibrasi dengan mata terbuka alami selama 5 detik. R untuk kalibrasi ulang, Q untuk selesai dan finalisasi file. Output setiap sesi berada di `outputs/smart_yawn/`. Perintah tanpa `--record-video` tidak merekam video penuh, tetapi tetap menyimpan log prediksi dan satu preview. Perekaman video tanpa audio mencakup raw kamera serta dashboard; timestamp CSV adalah acuan waktu rekaman, bukan FPS container video. FFV1 lossless dapat membutuhkan ruang disk besar dan finalisasi memverifikasi ulang file.

## Lingkup

Program lokal satu wajah, bukan detektor tahap tidur klinis. Kandidat menguap merupakan eksperimen dengan draft label; skor tidak terkalibrasi dan tidak mengubah keputusan mata. Versi ini untuk demonstrasi runnable, bukan sarana training. Model dan modul yang dibutuhkan jalur runtime dipertahankan; dependency training scikit-learn tidak diperlukan. Folder `.venv` dibuat khusus untuk PoC, tidak berbagi environment proyek penelitian.
