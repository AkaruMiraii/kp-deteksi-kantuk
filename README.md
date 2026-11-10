# KP Deteksi Kantuk: Proof of Concept

Source aplikasi lokal deteksi indikasi kantuk dengan EAR/MAR, kalibrasi per peserta,
dan identitas sementara per sesi untuk sampai enam wajah. Mode POC menggunakan
threshold dan tidak mengaktifkan pemeriksaan keaslian wajah. Jalur satu wajah
lama melalui `run.py` tetap dipertahankan.

## Menjalankan source

Gunakan CPython 3.11 x64 dengan Tk pada Windows dan dependensi berikut:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe kp_poc_entry.py
```

`models/face_landmarker.task` dan `models/smart_yawn.json` tetap tersedia di repo.
Tambahkan `models/sface.onnx` dari aset model paket portable yang telah disiapkan.
SHA-256 model yang diterima dicatat di `tools/assemble_portable.py`. Model biner
SFace dan runtime tidak diunggah dalam pembaruan source ini.

```powershell
.\.venv\Scripts\python.exe kp_poc_entry.py --self-test
.\.venv\Scripts\python.exe kp_poc_entry.py --threshold-only --session-identity --max-seconds 30
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

GUI mengaktifkan mode threshold dan identitas sesi secara bawaan. Untuk CLI,
sertakan kedua flag tersebut. Q keluar, H membuka rincian, C mengalibrasi peserta.
Output berada di `outputs/multiperson/<session>/`. Rekaman kamera tidak dibuat
kecuali opsi `--record-video` atau `--save-preview` diberikan. Output dapat memuat
informasi peserta dan tidak boleh dimasukkan ke Git.

## Membentuk folder portable Windows x64

Tool assembly menyalin source dan aset ke **folder baru**. Tidak mengunduh atau
memasang dependensi, tidak mengubah runtime asal, dan menolak output yang sudah
ada. Siapkan direktori CPython 3.11 x64 mandiri dengan `python.exe`, `pythonw.exe`,
Tk/Tcl, serta library sesuai `requirements.txt`. Interpreter embedded minimal
tanpa Tk dan library belum cukup. Tool ini menerima runtime yang sudah siap,
bukan membangun distribusi Python dari nol.

```powershell
python tools/assemble_portable.py --runtime-python "C:\prepared\runtime\python" --models "C:\prepared\app\models" --output "C:\packages\KP_POC_Windows_x64_new"
```

Hasilnya berisi `app/`, `runtime/python/`, launcher relatif, panduan, dan manifest
SHA-256. Launcher membawa runtime sendiri, membersihkan variabel Python luar,
dan tidak bergantung pada folder build atau Python global. Paketkan seluruh
folder tersebut untuk pemindahan, bukan launcher saja.

Sebelum distribusi, jalankan `KP_Deteksi_Kantuk.bat --self-test`, unit test dari
folder `app`, uji kamera dan buka ulang, lalu ulangi setelah memindahkan folder.
Manifest hanya mencakup file hasil assembly sebelum penggunaan, bukan output sesi.

## Lingkup repository

Pembaruan ini hanya source, launcher, dependency pin, test, tool assembly, dan
panduan. Runtime, build, ZIP, rekaman, embedding peserta, output, log, dan kredensial
tidak disertakan. Tidak ada ZIP release yang diunggah. Algoritme lama tetap tersedia
untuk kompatibilitas, termasuk eksperimen kandidat menguap. Keputusan POC bukan
diagnosis tahap tidur klinis.
