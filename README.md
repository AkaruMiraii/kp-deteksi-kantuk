# KP Deteksi Indikasi Kantuk

Prototipe lokal Windows untuk mengamati landmark wajah, menghitung EAR/MAR, menampilkan status, dan menyimpan hasil sesi. Keputusan utama memakai penutupan mata beserta durasinya. Bukaan mulut merupakan informasi pendukung. Identitas sesi dan dukungan beberapa wajah masih eksperimental.

## Pengguna aplikasi portable

Paket siap jalan harus berupa ZIP aplikasi yang berisi `runtime/`, `app/`, dan `KP_Deteksi_Kantuk.bat`. Ekstrak seluruh ZIP, lalu jalankan BAT. Python tidak perlu dipasang untuk paket tersebut. Jangan jalankan dari dalam ZIP atau pisahkan launcher dari folder pendampingnya. Gunakan folder tanpa emoji. ZIP source dari tombol **Code > Download ZIP** bukan paket portable.

Belum ada aset Release baru yang diterbitkan oleh pembaruan source ini. Jangan menganggap Download ZIP source sebagai installer.

## Menjalankan source

Gunakan Python 3.11 pada Windows dan jalankan dari root repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-poc.txt
.\.venv\Scripts\python.exe kp_poc_entry.py
```

Model `models/face_landmarker.task` tetap diperlukan. Siapkan model pengenalan wajah OpenCV SFace yang kompatibel sebagai `models/sface.onnx` sebelum menjalankan entry point ini. Bobot SFace tidak ditambahkan ke commit ini. Jangan mengganti nama sembarang model ONNX menjadi SFace.

Untuk jalur satu wajah versi sebelumnya:

```powershell
.\.venv\Scripts\python.exe run.py --eye-alertness
```

## Penggunaan

Pilih indeks kamera, lalu **Mulai Kamera**. Q untuk keluar monitor, H untuk rincian, C untuk kalibrasi peserta pada monitor prototipe. Launcher memakai DirectShow untuk sumber kamera Windows. Mode default menjalankan threshold dan identitas sesi tanpa pemeriksaan keaslian wajah. Model anti-spoof tidak dibutuhkan untuk finalisasi mode threshold.

Output monitor prototipe berada di `outputs/multiperson/`. Source tidak menyertakan hasil sesi, rekaman kamera, embedding identitas, atau log pribadi.

## Verifikasi paket pada PC Windows

Pada pengujian 10 November 2026 di PC DESKTOP-O86AT10, runtime bawaan berhasil membuka kamera, memproses 139 frame, dan selesai tanpa kesalahan cleanup. Pengujian terpisah membuka kamera kembali. Model dan replay video juga dijalankan. Screenshot yang diperiksa menunjukkan jendela pengaturan, bukan bukti validasi kejadian kantuk pada wajah nyata.

Path dengan emoji belum didukung. Validasi kejadian kantuk pada wajah nyata dan screenshot monitor langsung belum selesai. Pengujian eksekusi bukan ukuran akurasi deteksi.

## Isi repository

Hanya source runtime yang diperlukan, dependensi, lisensi vendor, dan tes regresi terarah. Runtime Python portable, ZIP distribusi, virtual environment, build, rekaman, hasil deteksi, dan log tidak dimasukkan ke commit. Aset siap pakai didistribusikan terpisah melalui GitHub Releases setelah pemeriksaan distribusi.
