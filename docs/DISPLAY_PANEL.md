# Tampilan indikator smart_live

Status: VERIFIED untuk rendering dan tes otomatis, bukan verifikasi akurasi deteksi.

- Preview kamera dipertahankan pada resolusi asli tanpa tulisan, landmark atau status yang menutup wajah.
- Panel di kanan: status mata utama + durasi, kandidat menguap + skor/bar dan durasi, EAR kanan/kiri, MAR, waktu proses dan batas EAR.
- Footer: R kalibrasi ulang, Q keluar, keterangan indikator eksperimental.
- Model, threshold, tracker dan keputusan tidak diubah. Display memakai frame asli, bukan overlay internal.
- Perintah tetap python -m scripts.smart_live --eye-alertness. Tutup instance lama dengan Q lalu jalankan ulang untuk memuat kode baru.
- Pada input640x480, panel1108x600. Tidak ada adaptive resize pada window kecil; tampilan butuh ruang itu. Tidak memotong/mengecilkan wajah untuk memaksa panel masuk.
- Timing log mencakup rendering dashboard. Angka ms pada panel diambil sesaat sebelum rendering panel, sehingga sedikit lebih rendah daripada log total, bukan FPS keseluruhan.
- 75 automated tests pass; test membuktikan piksel kamera tidak berubah dan semua drawing teks berada di luar area kamera. aislop explicit4 file clean, formatter/linter skipped; scan changes scratch0 file tidak digunakan sebagai bukti utama.
- Smoke test nyata menggunakan gambar mentah pilot, detector/model actual, exit0. Screenshot panel diperiksa, teks utama legible, tidak clipped. Tidak menyalakan webcam via SSH.
- Files: scripts/smart_display.py, scripts/smart_live.py, tests/test_smart_display.py, tests/test_smart_live.py. Backup source sebelumnya berada di outputs/_diagnostic_verification/smart_live_before_panel_20261008_094330.py. Hash source/tests cocok di laptop.
