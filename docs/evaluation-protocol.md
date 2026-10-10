# Protokol anotasi, kalibrasi dan evaluasi KP

Versi: rancangan operasional v1, disiapkan sebelum rekaman evaluasi baru. Bukan hasil evaluasi. Master/pembimbing perlu meninjau definisi target dan jumlah partisipan sebelum pengujian akhir. Ruang lingkup: satu wajah, kondisi terkontrol, indikator visual bukan diagnosis.

## 1. Bukti sumber dan persetujuan

Gunakan video mentah tanpa EAR/MAR/label prediksi, dengan frame_id dan waktu sumber yang sinkron ke log. Saat ini program hanya merekam snapshot periodik, belum video sumber sinkron; fitur itu dan playback offline perlu ditambahkan sebelum anotasi seluruh frame. Jangan mengisi frame yang tidak ada gambarnya dengan tebakan dari log fitur.

Penyimpanan video opt-in setelah persetujuan; tanpa audio secara default. Gunakan participant_id pseudonim, session_id unik, akses lokal terbatas. Video/log wajah tidak masuk Git. Tetapkan retensi/penghapusan sebelum merekam partisipan lain; jangan menghapus data tanpa persetujuan. Catat asal, hash video, resolusi, waktu, perangkat, model dan versi kode. Rekaman lama adalah development/smoke test, bukan held-out test.

## 2. Rubrik independen dari prediksi

Annotator melihat video mentah, tidak melihat nilai EAR/MAR, output timer, status program atau tombol Activity. Catat onset/offset dengan frame_id/waktu video dan jenis perilaku. Anotasi dapat diedit sebelum label dikunci; perubahan setelah membaca hasil test harus dicatat, bukan dipakai menyesuaikan hasil diam-diam.

- Mata: open, closed_both, unilateral_closed, uncertain, not_visible. Awal closed_both adalah frame pertama kedua kelopak tampak tertutup; akhir pada pembukaan yang terlihat. Jika kontur tidak dapat dibedakan karena tangan, objek, pantulan atau blur, pilih uncertain/not_visible, jangan closed.
- Mulut: relaxed, speaking, deliberate_opening, yawn_like, uncertain, not_visible. Bukaan sengaja tidak diubah menjadi menguap hanya karena besar atau lama. yawn_like memerlukan pola visual pembukaan bertahap, puncak bukaan luas dan penutupan dalam rangkaian video; jika tidak dapat dibedakan dari bukaan sengaja/ucapan, pilih uncertain. Status yawn_like adalah penilaian visual, bukan bukti fisiologis mengantuk.
- Visibility: eyes_visible dan mouth_visible secara terpisah; reason eye_occlusion, mouth_occlusion, face_absent, blur, extreme_pose atau other. Target gabungan hanya dapat dianotasi pasti bila semua bagian yang dibutuhkan terlihat.
- Label skenario pengambilan data dicatat terpisah, membantu provenance tetapi bukan penentu label setiap frame.

Untuk target per-frame mata, definisi operasional yang diusulkan: indikator penutupan berkepanjangan positif mulai setelah 1,0 detik closed_both kontinu, sampai mata dibuka. Nilai 1,0 detik adalah konvensi anotasi awal, bukan threshold detektor dan bukan batas medis; perlu disetujui/dibekukan sebelum test. Anotasi onset/offset mentah tetap disimpan agar aturan ini dapat diaudit. Laporkan pula event recall dan latensi onset karena detektor memiliki durasi tunggu sendiri.

Untuk target mulut, label positif adalah interval yawn_like yang jelas sejak onset sampai offset; speaking/deliberate_opening/relaxed negatif, ambiguous tidak dipaksa negatif. Sebelum gabungan final, tetapkan secara eksplisit apakah positif = indikator mata ATAU yawn_like; ini target label manusia, tidak otomatis mengizinkan OR prolonged_opening pada runtime.

Rubrik pilot diperiksa oleh dua annotator bila tersedia, tanpa output model. Adjudikasi perbedaan dan catat agreement. Jika hanya satu annotator tersedia, laporkan keterbatasan dan lakukan review kedua pada subset bila mungkin. Jangan mengarang jumlah annotator/partisipan.

## 3. Skema berkas yang akan digunakan

Manifest rekaman: session_id,participant_id,source_path,source_sha256,consent_reference,split,scenario,frame_count,frame_width,frame_height,timebase.

Anotasi per-frame: session_id,frame_id,timestamp_ms,eyes_state,mouth_behavior,eyes_visible,mouth_visible,reference_eye_positive,reference_mouth_positive,reference_combined_positive,annotation_reason,annotator_id,rubric_version.

Boolean reference hanya diisi untuk bagian yang dapat dinilai; gunakan nilai kosong untuk unknown, bukan 0. Penilaian tidak dapat dinilai manusia dipisahkan dari abstention detektor.

Prediksi menyimpan field program yang sudah ada termasuk policy/config. Input evaluasi harus cocok session_id/frame_id, jumlah frame, timestamp dan hash sumber; mismatch harus error, bukan silently skip. Jangan menyalin Activity ke reference labels.

## 4. Skenario dan split

Development mencakup mata terbuka, kedip biasa, penutupan sengaja singkat/panjang, penutupan unilateral, bibir santai, berbicara bervariasi, bukaan sengaja, yawn_like yang dapat diamati, wajah hilang, tangan/objek menutupi mata/mulut, serta perubahan pose/pencahayaan terkontrol. Tidak memaksa kantuk nyata atau mengurangi tidur. Jika tidak tersedia contoh menguap yang dapat dinilai, klaim kanal menguap tetap belum terpenuhi.

Tentukan jumlah rekaman/partisipan berdasarkan pelaksanaan nyata. Split per partisipan bila cukup; jika satu orang, split per sesi berbeda dan batasi klaim ke pengujian intra-subjek. Jangan random split frame bersebelahan. Rekaman yang digunakan dalam percakapan ini tidak boleh menjadi held-out test baru.

Cadangkan sesi held-out sebelum kalibrasi; gunakan misalnya 70% sesi development, 30% test dengan seluruh kelas cukup terwakili. Jika data kecil dan kelas test kosong, kumpulkan data tambahan, jangan menganggap metrik itu representatif.

## 5. Kalibrasi dan freeze

Pilih threshold/durasi kanal pada development berdasarkan label independen. Catat daftar konfigurasi yang dicoba dan pemilihan; optimasi F1 dengan pemeriksaan false-positive speaking/deliberate_opening dan coverage. Jangan mengatasi hasil buruk dengan menyembunyikan kelas/rekaman. Jika threshold/durasi MAR gagal membedakan menguap, diagnosis satu jalur lalu perluas fitur/aturan dalam scope atau minta persetujuan perubahan scope.

Setelah dipilih, simpan frozen config, hash config/model/code, rubric_version dan manifest split. Bekukan fusion rule yang telah divalidasi pada development. Held-out test dijalankan tanpa perubahan parameter/rubrik. Jika diubah setelah melihat test, test itu menjadi development dan diperlukan holdout baru.

## 6. Metrik dan coverage

Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). Jika penyebut nol, laporkan undefined disertai support, bukan angka sempurna. Confusion matrix membandingkan reference independen dan prediksi pada frame reference valid serta prediksi dapat dinilai. Laporan mencakup mata, mulut dan gabungan hanya sejauh endpoint benar-benar diimplementasikan/validasi; policy v2 eye-only bukan endpoint gabungan akhir.

Frame prediksi tidak_dapat_dinilai tidak dipaksa menjadi TN. Laporkan total frame, jumlah reference valid/unknown, jumlah prediksi evaluable/abstained, coverage = evaluable predictions / reference-valid frames, dan abstention pada reference positif/negatif secara terpisah. Oklusi yang reference unknown dilaporkan sebagai quality challenge, bukan TN. Laporkan kegagalan quality gate ketika sistem mengeluarkan angka padahal manusia tidak dapat melihat fitur. Sertakan event detection dan latensi untuk membedakan interval tunggu dari missed event.

## 7. Performa dan acceptance

processing_ms memakai perf_counter_ns dan lingkupnya dinyatakan: RGB, inferensi, geometri, timer, keputusan dan overlay, di luar camera read/disk/GUI. Laporkan distribusi dan pisahkan face-present/no-face. FPS keseluruhan = jumlah frame selesai / elapsed time loop yang diukur, bukan 1000/median_ms. Catat apakah recording aktif, frame drops, perangkat CPU/RAM/OS, dan jangan menganggap resolusi clock sebagai akurasi total.

Acceptance akhir membutuhkan label dan split terkunci, konfigurasi freeze, hasil test nyata semua endpoint yang diklaim, coverage dan batasan, FPS/timing, tes otomatis dan dokumentasi BAB IV. Tidak ada target angka akurasi yang dibuat tanpa keputusan pembimbing; prototipe runnable adalah gate terpisah dari acceptance akhir.
