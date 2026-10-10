# Training smart yawn v1: hasil nyata

## Hasil

Training LogisticRegression class-balanced + StandardScaler selesai pada laptop Windows. Inferensi live memakai koefisien JSON, bukan pickle/joblib, tanpa kebutuhan scikit-learn pada jalur inferensi. Fitur causal: MAR sekarang, rerata EAR, selisih EAR, rentang/rerata/std MAR selama 1,5 detik terakhir, rerata EAR window, slope MAR, dan elapsed window. Tidak mengakses frame masa depan. Invalid/gap lebih dari0,5 detik reset history/timer. Skor >=0,5 perlu bertahan0,4 detik untuk status yawn_candidate. Angka skor bukan probabilitas terkalibrasi.

## Data dan target

Rekaman 20261007_143428_b0eec45b: 4.898 frame, empat giliran subjek utama A-D. Target adalah kandidat gerakan menyerupai menguap dari draft anotasi visual AI, BUKAN menguap fisiologis atau kantuk. Label unknown, occlusion dan batas15 frame tiap ujung dibuang dari fitting, bukan dibuat negatif. Window masih memakai history sebelumnya sesuai perilaku causal. Pengambilan sampel sekitar10Hz. Dataset fit berisi320 baris:134 positif kandidat,186 negatif eksplisit. Negatif hanya contoh yang diberi label small/closed/moderate; belum mencakup cukup berbicara dan deliberate wide opening sebagai hard negatives.

Pembagian leave-one-subject-segment-out: model dilatih pada3 segmen dan memprediksi segmen tersisa. StandardScaler hanya di-fit pada train setiap fold. Tidak random split frame tetangga. C=1, score threshold0,5, dan fitur ditetapkan sebelum hasil fold, bukan diubah untuk mempercantik angka. Segmen D tidak mempunyai negatif yang lolos labeling, sehingga false-positive D tidak dapat diukur.

## Hasil cross-validation pada draft (sebelum penahan durasi live)

| Metode | Precision | Recall | F1 |
|---|---:|---:|---:|
| Model EAR/MAR temporal | 83,46% | 82,84% | 83,15% |
| Pembanding MAR >0,5 saja | 98,91% | 67,91% | 80,53% |

Matriks model [[TN164,FP22],[FN23,TP111]]. Model menaikkan recall namun meningkatkan false-positive. F1 per segmen: A89,32%, B80,00%, C43,48%, D95,92% (D positif saja). Ini bukan akurasi final dan tidak berlaku otomatis pada semua orang.

## Integrasi dan pengujian

Mode baru `python -m scripts.smart_live`. Perintah lama tidak berubah. Smart output terpisah dari final_status aturan mata v2, tidak mengubah pemicu kantuk menjadi positif dari ML. Default-off karena entrypoint terpisah. Output CSV per-frame, JSON sesi, dan preview di outputs/smart_yawn/. Model artifacts di outputs/training/smart_yawn_v1/.

66 automated tests pass, uv pip check kompatibel, aislop explicit scan6 file bersih (format/lint engines tidak tersedia pada scan tersebut). Model dijalankan ulang terhadap seluruh4.898 frame video lossless melalui read_raw_session dan timestamp sumber; exit0, source_complete. Ini in-sample runnable demo karena artifact akhir di-refit menggunakan seluruh data berlabel, bukan pengujian holdout tambahan. Hasil replay:2565 no_candidate,402 pending,1288 yawn_candidate,643 not_assessable. Model MAINTAINS conservative existing geometry gate, bukan mendeteksi tangan secara otomatis; occlusion parsial bisa lolos dan menghasilkan false positive.

Live webcam belum dijalankan oleh user. Waktu median replay 9.3720 ms; bukan FPS akhir. Kamera tidak diaktifkan via SSH. Rollback perilaku: gunakan perintah lama landmark_diagnostics tanpa menjalankan smart_live. Tidak ada core Hermes, model wajah, source rekaman maupun anotasi yang diubah. Requirements menambah scikit-learn beserta dependensi pada .venv proyek, bukan global.

## Langkah perbaikan berikut

Coba webcam dan review false-positive/false-negative. Koreksi anotasi draft dan tambahkan sampel berbicara, bukaan sengaja lebar, perubahan pose, serta negatif tiap subjek. Gunakan sesi baru terpisah setelah training freeze, bukan mengklaim cross-validation development ini sebagai test final. Jika ingin menangani menguap tertutup tangan, EAR/MAR saja tidak dapat mengobservasi mulut yang tersembunyi; butuh validitas visibility yang lebih baik atau bukti tambahan. Jangan memberi confidence terkalibrasi atau tingkat kantuk dari skor ini.
