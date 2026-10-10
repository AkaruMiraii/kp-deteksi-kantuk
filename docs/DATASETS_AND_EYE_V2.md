# Dataset dan deteksi mata independen v2

## Status kerja

Dataset internet ditemukan dan sumber/syarat diperiksa. BELUM diunduh atau digunakan untuk retraining. Jangan mengklaim model smart_yawn_v1 dilatih dengan dataset tersebut. Mode mata baru adalah aturan eksperimental terkalibrasi per pengguna, bukan classifier tidur yang dilatih dari label EEG.

## Dataset yang direkomendasikan

### YawDD: menguap versus berbicara

Sumber: https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
DOI dataset: https://doi.org/10.21227/e1qm-hb90
Paper Abtahi et al., ACM MMSys 2014: https://doi.org/10.1145/2557642.2563678
Katalog: https://qualinet.github.io/databases/video/yawdd_a_yawning_detection_dataset/

IEEE menyebut 322 video pada set pertama dan 29 video pada set kedua, RGB640x480,30FPS, tanpa audio. Kondisi normal, berbicara/menyanyi dan menguap. Video tidak diberi anotasi event/frame siap pakai, sehingga onset/offset memerlukan anotasi. Boleh untuk penelitian; publikasi gambar hanya untuk partisipan yang mengizinkan dalam Participants Information. IEEE membutuhkan login gratis menurut informasi open access halaman, namun modal generik subscription juga terlihat, jadi akses aktual akun belum diuji.

Arsip yang dirujuk katalog: https://skuld.cs.umass.edu/traces/mmsys/2014/user06.tar
HEAD diverifikasi HTTP200, Content-Length5307791360 bytes (sekitar5,31GB desimal). Tidak diunduh dalam tahap ini. Arsip belum dicek isi/checksum; hanya URL sumber tersedia. Hindari menampilkan wajah dataset sebelum memeriksa izin gambar masing-masing partisipan.

### UTA-RLDD: penurunan kewaspadaan

Sumber resmi: https://sites.google.com/view/utarldd/home
Paper Ghoddoosian et al.2019: https://arxiv.org/abs/1904.07312

60 partisipan,180 video, sekitar30 jam/111,3GB. Label0 alert,5 low vigilance,10 drowsy. Label diberikan partisipan menurut keadaan dominan per video menggunakan KSS, bukan label pasti per-frame atau sleep-stage EEG. Low vigilance KSS6-7 dan drowsy8-9; tidak boleh mengubah label5 menjadi tidur ringan atau label10 menjadi tidur pasti. Situs meminta sitasi dan membatasi publikasi wajah berdasarkan izin36 partisipan. Lisensi redistribusi/perangkat komersial dan URL file langsung belum sepenuhnya diverifikasi. Belum ada unduhan.

### NTHU-DDD: variasi mata, pose dan simulasi

Sumber: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/index.html
Lisensi: http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/NTHU-DDD-LicenseAgreement.pdf

36 subjek, sekitar9,5 jam, termasuk menguap, slow blink, nodding, falling asleep, tertawa, berbicara, kacamata, siang/malam. Simulasi facial behavior, bukan sleep stage klinis. Akses melalui perjanjian yang ditandatangani kepala departemen/lab dan email permohonan. Nonkomersial penelitian/pendidikan, tidak redistribusi, pembatasan publikasi wajah. Tidak mengirim email atau menandatangani atas nama pengguna.

### DROZY: referensi fisiologis

Sumber: https://www.drozy.ulg.ac.be/
Repositori: https://orbi.uliege.be/handle/2268/191620
Lisensi: https://orbi.uliege.be/bitstream/2268/191620/5/DROZY-LicenseAgreement-20180718.pdf

14 subjek, KSS, video near-infrared, timestamp, PSG (termasuk EEG) dan reaction time PVT. Ada pengujian hilang dan missing frame yang perlu ditangani menggunakan informasi resmi. Berguna untuk studi kaitan indikator dengan fisiologi; tidak langsung cocok RGB MediaPipe tanpa domain testing dan tanpa penurunan sleep labels dari sinyal secara benar. Penelitian/evaluasi nonkomersial, tidak redistribusi atau memodifikasi database, publikasi citra hanya yang diizinkan. Dataset belum diunduh dan izin turunan fitur perlu ditelaah sebelum penggunaan.

## Dasar indikator dan batasannya

PERCLOS resmi: persentase waktu dalam satu menit mata setidaknya80% tertutup, berkaitan dengan slow eyelid closure, bukan sekadar jumlah kedipan.
Sumber FMCSA1998: https://rosap.ntl.bts.gov/view/dot/113
Sumber DOT: https://www.federalregister.gov/documents/2000/10/11/00-26015/drowsy-driver-detection-device-laboratory-validation

EAR tidak langsung mengukur80% penutupan pupil, sehingga fitur eye_closed_fraction_30s saat ini bukan PERCLOS tervalidasi. Fitur ini dihitung dan dicatat tetapi belum dipakai sebagai pemicu tingkat.

Drowsiness bersifat tidak langsung diobservasi; EEG dan bukti lain dapat membantu, sedangkan kamera melihat ekspresi/gerakan. Microsleep dapat terjadi tanpa mata sepenuhnya tertutup:
https://www.federalregister.gov/documents/2024/01/05/2023-27665/advanced-impaired-driving-prevention-technology
https://stacks.cdc.gov/view/cdc/225467/cdc_225467_DS1.pdf

## Implementasi mata saat ini

Jalankan `python -m scripts.smart_live --eye-alertness`. Opsional, mode lama tanpa flag tetap sama. Lima detik awal harus mata terbuka alami, tatap kamera, satu orang. Median EAR tiap mata menjadi baseline. Closed rule65% baseline dengan hysteresis75% saat sedang closed. Parameter ini usulan engineering, bukan ambang medis atau hasil training dataset eksternal. Baseline terlalu rendah ditolak tetapi sistem tidak bisa otomatis membuktikan pengguna benar-benar membuka mata selama kalibrasi. R melakukan reset/recalibration saat perubahan orang/pose/kacamata. Jangan membelalak secara berlebihan saat kalibrasi.

Mata dihitung ulang dari titik mata pada hasil inferensi yang sama, tanpa inferensi model kedua. Titik mulut invalid tidak otomatis membuat fitur mata invalid. Face absent, titik mata invalid atau gap>0,5s memutus kontinuitas. Partial hand occlusion tetap dapat menghasilkan landmark tebakan; belum ada model visibility/pose gate, sehingga belum dijamin andal pada tangan/glare/down-gaze.

Tingkat display eksperimental:
- 1,5s closed: drowsiness_visual_indicator (indikasi kantuk mata).
- 3s closed: near_sleep_visual_indicator (indikasi hampir tidur visual).
- 8s closed: sleep_like_visual_indicator (indikasi menyerupai tidur visual).

Ini grading DURASI PENUTUPAN MATA, bukan sleep stage, tidak mampu membedakan mata sengaja ditutup, berdoa, atau tidur nyata tanpa bukti tambahan. Final_status legacy mata tetap disimpan tidak ditimpa; gunakan eye_alertness_state untuk output adaptif. Skor yawn terpisah. Blink biasa tidak otomatis harus masuk tingkat kantuk.

## Verifikasi nyata

Tes terbaru a6ae5ecbe3c8 berisi5725 frame,274 frame final_status indikasi_kantuk,117 invalid. Jadi jalur mata lama memang aktif; tidak benar bahwa semua keputusan kantuk berasal dari mulut. Tidak ada video baru pada smart_live mode lama, maka label mana yang salah tidak dapat ditetapkan hanya dari log.

72 automated tests lolos, termasuk quality gate mata independen saat mulut invalid, reset invalid/gap, kalibrasi, tier dan CLI opt-in. Scan aislop explicit3 file bersih, --changes0 file di scratch sehingga bukan bukti scan perubahan repo. Gambar runtime kalibrasi diperiksa visual; instruksi dan ML row terpisah. Replay pilot4592 frame exit0:3522 no_eye_indicator,514 pending,150 near-sleep,149 calibrating,119 invalid,82 drowsiness,56 sleep-like. Pada bagian49-57s yang direview manusia, output meningkat sesuai duration, namun bukan accuracy atau validasi tidur. Rekaman awal sendiri belum disiapkan sebagai kalibrasi supervised terbuka, sehingga replay adalah integration smoke test, bukan gold-standard.

## Tahap berikut

Uji webcam dengan mata terbuka alami saat kalibrasi, kemudian penutupan sengaja dengan mulut santai untuk memeriksa logika, bukan sengaja mengurangi tidur. Kumpulkan label terbuka, downward gaze, berkedip, closure panjang, speaking, deliberate-wide dan visibility. Untuk retraining lebih andal: unduh subset YawDD yang sah, anotasi independen, simpan raw pristine, ekstrak fitur EAR/MAR dengan timestamp, split per partisipan, bandingkan baseline, train/dev/test terpisah. UTA-RLDD dibutuhkan untuk belajar low-vigilance vs drowsiness lebih jauh; deteksi tidur sebenarnya belum selesai dan membutuhkan referensi fisiologis atau label expert yang tepat.
