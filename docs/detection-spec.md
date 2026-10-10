# Kontrak deteksi v2

Status: milestone sementara. Sumber kebutuhan: laporan KP BAB I 1.1, 1.3, 1.5 dan BAB III 3.5-3.8. Tujuan akhir tetap prototipe EAR dan MAR dengan evaluasi kanal mata, indikator menguap dan hasil gabungan. Dokumen ini tidak menyatakan tujuan akhir selesai.

## Fitur dan validitas

EAR menggunakan enam titik per mata: R [33,160,158,133,153,144], L [362,385,387,263,373,380]. MAR = distance(13,14) / distance(78,308). Jarak memakai koordinat float piksel x*width, y*height. Pembulatan hanya untuk menggambar.

Gerbang kualitas saat ini bersifat global: wajah ditemukan, indeks tersedia, koordinat terpilih finite, berada di frame, lebar tidak nol. Gerbang ini belum mengukur visibility/occlusion secara langsung. MediaPipe masih dapat mengestimasi titik di balik tangan/objek. num_faces=1 bukan detektor bahwa tidak ada orang kedua; adegan pengujian wajib satu orang.

Timer berjalan dari timestamp monotonic webcam dan diputus oleh observasi invalid atau gap berlebih. Mata: kedua EAR < threshold. Mulut: MAR > threshold. Nilai CLI merupakan parameter pengembangan, bukan nilai final atau aturan anotasi manusia.

## Tabel keputusan sementara

Policy: v2_eye_evidence_mouth_observation.

| Kondisi | Status akhir | Pemicu / alasan |
|---|---|---|
| Wajah/geometri invalid atau kanal aktif not_assessable | tidak_dapat_dinilai | none / alasan invalid |
| Kedua kanal nonaktif | tidak_dapat_dinilai | none / no_active_channels |
| Mata nonaktif, mulut aktif | tidak_dapat_dinilai | none / eye_channel_disabled |
| Mata valid dan prolonged_closure | indikasi_kantuk | eye / prolonged_closure |
| Mata valid belum prolonged, mulut prolonged_opening | tidak_terdeteksi_indikasi_kantuk | none / mouth_opening_not_validated |
| Mata valid belum prolonged | tidak_terdeteksi_indikasi_kantuk | none / no_prolonged_eye_closure |

Bukaan mulut panjang tetap dicatat, tidak dihapus atau disebut menguap. Jika kedua kanal prolonged, bukti keputusan sementara hanya mata. Tidak terdeteksi berarti aturan mata belum menemukan indikator, bukan bukti orang waspada atau bukti menguap tidak terjadi. Kehilangan validitas memutus timer; kanal yang sekadar dinonaktifkan tidak mengubah timer kanal aktif.

Log menyimpan final_status, drowsiness_trigger, decision_reason dan decision_policy. Metadata mencatat mouth_evidence_validated=false, full_project_acceptance=false. Status tidak dikenal adalah bug dan menghasilkan ValueError, bukan fallback negatif.

## Gerbang penyelesaian proyek

1. Rekaman mentah dan timestamp sinkron untuk anotasi independen. Snapshot 2 detik tidak cukup untuk evaluasi per-frame.
2. Rubrik dan split data ditetapkan sebelum pengujian akhir.
3. Kalibrasi threshold/durasi pada development saja.
4. Evaluasi MAR terhadap contoh menguap, berbicara dan bukaan sengaja. Jika tidak cukup spesifik, ubah fitur/aturan temporal berdasarkan development atau laporkan keterbatasan; jangan menyebut bukaan besar sebagai menguap secara otomatis.
5. Pilih dan dokumentasikan aturan gabungan hanya setelah bukti development, kemudian bekukan kode/config sebelum held-out test. Jangan memakai anotasi manusia atau tombol Activity sebagai input prediksi runtime.
6. Laporkan precision/recall/F1/confusion matrix kanal mata, kanal mulut yang telah divalidasi, gabungan, coverage, abstention per kelas, timing dan FPS nyata.

## Koreksi interpretasi bukti lama

Log dengan policy OR adalah bukti implementasi aturan lama, bukan akurasi dan bukan kontrak final. Replay yang cocok menguji konsistensi, bukan kebenaran label. Median processing_ms tidak boleh dibalik untuk mengklaim FPS keseluruhan. Snapshot wajah ditutupi yang menghasilkan invalid tidak membuktikan penanganan semua oklusi. Penutupan mata sengaja dan bukaan sengaja adalah skenario simulasi, bukan diagnosis kantuk.
