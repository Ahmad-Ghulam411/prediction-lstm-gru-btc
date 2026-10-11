# Daftar Perhitungan Manual

Berkas di folder ini ditulis otomatis oleh notebook `notebooks/btc_lstm_vs_gru.ipynb`.
Nama berkas `tahap_TT_SS_nama.md` berarti **Tahap TT, subbagian TT.SS** di notebook,
sehingga urutan berkas sama dengan urutan pengerjaan dari tahap pertama sampai
terakhir. Setiap berkas diakhiri tautan ke berkas sebelumnya dan berikutnya.

## Tahap 2 — Eksplorasi data

- [Tahap 2.8: Rata-rata dan Standar Deviasi Close Price](tahap_02_08_rata_rata_dan_standar_deviasi_close_price.md)

## Tahap 5 — Normalisasi Min-Max

- [Tahap 5.4: Normalisasi Min-Max Close Price](tahap_05_04_normalisasi_min_max_close_price.md)

## Tahap 6 — Pembentukan sliding window

- [Tahap 6.2: Pembentukan Sliding Window](tahap_06_02_sliding_window.md)

## Tahap 7 — Model LSTM

- [Tahap 7.1: Alur Sel LSTM (Gambar 1-6) dengan Model Mini](tahap_07_01_alur_sel_lstm.md)
- [Tahap 7.2: Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md)
- [Tahap 7.3: Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time](tahap_07_03_simulasi_bptt_lstm.md)
- [Tahap 7.4: Simulasi Pelatihan LSTM, Langkah 4: Update Adam](tahap_07_04_simulasi_update_adam_lstm.md)
- [Tahap 7.8: Jumlah Parameter Model LSTM](tahap_07_08_jumlah_parameter_lstm.md)
- [Tahap 7.10: Membaca Kurva Loss Model LSTM](tahap_07_10_kurva_loss_lstm.md)
- [Tahap 7.11: Forward Pass LSTM (Jendela Pertama Data Uji)](tahap_07_11_forward_pass_lstm.md)
- [Tahap 7.12: Bias pada LSTM: Peran dan Cara Menghitungnya](tahap_07_12_bias_lstm.md)

## Tahap 8 — Model GRU

- [Tahap 8.1: Alur Sel GRU (Gambar 7) dengan Model Mini](tahap_08_01_alur_sel_gru.md)
- [Tahap 8.2: Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md)
- [Tahap 8.3: Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time](tahap_08_03_simulasi_bptt_gru.md)
- [Tahap 8.4: Simulasi Pelatihan GRU, Langkah 4: Update Adam](tahap_08_04_simulasi_update_adam_gru.md)
- [Tahap 8.8: Jumlah Parameter Model GRU](tahap_08_08_jumlah_parameter_gru.md)
- [Tahap 8.10: Membaca Kurva Loss Model GRU](tahap_08_10_kurva_loss_gru.md)
- [Tahap 8.11: Forward Pass GRU (Jendela Pertama Data Uji)](tahap_08_11_forward_pass_gru.md)
- [Tahap 8.12: Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias](tahap_08_12_bias_gru.md)

## Tahap 9 — Perbandingan arsitektur dan kestabilan

- [Tahap 9.2: Perbandingan Struktur LSTM dan GRU](tahap_09_02_perbandingan_struktur_lstm_gru.md)

## Tahap 10 — Denormalisasi

- [Tahap 10.1: Denormalisasi Prediksi ke Skala USD](tahap_10_01_denormalisasi.md)

## Tahap 11 — Evaluasi model

- [Tahap 11.1: Metrik Evaluasi (RMSE, MAE, MAPE, Akurasi Arah)](tahap_11_01_metrik_evaluasi.md)

## Tahap 12 — Uji signifikansi Diebold-Mariano

- [Tahap 12.1: Uji Diebold-Mariano LSTM vs GRU](tahap_12_01_uji_diebold_mariano.md)

Tahap tanpa perhitungan manual: 0 (Persiapan lingkungan), 1 (Pengumpulan data), 3 (Pembersihan data), 4 (Pembagian data (sebelum normalisasi)), 13 (Visualisasi), 14 (Kesimpulan otomatis).
