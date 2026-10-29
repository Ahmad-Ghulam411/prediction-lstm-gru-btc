# Daftar Perhitungan Manual

Berkas di folder ini ditulis otomatis oleh notebook `notebooks/btc_lstm_vs_gru_model_utama.ipynb`.
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

- [Tahap 7.4: Jumlah Parameter Model LSTM](tahap_07_04_jumlah_parameter_lstm.md)
- [Tahap 7.6: Fungsi Aktivasi dan Alur Sel LSTM pada Model Utama (Gambar 1-6)](tahap_07_06_alur_sel_lstm_model_utama.md)
- [Tahap 7.7: Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_07_07_pelatihan_lstm_forward_pass_dan_loss.md)
- [Tahap 7.8: Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time](tahap_07_08_pelatihan_lstm_bptt.md)
- [Tahap 7.9: Pelatihan LSTM Model Utama, Langkah 4: Update Adam](tahap_07_09_pelatihan_lstm_update_adam.md)
- [Tahap 7.10: Membaca Kurva Loss Model LSTM](tahap_07_10_kurva_loss_lstm.md)
- [Tahap 7.11: Forward Pass LSTM (Jendela Pertama Data Uji)](tahap_07_11_forward_pass_lstm.md)
- [Tahap 7.12: Bias pada LSTM: Peran dan Cara Menghitungnya](tahap_07_12_bias_lstm.md)

## Tahap 8 — Model GRU

- [Tahap 8.4: Jumlah Parameter Model GRU](tahap_08_04_jumlah_parameter_gru.md)
- [Tahap 8.6: Alur Sel GRU pada Model Utama (Gambar 7)](tahap_08_06_alur_sel_gru_model_utama.md)
- [Tahap 8.7: Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_08_07_pelatihan_gru_forward_pass_dan_loss.md)
- [Tahap 8.8: Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time](tahap_08_08_pelatihan_gru_bptt.md)
- [Tahap 8.9: Pelatihan GRU Model Utama, Langkah 4: Update Adam](tahap_08_09_pelatihan_gru_update_adam.md)
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
