# Perhitungan Manual - Tahap 9.2: Perbandingan Struktur LSTM dan GRU

Subbagian ini merangkum perbedaan struktur LSTM (Tahap 7) dan GRU (Tahap 8) yang sudah
dihitung terpisah, sebagai pelengkap Tabel 13 dan subbab 1.5.10 (Tabel 2). Alur sel
masing-masing dijelaskan di [Tahap 7.1 (Alur Sel LSTM (Gambar 1-6) dengan Model Mini)](tahap_07_01_alur_sel_lstm.md) dan [Tahap 8.1 (Alur Sel GRU (Gambar 7) dengan Model Mini)](tahap_08_01_alur_sel_gru.md).

**Daftar isi**

- [1. Perbedaan Struktur Sel](#1-perbedaan-struktur-sel)
- [2. Jalan Tol Gradien pada Kedua Model](#2-jalan-tol-gradien-pada-kedua-model)
- [3. Jumlah Parameter pada Jumlah Neuron yang Sama](#3-jumlah-parameter-pada-jumlah-neuron-yang-sama)
- [4. Model Mini: Siklus Sama, Rumus Sel Berbeda](#4-model-mini-siklus-sama-rumus-sel-berbeda)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Perbedaan Struktur Sel

| Aspek | LSTM | GRU | Penjelasan |
|---|---|---|---|
| Gerbang | 3 (forget, input, output) | 2 (update, reset) | GRU tidak punya output gate |
| Simpan vs terima | fₜ dan iₜ **independen** | zₜ dan (1 − zₜ) **terikat** | pada GRU, menyimpan lebih banyak yang lama berarti menerima lebih sedikit yang baru |
| Memori internal | cₜ dan hₜ | hanya hₜ | GRU menyimpan memori langsung di hidden state |
| Peran masa lalu pada kandidat | lewat U_c | diatur rₜ (reset gate) | tidak ada padanan langsung reset gate pada LSTM |
| Himpunan bobot | 4 (i, f, c, o) | 3 (z, r, h) | parameter GRU lebih sedikit |
| Bias per himpunan (Keras) | n_u | 2n_u | GRU memakai bias masukan dan bias rekuren |

## 2. Jalan Tol Gradien pada Kedua Model

Kedua model punya jalur sederhana yang membuat gradien tidak cepat mengecil saat dikirim
mundur ke time step sebelumnya:

- **LSTM**: jalur cell state (Gambar 2), gradien hanya dikalikan fₜ. Pada model mini,
  **86.4%** sinyal kesalahan ke c₁ lewat jalur ini
  ([Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md)).
- **GRU**: jalur langsung zₜ ⊙ hₜ₋₁ (Gambar 7), gradien hanya dikalikan zₜ. Pada model
  mini, **97.3%** sinyal kesalahan ke h₁ lewat jalur ini
  ([Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md)).

## 3. Jumlah Parameter pada Jumlah Neuron yang Sama

Jumlah parameter total (lapisan rekuren + dense) dengan 10 fitur, persamaan 22 dan 27:

| Neuron n_u | LSTM | GRU | GRU / LSTM |
|---|---|---|---|
| 10 | 851 | 671 | 78.8% |
| 20 | 2,501 | 1,941 | 77.6% |
| 30 | 4,951 | 3,811 ← terbaik | 77.0% |
| 40 | 8,201 ← terbaik | 6,281 | 76.6% |
| 50 | 12,251 | 9,351 | 76.3% |

Pada jumlah neuron yang sama, GRU selalu memerlukan parameter lebih sedikit. Model
terbaik penelitian adalah LSTM 40 neuron (8,201 parameter)
dan GRU 30 neuron (3,811 parameter).

## 4. Model Mini: Siklus Sama, Rumus Sel Berbeda

| Besaran | LSTM mini | GRU mini |
|---|---|---|
| Jumlah parameter (n_u = 1, n_f = 1) | 14 | 14 |
| Prediksi awal ŷ′ | 0.174235 | 0.204811 |
| Loss awal | 0.276429 | 0.245212 |
| Loss setelah 1 update | 0.274235 | 0.242723 |
| Loss setelah 500 update | 0.0000055602 | 0.0000000036 |

Urutan siklus pelatihan (forward → loss → BPTT → Adam) dan pengaturan Adam sama persis;
yang berbeda hanya rumus di dalam sel. Pada model mini dengan n_u = 1, kebetulan kedua model
sama-sama punya 14 parameter. Angka loss model mini **tidak dapat dipakai untuk
menyimpulkan** model mana yang lebih baik, karena bobot awalnya dipilih bulat dan hanya ada
satu sampel. Perbandingan yang sah adalah hasil evaluasi dan uji Diebold-Mariano pada
Tahap 11 dan 12.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.12: Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias](tahap_08_12_bias_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 10.1: Denormalisasi Prediksi ke Skala USD](tahap_10_01_denormalisasi.md)
