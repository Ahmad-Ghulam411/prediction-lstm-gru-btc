# Perhitungan Manual - Tahap 9.2: Perbandingan Struktur LSTM dan GRU

Subbagian ini merangkum perbedaan struktur LSTM (Tahap 7) dan GRU (Tahap 8) yang sudah
dihitung terpisah, sebagai pelengkap Tabel 13 dan subbab 1.5.10 (Tabel 2). Semua angka
berasal dari model utama: LSTM 40 neuron dan GRU 30 neuron.

**Daftar isi**

- [1. Perbedaan Struktur Sel](#1-perbedaan-struktur-sel)
- [2. Jalan Tol Gradien pada Kedua Model](#2-jalan-tol-gradien-pada-kedua-model)
- [3. Jumlah Parameter pada Jumlah Neuron yang Sama](#3-jumlah-parameter-pada-jumlah-neuron-yang-sama)
- [4. Satu Siklus Pelatihan Model Utama](#4-satu-siklus-pelatihan-model-utama)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Perbedaan Struktur Sel

| Aspek | LSTM | GRU | Penjelasan |
|---|---|---|---|
| Gerbang | 3 (forget, input, output) | 2 (update, reset) | GRU tidak punya output gate |
| Simpan vs terima | fₜ dan iₜ **bebas** satu sama lain | zₜ dan (1 − zₜ) **terikat** | pada GRU, menyimpan lebih banyak yang lama berarti menerima lebih sedikit yang baru |
| Memori internal | cₜ dan hₜ | hanya hₜ | GRU menyimpan memori langsung di hidden state |
| Peran masa lalu pada kandidat | lewat U_c | diatur rₜ (reset gate) | LSTM tidak punya padanan langsung reset gate |
| Himpunan bobot | 4 (i, f, c, o) | 3 (z, r, h) | parameter GRU lebih sedikit |
| Bias per himpunan (Keras) | n_u | 2n_u | GRU memakai bias masukan dan bias rekuren |

## 2. Jalan Tol Gradien pada Kedua Model

Kedua model punya jalur sederhana yang membuat gradien tidak cepat mengecil saat dikirim
mundur. Pada iterasi pertama pelatihan (rata-rata seluruh 32 sampel × neuron):

- **LSTM**: jalur cell state (Gambar 2), sinyal hanya dikali fₜ. Sebanyak
  **87.3%** besar sinyal kesalahan ke c₆ lewat jalur ini
  ([Tahap 7.8 (Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time)](tahap_07_08_pelatihan_lstm_bptt.md)).
- **GRU**: jalur langsung zₜ ⊙ hₜ₋₁ (Gambar 7), sinyal hanya dikali zₜ. Sebanyak
  **76.9%** besar sinyal kesalahan ke h₆ lewat jalur ini
  ([Tahap 8.8 (Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time)](tahap_08_08_pelatihan_gru_bptt.md)).

## 3. Jumlah Parameter pada Jumlah Neuron yang Sama

Jumlah parameter total (lapisan rekuren + dense) dengan 10 fitur, persamaan 22 dan 27:

| Neuron n_u | LSTM | GRU | GRU / LSTM |
|---|---|---|---|
| 10 | 851 | 671 | 78.8% |
| 20 | 2,501 | 1,941 | 77.6% |
| 30 | 4,951 | 3,811 ← terbaik | 77.0% |
| 40 | 8,201 ← terbaik | 6,281 | 76.6% |
| 50 | 12,251 | 9,351 | 76.3% |

Pada jumlah neuron yang sama, GRU selalu butuh parameter lebih sedikit. Model terbaik
penelitian adalah LSTM 40 neuron (8,201 parameter) dan GRU
30 neuron (3,811 parameter).

## 4. Satu Siklus Pelatihan Model Utama

| Besaran | LSTM (40 neuron) | GRU (30 neuron) |
|---|---|---|
| Jumlah parameter | 8,201 | 3,811 |
| Loss batch 1 dengan bobot awal | 0.000198 | 0.013489 |
| Loss batch 1 setelah 1 update | 0.001783 | 0.004921 |
| Jumlah epoch | 500 | 500 |
| Total update Adam | 13,000 | 13,000 |
| Loss latih epoch terakhir | 0.00063068 | 0.00051683 |
| Loss validasi minimum | 0.00059231 | 0.00061486 |

Urutan siklus pelatihan (forward → loss → BPTT → Adam) dan pengaturan Adam pada kedua model
sama persis; yang berbeda hanya rumus di dalam sel. Loss pada awal pelatihan bergantung pada
undian bobot awal, jadi **tidak bisa dipakai untuk menyimpulkan** model mana yang lebih
baik. Perbandingan yang sah adalah hasil evaluasi data uji dan uji Diebold-Mariano pada
Tahap 11 dan 12.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.12: Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias](tahap_08_12_bias_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 10.1: Denormalisasi Prediksi ke Skala USD](tahap_10_01_denormalisasi.md)
