# Perhitungan Manual - Tahap 7.7: Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss

Di [Tahap 7.6 (Fungsi Aktivasi dan Alur Sel LSTM pada Model Utama (Gambar 1-6))](tahap_07_06_alur_sel_lstm_model_utama.md) kita sudah menghitung satu sampel melewati sel LSTM. Pelatihan
yang sesungguhnya tidak berhenti di situ. Keras mengulang siklus empat langkah berikut
ribuan kali, dan bagian ini membahas **langkah 1 dan 2** pada iterasi pertama model LSTM
utama.

**Semua angka di sini berasal dari model LSTM utama penelitian**, yaitu model terbaik hasil
tuning: **40 neuron**, masukan **7 hari × 10 fitur**, batch
**32**, dan dilatih 500 epoch. Tidak ada model mini atau angka
karangan. Karena angkanya banyak (8,201 parameter), triknya begini: kita ikuti
**satu neuron (neuron ke-1)** dan **satu sampel** secara rinci sampai bisa dicek dengan
kalkulator, sedangkan neuron dan sampel lainnya dihitung komputer dengan rumus yang sama
lalu dicocokkan dengan Keras.

**Daftar isi**

- [1. Satu Siklus Pelatihan dan Letaknya di Notebook](#1-satu-siklus-pelatihan-dan-letaknya-di-notebook)
- [2. Dari Satu Sampel ke Satu Batch](#2-dari-satu-sampel-ke-satu-batch)
- [3. Langkah 1: Forward Pass 32 Sampel](#3-langkah-1-forward-pass-32-sampel)
- [4. Langkah 2: Loss MSE](#4-langkah-2-loss-mse)
- [5. Apa Artinya Angka Ini?](#5-apa-artinya-angka-ini)
- [6. Catatan untuk Naskah Skripsi](#6-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Satu Siklus Pelatihan dan Letaknya di Notebook

| Langkah | Isi | Dibahas di |
|---|---|---|
| 1. Forward pass | hitung prediksi ŷ′ untuk satu batch | Tahap 7.7 |
| 2. Loss | ukur seberapa jauh ŷ′ dari y′ (MSE, persamaan 28) | Tahap 7.7 |
| 3. BPTT | hitung gradien loss terhadap setiap bobot dan bias | Tahap 7.8 |
| 4. Update Adam | geser setiap parameter sedikit ke arah yang menurunkan loss | Tahap 7.9 |

Satu putaran empat langkah ini disebut **satu iterasi**. Model LSTM terbaik dilatih
500 epoch × 26 iterasi per epoch = **13,000 iterasi**. Yang kita
hitung di sini adalah iterasi pertama, persis seperti yang dikerjakan `model.fit()` saat
model ini dilatih.

## 2. Dari Satu Sampel ke Satu Batch

Keras tidak memperbarui bobot setelah setiap sampel, tetapi setelah setiap **batch** berisi
32 sampel. Karena data runtun waktu tidak diacak (`shuffle=False`), batch pertama
berisi 32 jendela pertama data latih, yaitu target tanggal **2023-07-08 s.d.
2023-08-08**. Dengan 804 sampel latih, satu epoch terdiri dari
25 batch × 32 sampel + 1 batch × 4 sampel =
26 batch.

Ke-32 sampel itu dihitung **bersamaan** dengan bobot yang sama, masing-masing persis seperti
sampel pertama di Tahap 7.6. Sampel ke-2 adalah jendela yang digeser satu hari,
dan seterusnya.

## 3. Langkah 1: Forward Pass 32 Sampel

Hasil forward pass seluruh batch pertama dengan bobot awal:

| No | Tanggal target | y′ (target) | ŷ′ (prediksi) | e = y′ − ŷ′ | e² |
|---|---|---|---|---|---|
| 1 | 2023-07-08 | 0.052770 | 0.041640 | 0.011130 | 0.000124 |
| 2 | 2023-07-09 | 0.051446 | 0.030432 | 0.021014 | 0.000442 |
| 3 | 2023-07-10 | 0.051446 | 0.026984 | 0.024462 | 0.000598 |
| 4 | 2023-07-11 | 0.053454 | 0.022816 | 0.030638 | 0.000939 |
| 5 | 2023-07-12 | 0.055603 | 0.021505 | 0.034097 | 0.001163 |
| 6 | 2023-07-13 | 0.053250 | 0.033138 | 0.020112 | 0.000404 |
| 7 | 2023-07-14 | 0.064388 | 0.046608 | 0.017780 | 0.000316 |
| 8 | 2023-07-15 | 0.052624 | 0.065211 | (-0.012587) | 0.000158 |
| 9 | 2023-07-16 | 0.052232 | 0.059222 | (-0.006990) | 4.8866e-05 |
| 10 | 2023-07-17 | 0.051708 | 0.045297 | 0.006411 | 4.1104e-05 |
| 11 | 2023-07-18 | 0.050730 | 0.052571 | (-0.001841) | 3.3893e-06 |
| 12 | 2023-07-19 | 0.047828 | 0.050652 | (-0.002824) | 7.9761e-06 |
| 13 | 2023-07-20 | 0.048370 | 0.048678 | (-0.000308) | 9.4893e-08 |
| 14 | 2023-07-21 | 0.047248 | 0.056062 | (-0.008814) | 7.7682e-05 |
| 15 | 2023-07-22 | 0.048323 | 0.062523 | (-0.014200) | 0.000202 |
| 16 | 2023-07-23 | 0.046963 | 0.043631 | 0.003332 | 1.1105e-05 |
| 17 | 2023-07-24 | 0.050095 | 0.027111 | 0.022984 | 0.000528 |
| 18 | 2023-07-25 | 0.040895 | 0.027733 | 0.013162 | 0.000173 |
| 19 | 2023-07-26 | 0.041381 | 0.032914 | 0.008467 | 7.1684e-05 |
| 20 | 2023-07-27 | 0.042587 | 0.038671 | 0.003916 | 1.5334e-05 |
| 21 | 2023-07-28 | 0.041256 | 0.046715 | (-0.005459) | 2.9798e-05 |
| 22 | 2023-07-29 | 0.042297 | 0.049434 | (-0.007137) | 5.0944e-05 |
| 23 | 2023-07-30 | 0.042743 | 0.044359 | (-0.001617) | 2.6139e-06 |
| 24 | 2023-07-31 | 0.041878 | 0.036382 | 0.005496 | 3.0202e-05 |
| 25 | 2023-08-01 | 0.041405 | 0.037826 | 0.003579 | 1.2807e-05 |
| 26 | 2023-08-02 | 0.046039 | 0.051162 | (-0.005122) | 2.6238e-05 |
| 27 | 2023-08-03 | 0.040630 | 0.061238 | (-0.020608) | 0.000425 |
| 28 | 2023-08-04 | 0.040945 | 0.055312 | (-0.014367) | 0.000206 |
| 29 | 2023-08-05 | 0.039841 | 0.051836 | (-0.011994) | 0.000144 |
| 30 | 2023-08-06 | 0.039567 | 0.042805 | (-0.003238) | 1.0485e-05 |
| 31 | 2023-08-07 | 0.039495 | 0.032329 | 0.007167 | 5.1363e-05 |
| 32 | 2023-08-08 | 0.040928 | 0.037980 | 0.002949 | 8.6938e-06 |

Baris pertama sama dengan hasil Tahap 7.6 (ŷ′ = 0.041640). Prediksi ke-32
sampel cocok dengan `model.predict()` Keras, selisih terbesarnya 2.5e-08.

## 4. Langkah 2: Loss MSE

$$L = \frac{1}{B}\sum_{n=1}^{B}\left(y'_n - \hat{y}'_n\right)^2 \qquad (28)$$

```
Σ e² = 0.000124 + 0.000442 + … + 8.6938e-06   (32 suku)
     = 0.006323
L    = 0.006323 / 32 = 0.000198

Loss dari Keras untuk batch yang sama = 0.000198   (selisih 1.6e-11)
```

Supaya terasa maknanya, ubah ke dolar. Karena normalisasi min-max linear, RMSE dalam USD =
(x_max − x_min) × √L = 98,196.75 × √0.000198 = **1,380.30 USD**.
Jadi pada awal pelatihan, prediksi model untuk batch ini rata-rata meleset sekitar
1,380.30 USD.

## 5. Apa Artinya Angka Ini?

- Rata-rata prediksi batch ini 0.043149, sedangkan rata-rata targetnya 0.046886.
  Sebanyak 17 dari 32 sampel diprediksi terlalu rendah (e > 0) dan 15 sampel terlalu tinggi (e < 0). Tanda galat inilah yang nanti menentukan arah perbaikan bobot di langkah 3.
- Bobot masih acak, jadi model belum "membaca" pola apa pun dari data. Loss yang tampak kecil di batch ini lebih karena kebetulan: harga di awal data latih dekat dengan harga terendah (nilai ternormalisasinya dekat 0), dan model yang belum dilatih juga mengeluarkan angka di sekitar 0. Batch lain yang harganya lebih tinggi bisa memberi loss jauh lebih besar.
- Angka L ini **bukan** titik pertama kurva loss di Tahap 7.10. Loss latih
  epoch 1 pada kurva adalah rata-rata loss dari 26 batch selama epoch pertama,
  ketika bobotnya sudah ikut berubah di setiap batch.

## 6. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11, setelah persamaan 28):

> Pelatihan dilakukan per mini-batch berukuran 32 sampel tanpa pengacakan urutan,
> sehingga satu epoch terdiri atas 26 iterasi. Pada setiap iterasi, nilai loss
> dihitung sebagai rata-rata kuadrat galat 32 sampel dalam batch tersebut.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.6: Fungsi Aktivasi dan Alur Sel LSTM pada Model Utama (Gambar 1-6)](tahap_07_06_alur_sel_lstm_model_utama.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.8: Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time](tahap_07_08_pelatihan_lstm_bptt.md)
