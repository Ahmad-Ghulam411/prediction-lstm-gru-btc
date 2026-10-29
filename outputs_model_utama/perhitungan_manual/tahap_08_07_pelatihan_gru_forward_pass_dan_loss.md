# Perhitungan Manual - Tahap 8.7: Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss

Di [Tahap 8.6 (Alur Sel GRU pada Model Utama (Gambar 7))](tahap_08_06_alur_sel_gru_model_utama.md) kita sudah menghitung satu sampel melewati sel GRU. Pelatihan
yang sesungguhnya tidak berhenti di situ. Keras mengulang siklus empat langkah berikut
ribuan kali, dan bagian ini membahas **langkah 1 dan 2** pada iterasi pertama model GRU
utama.

**Semua angka di sini berasal dari model GRU utama penelitian**, yaitu model terbaik hasil
tuning: **30 neuron**, masukan **7 hari × 10 fitur**, batch
**32**, dan dilatih 500 epoch. Tidak ada model mini atau angka
karangan. Karena angkanya banyak (3,811 parameter), triknya begini: kita ikuti
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
| 1. Forward pass | hitung prediksi ŷ′ untuk satu batch | Tahap 8.7 |
| 2. Loss | ukur seberapa jauh ŷ′ dari y′ (MSE, persamaan 28) | Tahap 8.7 |
| 3. BPTT | hitung gradien loss terhadap setiap bobot dan bias | Tahap 8.8 |
| 4. Update Adam | geser setiap parameter sedikit ke arah yang menurunkan loss | Tahap 8.9 |

Satu putaran empat langkah ini disebut **satu iterasi**. Model GRU terbaik dilatih
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
sampel pertama di Tahap 8.6. Sampel ke-2 adalah jendela yang digeser satu hari,
dan seterusnya.

## 3. Langkah 1: Forward Pass 32 Sampel

Hasil forward pass seluruh batch pertama dengan bobot awal:

| No | Tanggal target | y′ (target) | ŷ′ (prediksi) | e = y′ − ŷ′ | e² |
|---|---|---|---|---|---|
| 1 | 2023-07-08 | 0.052770 | -0.125065 | 0.177835 | 0.031625 |
| 2 | 2023-07-09 | 0.051446 | -0.108161 | 0.159607 | 0.025474 |
| 3 | 2023-07-10 | 0.051446 | -0.093345 | 0.144791 | 0.020964 |
| 4 | 2023-07-11 | 0.053454 | -0.088875 | 0.142329 | 0.020258 |
| 5 | 2023-07-12 | 0.055603 | -0.071416 | 0.127019 | 0.016134 |
| 6 | 2023-07-13 | 0.053250 | -0.078305 | 0.131555 | 0.017307 |
| 7 | 2023-07-14 | 0.064388 | -0.041195 | 0.105583 | 0.011148 |
| 8 | 2023-07-15 | 0.052624 | -0.060073 | 0.112697 | 0.012701 |
| 9 | 2023-07-16 | 0.052232 | -0.059359 | 0.111591 | 0.012453 |
| 10 | 2023-07-17 | 0.051708 | -0.077853 | 0.129562 | 0.016786 |
| 11 | 2023-07-18 | 0.050730 | -0.062988 | 0.113718 | 0.012932 |
| 12 | 2023-07-19 | 0.047828 | -0.114324 | 0.162152 | 0.026293 |
| 13 | 2023-07-20 | 0.048370 | -0.113483 | 0.161853 | 0.026196 |
| 14 | 2023-07-21 | 0.047248 | -0.053660 | 0.100908 | 0.010182 |
| 15 | 2023-07-22 | 0.048323 | -0.013583 | 0.061907 | 0.003832 |
| 16 | 2023-07-23 | 0.046963 | 0.010890 | 0.036073 | 0.001301 |
| 17 | 2023-07-24 | 0.050095 | -0.027900 | 0.077995 | 0.006083 |
| 18 | 2023-07-25 | 0.040895 | -0.071229 | 0.112123 | 0.012572 |
| 19 | 2023-07-26 | 0.041381 | -0.075175 | 0.116556 | 0.013585 |
| 20 | 2023-07-27 | 0.042587 | -0.080278 | 0.122865 | 0.015096 |
| 21 | 2023-07-28 | 0.041256 | -0.065018 | 0.106275 | 0.011294 |
| 22 | 2023-07-29 | 0.042297 | -0.062996 | 0.105292 | 0.011086 |
| 23 | 2023-07-30 | 0.042743 | -0.042650 | 0.085392 | 0.007292 |
| 24 | 2023-07-31 | 0.041878 | -0.084819 | 0.126697 | 0.016052 |
| 25 | 2023-08-01 | 0.041405 | -0.067648 | 0.109053 | 0.011893 |
| 26 | 2023-08-02 | 0.046039 | -0.054917 | 0.100957 | 0.010192 |
| 27 | 2023-08-03 | 0.040630 | -0.053246 | 0.093876 | 0.008813 |
| 28 | 2023-08-04 | 0.040945 | -0.063475 | 0.104421 | 0.010904 |
| 29 | 2023-08-05 | 0.039841 | -0.019759 | 0.059601 | 0.003552 |
| 30 | 2023-08-06 | 0.039567 | -0.042911 | 0.082478 | 0.006803 |
| 31 | 2023-08-07 | 0.039495 | -0.065780 | 0.105275 | 0.011083 |
| 32 | 2023-08-08 | 0.040928 | -0.057861 | 0.098789 | 0.009759 |

Baris pertama sama dengan hasil Tahap 8.6 (ŷ′ = -0.125065). Prediksi ke-32
sampel cocok dengan `model.predict()` Keras, selisih terbesarnya 4.2e-08.

## 4. Langkah 2: Loss MSE

$$L = \frac{1}{B}\sum_{n=1}^{B}\left(y'_n - \hat{y}'_n\right)^2 \qquad (28)$$

```
Σ e² = 0.031625 + 0.025474 + … + 0.009759   (32 suku)
     = 0.431646
L    = 0.431646 / 32 = 0.013489

Loss dari Keras untuk batch yang sama = 0.013489   (selisih 2.6e-10)
```

Supaya terasa maknanya, ubah ke dolar. Karena normalisasi min-max linear, RMSE dalam USD =
(x_max − x_min) × √L = 98,196.75 × √0.013489 = **11,404.75 USD**.
Jadi pada awal pelatihan, prediksi model untuk batch ini rata-rata meleset sekitar
11,404.75 USD.

## 5. Apa Artinya Angka Ini?

- Rata-rata prediksi batch ini -0.065202, sedangkan rata-rata targetnya 0.046886.
  Semua 32 sampel diprediksi terlalu rendah (e > 0). Rata-rata prediksinya bahkan negatif, artinya di bawah harga terendah data latih (25,162.70 USD). Tanda galat inilah yang nanti menentukan arah perbaikan bobot di langkah 3.
- Bobot masih acak, jadi model belum "membaca" pola apa pun dari data.
- Angka L ini **bukan** titik pertama kurva loss di Tahap 8.10. Loss latih
  epoch 1 pada kurva adalah rata-rata loss dari 26 batch selama epoch pertama,
  ketika bobotnya sudah ikut berubah di setiap batch.

## 6. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11, setelah persamaan 28):

> Pelatihan dilakukan per mini-batch berukuran 32 sampel tanpa pengacakan urutan,
> sehingga satu epoch terdiri atas 26 iterasi. Pada setiap iterasi, nilai loss
> dihitung sebagai rata-rata kuadrat galat 32 sampel dalam batch tersebut.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.6: Alur Sel GRU pada Model Utama (Gambar 7)](tahap_08_06_alur_sel_gru_model_utama.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.8: Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time](tahap_08_08_pelatihan_gru_bptt.md)
