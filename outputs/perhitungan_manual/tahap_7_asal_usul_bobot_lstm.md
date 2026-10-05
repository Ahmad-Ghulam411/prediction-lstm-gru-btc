# Perhitungan Manual - Tahap 7: Asal-Usul Bobot LSTM (Inisialisasi sampai Bobot Akhir)

## Gambaran besar

Bobot pada tabel bobot neuron ke-1 yang dipakai untuk *forward pass* **tidak dihitung dengan satu rumus**. Bobot itu adalah hasil akhir dari proses berikut:

| Langkah | Apa yang terjadi | Hasil |
|---|---|---|
| 1 | Menentukan n_in, n_out, dan batas acak Glorot uniform | a = 0.187867 |
| 2 | Mengambil bilangan acak di dalam batas itu (seed 42) | bobot awal |
| 3 | Menghitung banyaknya batch per epoch | 26 batch/epoch, 13,000 pembaruan |
| 4 | Memperbarui bobot satu kali dengan Adam (contoh satu bobot) | bobot bergeser sedikit |
| 5 | Mengulang pembaruan sebanyak 13,000 kali | bobot akhir |
| 6 | Membaca bobot akhir neuron ke-1 | tabel bobot pada Bab III |

Ibaratnya, setiap bobot adalah kenop pengatur. Kenop mula-mula diputar ke posisi acak yang kecil, lalu digeser sedikit demi sedikit ke arah yang membuat galat prediksi makin kecil.

## Langkah 1 — Menentukan n_in, n_out, dan batas Glorot uniform

**Apa yang dilakukan?** Sebelum pelatihan, setiap bobot masukan diisi bilangan acak kecil. Agar tidak terlalu besar atau terlalu kecil, Keras memakai aturan Glorot uniform: bobot diambil acak secara merata di antara −a dan +a.

**Dari mana n_in dan n_out?** Keras tidak membuat matriks terpisah untuk tiap gerbang. Bobot masukan 4 blok (i, f, c, o) disimpan berdampingan dalam SATU matriks kernel, dan batas Glorot dihitung satu kali untuk matriks gabungan itu. Jadi n_in = jumlah baris kernel dan n_out = jumlah kolom kernel.

| Besaran | Cara menentukan | Nilai |
|---|---|---|
| n_fitur | banyaknya variabel masukan (X1–X10) | 10 |
| n_unit | banyaknya neuron model terbaik | 40 |
| Bentuk kernel W | (n_fitur, 4 × n_unit), urutan blok i, f, c, o | (10, 160) |
| n_in | jumlah baris kernel = n_fitur | 10 |
| n_out | jumlah kolom kernel = 4 × n_unit = 4 × 40 | 160 |

```
a = √( 6 / (n_in + n_out) )
  = √( 6 / (10 + 160) )
  = √( 6 / 170 )
  = √( 0.035294 )
  = 0.187867

Jadi setiap bobot masukan awal W_i, W_f, W_c, W_o ~ U(−0.187867, +0.187867)
```

**Dari mana angka 6?** Glorot menginginkan varians bobot 2 / (n_in + n_out). Distribusi seragam U(−a, a) memiliki varians a²/3. Dengan menyamakan keduanya, a²/3 = 2 / (n_in + n_out), sehingga a = √(6 / (n_in + n_out)).

**Tiga hal yang sering keliru:**

1. Menghitung per gerbang dengan matriks (10, 40) menghasilkan √(6 / 50) = 0.346410. Angka ini *bukan* yang dipakai Keras.
2. LOOKBACK = 7 tidak mengubah n_in. Bobot yang sama dipakai ulang pada setiap hari t = 1, …, 7, sehingga pada satu langkah waktu yang masuk hanya 10 fitur.
3. Bobot rekuren U tidak memakai Glorot, melainkan inisialisasi Orthogonal. Bias diisi 0, kecuali bias forget gate yang diisi 1 (unit_forget_bias=True).

**Pemeriksaan:** nilai mutlak terbesar pada kernel awal = 0.187772 ≤ 0.187867 → OK.

## Langkah 2 — Membangkitkan bobot awal (seed 42)

**Apa yang dilakukan?** Komputer mengambil 1,600 bilangan acak dari U(−0.187867, +0.187867) untuk mengisi kernel (10, 160). Seed 42 dikunci lebih dahulu, sehingga bilangan acak yang keluar selalu sama setiap kali notebook dijalankan.

**Mana yang milik neuron ke-1?** Neuron ke-1 adalah kolom pertama dari setiap blok gerbang, yaitu kolom ke-1, ke-41, ke-81, ke-121 pada kernel gabungan (urutan i, f, c, o).

**Tabel bobot awal neuron ke-1 LSTM (sebelum pelatihan)**

| Variabel | W_i[:,1] | W_f[:,1] | W_c[:,1] | W_o[:,1] |
|---|---|---|---|---|
| X1 Close Price (USD) | 0.134584 | -0.171849 | -0.113418 | 0.006389 |
| X2 Miners Revenue (USD) | 0.027743 | -0.114586 | 0.076105 | 0.035660 |
| X3 Difficulty | -0.076258 | -0.039092 | -0.160923 | 0.128925 |
| X4 Hash Rate | -0.179887 | 0.003295 | -0.051665 | -0.112403 |
| X5 Median Confirmation Time | 0.169334 | 0.035027 | 0.165544 | -0.069591 |
| X6 Average Block Size | 0.121994 | -0.041692 | -0.121751 | 0.009207 |
| X7 Total Unique Addresses | 0.169539 | -0.047612 | 0.092225 | 0.166509 |
| X8 Transaction per Block | -0.076353 | 0.039421 | -0.168962 | 0.072687 |
| X9 Confirmed Transaction | -0.094984 | 0.031904 | 0.027057 | 0.106332 |
| X10 Cost % per Transaction | 0.175737 | 0.104081 | -0.075687 | -0.059456 |
| Bias b[1] | 0.000000 | 1.000000 | 0.000000 | 0.000000 |

**Pemeriksaan:** seluruh 1,600 bobot awal berada di dalam ±0.187867. Varians empirisnya 0.011281, dekat dengan varians teori 2 / (n_in + n_out) = 0.011765 → OK.

## Langkah 3 — Menghitung banyaknya batch dan pembaruan bobot

**Apa yang dilakukan?** Bobot tidak diperbarui setelah melihat seluruh data sekaligus, tetapi setiap selesai memproses satu kelompok kecil (*batch*) berisi 32 jendela. Satu *epoch* berarti seluruh 804 jendela latih sudah dipakai tepat satu kali.

| No | Besaran | Perhitungan | Hasil |
|---|---|---|---|
| 1 | Jumlah hari seluruh data | — | 1,127 |
| 2 | Hari latih + validasi | floor(0.8 × 1,127) = floor(901.6) | 901 |
| 3 | Hari validasi | floor(0.1 × 901) = floor(90.1) | 90 |
| 4 | Hari latih | 901 − 90 | 811 |
| 5 | Jendela latih | hari latih − LOOKBACK = 811 − 7 | 804 |
| 6 | Batch size | ditetapkan peneliti pada CONFIG (tidak dihitung) | 32 |
| 7 | Batch per epoch | ⌈804 / 32⌉ = ⌈25.125⌉ | 26 |
| 8 | Epoch | konfigurasi terbaik hasil tuning | 500 |
| 9 | Total pembaruan bobot | 26 × 500 | 13,000 |

**Mengapa 811 − 7?** Setiap jendela membutuhkan 7 hari sebelumnya sebagai masukan. Tujuh hari pertama data latih tidak memiliki 7 hari sebelumnya, sehingga hanya berperan sebagai masukan dan tidak pernah menjadi target. Target pertama adalah 2023-07-08 dan target terakhir 2025-09-18.

**Mengapa dibulatkan ke atas?** 804 = 25 × 32 + 4. Keras tidak membuang 4 jendela sisa, melainkan menjadikannya batch ke-26 yang berisi 4 jendela.

**Mengapa batch size 32?** Batch size adalah hyperparameter yang ditetapkan peneliti, bukan hasil perhitungan. Nilainya dibuat sama untuk LSTM dan GRU agar perbandingan adil, tidak ikut di-tuning, dan kebetulan juga merupakan nilai baku Keras.

**Pembagian batch dalam satu epoch** (urutan kronologis karena shuffle=False):

| Batch | Jendela ke- | Banyak jendela | Tanggal target |
|---|---|---|---|
| 1 | 1 – 32 | 32 | 2023-07-08 s.d. 2023-08-08 |
| 2 | 33 – 64 | 32 | 2023-08-09 s.d. 2023-09-09 |
| 3 | 65 – 96 | 32 | 2023-09-10 s.d. 2023-10-11 |
| 4 | 97 – 128 | 32 | 2023-10-12 s.d. 2023-11-12 |
| 5 | 129 – 160 | 32 | 2023-11-13 s.d. 2023-12-14 |
| 6 | 161 – 192 | 32 | 2023-12-15 s.d. 2024-01-15 |
| 7 | 193 – 224 | 32 | 2024-01-16 s.d. 2024-02-16 |
| 8 | 225 – 256 | 32 | 2024-02-17 s.d. 2024-03-19 |
| 9 | 257 – 288 | 32 | 2024-03-20 s.d. 2024-04-20 |
| 10 | 289 – 320 | 32 | 2024-04-21 s.d. 2024-05-22 |
| 11 | 321 – 352 | 32 | 2024-05-23 s.d. 2024-06-23 |
| 12 | 353 – 384 | 32 | 2024-06-24 s.d. 2024-07-25 |
| 13 | 385 – 416 | 32 | 2024-07-26 s.d. 2024-08-26 |
| 14 | 417 – 448 | 32 | 2024-08-27 s.d. 2024-09-27 |
| 15 | 449 – 480 | 32 | 2024-09-28 s.d. 2024-10-29 |
| 16 | 481 – 512 | 32 | 2024-10-30 s.d. 2024-11-30 |
| 17 | 513 – 544 | 32 | 2024-12-01 s.d. 2025-01-01 |
| 18 | 545 – 576 | 32 | 2025-01-02 s.d. 2025-02-02 |
| 19 | 577 – 608 | 32 | 2025-02-03 s.d. 2025-03-06 |
| 20 | 609 – 640 | 32 | 2025-03-07 s.d. 2025-04-07 |
| 21 | 641 – 672 | 32 | 2025-04-08 s.d. 2025-05-09 |
| 22 | 673 – 704 | 32 | 2025-05-10 s.d. 2025-06-10 |
| 23 | 705 – 736 | 32 | 2025-06-11 s.d. 2025-07-12 |
| 24 | 737 – 768 | 32 | 2025-07-13 s.d. 2025-08-13 |
| 25 | 769 – 800 | 32 | 2025-08-14 s.d. 2025-09-14 |
| 26 | 801 – 804 | 4 | 2025-09-15 s.d. 2025-09-18 |

## Langkah 4 — Satu kali pembaruan bobot (contoh: W_i[1,1])

Bobot yang dicontohkan adalah W_i[1,1], yaitu bobot dari X1 (Close Price) ke input gate neuron ke-1. Nilai awalnya 0.134584 (Langkah 2). Setiap pembaruan selalu terdiri atas tiga langkah kecil: hitung loss, hitung gradien, lalu geser bobot.

### 4a. Hitung prediksi dan loss batch pertama

Batch pertama berisi jendela 1–32. Dengan bobot awal, model memprediksi Close Price (ternormalisasi) untuk 32 target tersebut memakai *forward pass* yang sama dengan Tahap 7, lalu loss MSE dihitung:

```
L = (1/32) × Σ (y − ŷ)²
```

| No | Tanggal target | y (aktual) | ŷ (prediksi awal) | (y − ŷ)² |
|---|---|---|---|---|
| 1 | 2023-07-08 | 0.052770 | 0.041640 | 0.00012387 |
| 2 | 2023-07-09 | 0.051446 | 0.030432 | 0.00044158 |
| 3 | 2023-07-10 | 0.051446 | 0.026984 | 0.00059840 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| 32 | 2023-08-08 | 0.040928 | 0.037980 | 0.00000869 |
|  | **Jumlah 32 suku** |  |  | 0.00632268 |
|  | **L = jumlah / 32** |  |  | **0.00019758** |

Loss manual (NumPy) = 0.00019758; loss Keras = 0.00019758 → OK.

### 4b. Hitung gradien g = ∂L/∂w

Gradien menjawab pertanyaan: *kalau bobot ini dinaikkan sedikit, loss naik atau turun, dan seberapa cepat?* Keras menghitungnya secara eksak dengan *backpropagation through time* (BPTT), yaitu aturan rantai yang dirambatkan mundur dari hari ke-7 ke hari ke-1.

Untuk memeriksanya tanpa kalkulus, bobot digeser sedikit (h = 0.0001) ke atas dan ke bawah, lalu loss dihitung ulang (beda hingga terpusat):

```
L(w + h) = L(0.134684) = 0.000197583956
L(w − h) = L(0.134484) = 0.000197583841

g ≈ [ L(w + h) − L(w − h) ] / (2h)
  = (0.000197583956 − 0.000197583841) / 0.0002
  = 5.76187432e-07

Gradien Keras (BPTT)  = 5.76187574e-07
Selisih relatif       = 2.5e-07  (praktis sama)
```

Gradien bernilai positif: loss naik bila w dinaikkan, sehingga w harus diturunkan agar loss mengecil.

### 4c. Geser bobot dengan Adam

Parameter Adam: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-07. Ini langkah pertama (t = 1), sehingga m₀ = v₀ = 0.

```
m₁ = β₁·m₀ + (1 − β₁)·g       = 0.9 × 0 + 0.1 × 5.761876e-07
                             = 5.761876e-08
v₁ = β₂·v₀ + (1 − β₂)·g²      = 0.999 × 0 + 0.001 × (5.761876e-07)²
                             = 3.319921e-16
α₁ = η·√(1 − β₂¹) / (1 − β₁¹) = 0.001 × √0.001 / 0.1
                             = 3.162278e-04
Δw = α₁ · m₁ / (√v₁ + ε)      = 3.162278e-04 × 5.761876e-08 / (1.822065e-08 + 1e-07)
                             = 1.541241e-04

w baru = w − Δw = 0.13458404 − (0.00015412) = 0.13442992
```

Hasil Keras setelah satu langkah (train_on_batch pada batch yang sama): 0.13442992. Selisihnya 1.4e-09 → OK.

**Catatan:** Adam membagi m dengan √v, sehingga besar langkah hampir tidak bergantung pada besar gradien. Untuk gradien yang cukup besar, langkah pertama Adam kira-kira sama dengan η = 0.001. Pada contoh ini gradiennya sangat kecil sehingga ε ikut berperan dan langkahnya hanya 0.000154.

## Langkah 5 — Mengulang pembaruan sebanyak 13,000 kali

Langkah 4 diulang untuk setiap batch (26 batch) pada setiap epoch (500 epoch). Pelatihan diulang dari awal dengan seed 42, dan bobot neuron ke-1 dicatat pada akhir epoch tertentu:

| Epoch | Pembaruan ke- | W_i[1,1] | W_f[1,1] | W_c[1,1] | W_o[1,1] | b_f[1] | Loss latih (MSE) |
|---|---|---|---|---|---|---|---|
| 0 (awal) | 0 | 0.134584 | -0.171849 | -0.113418 | 0.006389 | 1.000000 | — |
| 1 | 26 | 0.119599 | -0.186656 | -0.098260 | -0.008763 | 0.987200 | 0.022093 |
| 2 | 52 | 0.108958 | -0.197966 | -0.093600 | -0.019569 | 0.972327 | 0.100346 |
| 5 | 130 | 0.097919 | -0.209762 | -0.086105 | -0.030712 | 0.962647 | 0.004206 |
| 10 | 260 | 0.100334 | -0.209310 | -0.087695 | -0.028181 | 0.962666 | 0.002258 |
| 50 | 1,300 | 0.100471 | -0.208693 | -0.083355 | -0.027531 | 0.953846 | 0.000979 |
| 100 | 2,600 | 0.097381 | -0.202421 | -0.064363 | -0.029846 | 0.947999 | 0.000854 |
| 200 | 5,200 | 0.136300 | -0.153011 | 0.004259 | 0.018021 | 0.934346 | 0.001129 |
| 300 | 7,800 | 0.219202 | -0.140527 | 0.090471 | 0.112700 | 0.838726 | 0.001081 |
| 400 | 10,400 | 0.333889 | -0.098343 | 0.190222 | 0.244989 | 0.742818 | 0.000611 |
| 500 | 13,000 | 0.413355 | -0.066825 | 0.158535 | 0.321498 | 0.681141 | 0.000631 |

Loss latih adalah rata-rata loss seluruh batch pada epoch tersebut (riwayat pelatihan Keras).

**Pengamatan:**

- Setiap langkah umumnya menggeser bobot tidak lebih dari sekitar η = 0.001, tetapi setelah 13,000 langkah total pergeserannya menjadi besar.
- Loss latih turun dari 0.022093 (epoch 1) menjadi 0.000631 (epoch 500).
- Bobot akhir boleh keluar dari batas Glorot ±0.187867. Contohnya W_o[10,1] berakhir di -1.053337. Batas Glorot hanya berlaku untuk bobot awal.

**Pemeriksaan:** penghitung iterasi optimizer Keras = 13,000 = 26 × 500 → OK.

## Langkah 6 — Bobot akhir neuron ke-1 (yang dipakai pada *forward pass*)

Setelah 13,000 pembaruan, bobot model dibaca dengan get_weights(). Kernel (10, 160) dipecah menjadi 4 blok berurutan i, f, c, o, lalu kolom pertama tiap blok diambil sebagai neuron ke-1. Bias neuron ke-1 diambil dari elemen ke-1, ke-41, ke-81, ke-121 vektor bias (160,).

**Tabel bobot akhir neuron ke-1 LSTM (setelah pelatihan)**

| Variabel | W_i[:,1] | W_f[:,1] | W_c[:,1] | W_o[:,1] |
|---|---|---|---|---|
| X1 Close Price (USD) | 0.413355 | -0.066825 | 0.158535 | 0.321498 |
| X2 Miners Revenue (USD) | -0.016684 | 0.108273 | 0.309700 | -0.126170 |
| X3 Difficulty | 0.125940 | 0.047586 | -0.162692 | 0.379633 |
| X4 Hash Rate | -0.200919 | -0.092290 | -0.077519 | -0.219034 |
| X5 Median Confirmation Time | -0.059908 | -0.222868 | 0.197350 | -0.206085 |
| X6 Average Block Size | -0.082207 | -0.322515 | 0.000810 | -0.152426 |
| X7 Total Unique Addresses | -0.518425 | -0.626940 | 0.068815 | -0.604805 |
| X8 Transaction per Block | -0.265667 | -0.091566 | -0.092497 | -0.106231 |
| X9 Confirmed Transaction | -0.408494 | -0.190472 | 0.074651 | -0.281759 |
| X10 Cost % per Transaction | -0.317435 | -0.258931 | 0.007119 | -1.053337 |
| Bias b[1] | -0.342494 | 0.681141 | -0.011402 | -0.373906 |

**Pemeriksaan:** bobot hasil pelatihan ulang dibandingkan dengan bobot model terbaik dari Tahap 7.1 untuk seluruh 8,160 parameter lapisan rekuren. Selisih mutlak terbesarnya 0.0e+00 → identik. Tabel di atas sama persis dengan bobot yang dipakai pada *forward pass* Tahap 7.

## Kesimpulan

Bobot LSTM neuron ke-1 berasal dari bilangan acak Glorot uniform U(−0.187867, +0.187867) dengan n_in = 10 dan n_out = 4 × 40 = 160, yang dibangkitkan dengan seed 42. Bobot itu kemudian diperbarui oleh Adam sebanyak 26 batch × 500 epoch = 13,000 kali berdasarkan gradien BPTT dari loss MSE. Satu langkah Adam yang dihitung manual sama dengan hasil Keras, dan pelatihan ulang menghasilkan bobot yang identik dengan model terbaik, sehingga seluruh nilai bobot pada Bab III dapat ditelusuri asal-usulnya.
