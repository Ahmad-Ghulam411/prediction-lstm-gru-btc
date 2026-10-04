# Perhitungan Manual - Tahap 8: Asal-Usul Bobot GRU (Inisialisasi sampai Bobot Akhir)

## Gambaran besar

Bobot pada tabel bobot neuron ke-1 yang dipakai untuk *forward pass* **tidak dihitung dengan satu rumus**. Bobot itu adalah hasil akhir dari proses berikut:

| Langkah | Apa yang terjadi | Hasil |
|---|---|---|
| 1 | Menentukan n_in, n_out, dan batas acak Glorot uniform | a = 0.244949 |
| 2 | Mengambil bilangan acak di dalam batas itu (seed 42) | bobot awal |
| 3 | Menghitung banyaknya batch per epoch | 26 batch/epoch, 13,000 pembaruan |
| 4 | Memperbarui bobot satu kali dengan Adam (contoh satu bobot) | bobot bergeser sedikit |
| 5 | Mengulang pembaruan sebanyak 13,000 kali | bobot akhir |
| 6 | Membaca bobot akhir neuron ke-1 | tabel bobot pada Bab III |

Ibaratnya, setiap bobot adalah kenop pengatur. Kenop mula-mula diputar ke posisi acak yang kecil, lalu digeser sedikit demi sedikit ke arah yang membuat galat prediksi makin kecil.

## Langkah 1 — Menentukan n_in, n_out, dan batas Glorot uniform

**Apa yang dilakukan?** Sebelum pelatihan, setiap bobot masukan diisi bilangan acak kecil. Agar tidak terlalu besar atau terlalu kecil, Keras memakai aturan Glorot uniform: bobot diambil acak secara merata di antara −a dan +a.

**Dari mana n_in dan n_out?** Keras tidak membuat matriks terpisah untuk tiap gerbang. Bobot masukan 3 blok (z, r, h) disimpan berdampingan dalam SATU matriks kernel, dan batas Glorot dihitung satu kali untuk matriks gabungan itu. Jadi n_in = jumlah baris kernel dan n_out = jumlah kolom kernel.

| Besaran | Cara menentukan | Nilai |
|---|---|---|
| n_fitur | banyaknya variabel masukan (X1–X10) | 10 |
| n_unit | banyaknya neuron model terbaik | 30 |
| Bentuk kernel W | (n_fitur, 3 × n_unit), urutan blok z, r, h | (10, 90) |
| n_in | jumlah baris kernel = n_fitur | 10 |
| n_out | jumlah kolom kernel = 3 × n_unit = 3 × 30 | 90 |

```
a = √( 6 / (n_in + n_out) )
  = √( 6 / (10 + 90) )
  = √( 6 / 100 )
  = √( 0.060000 )
  = 0.244949

Jadi setiap bobot masukan awal W_z, W_r, W_h ~ U(−0.244949, +0.244949)
```

**Dari mana angka 6?** Glorot menginginkan varians bobot 2 / (n_in + n_out). Distribusi seragam U(−a, a) memiliki varians a²/3. Dengan menyamakan keduanya, a²/3 = 2 / (n_in + n_out), sehingga a = √(6 / (n_in + n_out)).

**Tiga hal yang sering keliru:**

1. Menghitung per gerbang dengan matriks (10, 30) menghasilkan √(6 / 40) = 0.387298. Angka ini *bukan* yang dipakai Keras.
   Walaupun pembahasan sering hanya menyebut gerbang update dan reset, blok kandidat h ikut berada di matriks yang sama, sehingga n_out = 3 × n_unit, bukan 2 × n_unit.
2. LOOKBACK = 7 tidak mengubah n_in. Bobot yang sama dipakai ulang pada setiap hari t = 1, …, 7, sehingga pada satu langkah waktu yang masuk hanya 10 fitur.
3. Bobot rekuren U tidak memakai Glorot, melainkan inisialisasi Orthogonal. Kedua baris bias (b_in dan b_rec) diisi 0.

**Pemeriksaan:** nilai mutlak terbesar pada kernel awal = 0.244430 ≤ 0.244949 → OK.

## Langkah 2 — Membangkitkan bobot awal (seed 42)

**Apa yang dilakukan?** Komputer mengambil 900 bilangan acak dari U(−0.244949, +0.244949) untuk mengisi kernel (10, 90). Seed 42 dikunci lebih dahulu, sehingga bilangan acak yang keluar selalu sama setiap kali notebook dijalankan.

**Mana yang milik neuron ke-1?** Neuron ke-1 adalah kolom pertama dari setiap blok gerbang, yaitu kolom ke-1, ke-31, ke-61 pada kernel gabungan (urutan z, r, h).

**Tabel bobot awal neuron ke-1 GRU (sebelum pelatihan)**

| Variabel | W_z[:,1] | W_r[:,1] | W_h[:,1] |
|---|---|---|---|
| X1 Close Price (USD) | 0.175476 | -0.142221 | 0.126915 |
| X2 Miners Revenue (USD) | 0.022169 | 0.008330 | 0.045909 |
| X3 Difficulty | -0.048002 | 0.010498 | 0.099229 |
| X4 Hash Rate | -0.191756 | 0.074515 | 0.220549 |
| X5 Median Confirmation Time | -0.050970 | -0.120957 | -0.015057 |
| X6 Average Block Size | -0.070206 | -0.234544 | -0.113217 |
| X7 Total Unique Addresses | -0.238462 | -0.109583 | -0.146556 |
| X8 Transaction per Block | -0.093799 | 0.157352 | -0.132701 |
| X9 Confirmed Transaction | 0.215843 | 0.044872 | -0.191412 |
| X10 Cost % per Transaction | -0.072847 | -0.054360 | 0.075998 |
| Bias b_in[1] | 0.000000 | 0.000000 | 0.000000 |
| Bias b_rec[1] | 0.000000 | 0.000000 | 0.000000 |

**Pemeriksaan:** seluruh 900 bobot awal berada di dalam ±0.244949. Varians empirisnya 0.019202, dekat dengan varians teori 2 / (n_in + n_out) = 0.020000 → OK.

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

## Langkah 4 — Satu kali pembaruan bobot (contoh: W_z[1,1])

Bobot yang dicontohkan adalah W_z[1,1], yaitu bobot dari X1 (Close Price) ke update gate neuron ke-1. Nilai awalnya 0.175476 (Langkah 2). Setiap pembaruan selalu terdiri atas tiga langkah kecil: hitung loss, hitung gradien, lalu geser bobot.

### 4a. Hitung prediksi dan loss batch pertama

Batch pertama berisi jendela 1–32. Dengan bobot awal, model memprediksi Close Price (ternormalisasi) untuk 32 target tersebut memakai *forward pass* yang sama dengan Tahap 8, lalu loss MSE dihitung:

```
L = (1/32) × Σ (y − ŷ)²
```

| No | Tanggal target | y (aktual) | ŷ (prediksi awal) | (y − ŷ)² |
|---|---|---|---|---|
| 1 | 2023-07-08 | 0.052770 | -0.125065 | 0.03162531 |
| 2 | 2023-07-09 | 0.051446 | -0.108161 | 0.02547441 |
| 3 | 2023-07-10 | 0.051446 | -0.093345 | 0.02096445 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| 32 | 2023-08-08 | 0.040928 | -0.057861 | 0.00975936 |
|  | **Jumlah 32 suku** |  |  | 0.43164588 |
|  | **L = jumlah / 32** |  |  | **0.01348893** |

Loss manual (NumPy) = 0.01348893; loss Keras = 0.01348893 → OK.

### 4b. Hitung gradien g = ∂L/∂w

Gradien menjawab pertanyaan: *kalau bobot ini dinaikkan sedikit, loss naik atau turun, dan seberapa cepat?* Keras menghitungnya secara eksak dengan *backpropagation through time* (BPTT), yaitu aturan rantai yang dirambatkan mundur dari hari ke-7 ke hari ke-1.

Untuk memeriksanya tanpa kalkulus, bobot digeser sedikit (h = 0.0001) ke atas dan ke bawah, lalu loss dihitung ulang (beda hingga terpusat):

```
L(w + h) = L(0.175576) = 0.013488932629
L(w − h) = L(0.175376) = 0.013488934781

g ≈ [ L(w + h) − L(w − h) ] / (2h)
  = (0.013488932629 − 0.013488934781) / 0.0002
  = -1.07593398e-05

Gradien Keras (BPTT)  = -1.07593405e-05
Selisih relatif       = 6.9e-08  (praktis sama)
```

Gradien bernilai negatif: loss turun bila w dinaikkan, sehingga w harus dinaikkan agar loss mengecil.

### 4c. Geser bobot dengan Adam

Parameter Adam: η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-07. Ini langkah pertama (t = 1), sehingga m₀ = v₀ = 0.

```
m₁ = β₁·m₀ + (1 − β₁)·g       = 0.9 × 0 + 0.1 × -1.075934e-05
                             = -1.075934e-06
v₁ = β₂·v₀ + (1 − β₂)·g²      = 0.999 × 0 + 0.001 × (-1.075934e-05)²
                             = 1.157634e-13
α₁ = η·√(1 − β₂¹) / (1 − β₁¹) = 0.001 × √0.001 / 0.1
                             = 3.162278e-04
Δw = α₁ · m₁ / (√v₁ + ε)      = 3.162278e-04 × -1.075934e-06 / (3.402402e-07 + 1e-07)
                             = -7.728513e-04

w baru = w − Δw = 0.17547609 − (-0.00077285) = 0.17624894
```

Hasil Keras setelah satu langkah (train_on_batch pada batch yang sama): 0.17624894. Selisihnya 2.6e-09 → OK.

**Catatan:** Adam membagi m dengan √v, sehingga besar langkah hampir tidak bergantung pada besar gradien. Untuk gradien yang cukup besar, langkah pertama Adam kira-kira sama dengan η = 0.001. Pada contoh ini gradiennya sangat kecil sehingga ε ikut berperan dan langkahnya hanya 0.000773.

## Langkah 5 — Mengulang pembaruan sebanyak 13,000 kali

Langkah 4 diulang untuk setiap batch (26 batch) pada setiap epoch (500 epoch). Pelatihan diulang dari awal dengan seed 42, dan bobot neuron ke-1 dicatat pada akhir epoch tertentu:

| Epoch | Pembaruan ke- | W_z[1,1] | W_r[1,1] | W_h[1,1] | b_z_in[1] | Loss latih (MSE) |
|---|---|---|---|---|---|---|
| 0 (awal) | 0 | 0.175476 | -0.142221 | 0.126915 | 0.000000 | — |
| 1 | 26 | 0.177401 | -0.127611 | 0.144307 | 0.007903 | 0.009302 |
| 2 | 52 | 0.170242 | -0.137414 | 0.154159 | 0.014825 | 0.120103 |
| 5 | 130 | 0.164030 | -0.150490 | 0.163683 | 0.017425 | 0.004089 |
| 10 | 260 | 0.168261 | -0.155562 | 0.169812 | 0.028025 | 0.003042 |
| 50 | 1,300 | 0.155381 | -0.166959 | 0.202841 | 0.051224 | 0.000712 |
| 100 | 2,600 | 0.104166 | -0.216805 | 0.212998 | 0.035184 | 0.000840 |
| 200 | 5,200 | 0.089329 | -0.277929 | 0.259492 | 0.007522 | 0.000369 |
| 300 | 7,800 | 0.022448 | -0.369293 | 0.319006 | 0.034628 | 0.000835 |
| 400 | 10,400 | -0.036529 | -0.422422 | 0.357325 | 0.080047 | 0.000502 |
| 500 | 13,000 | -0.004894 | -0.466489 | 0.398500 | 0.211800 | 0.000517 |

Loss latih adalah rata-rata loss seluruh batch pada epoch tersebut (riwayat pelatihan Keras).

**Pengamatan:**

- Setiap langkah umumnya menggeser bobot tidak lebih dari sekitar η = 0.001, tetapi setelah 13,000 langkah total pergeserannya menjadi besar.
- Loss latih turun dari 0.009302 (epoch 1) menjadi 0.000517 (epoch 500).
- Bobot akhir boleh keluar dari batas Glorot ±0.244949. Contohnya W_r[6,1] berakhir di -0.476919. Batas Glorot hanya berlaku untuk bobot awal.

**Pemeriksaan:** penghitung iterasi optimizer Keras = 13,000 = 26 × 500 → OK.

## Langkah 6 — Bobot akhir neuron ke-1 (yang dipakai pada *forward pass*)

Setelah 13,000 pembaruan, bobot model dibaca dengan get_weights(). Kernel (10, 90) dipecah menjadi 3 blok berurutan z, r, h, lalu kolom pertama tiap blok diambil sebagai neuron ke-1. Bias neuron ke-1 diambil dari elemen ke-1, ke-31, ke-61 pada baris b_in dan baris b_rec matriks bias (2, 90).

**Tabel bobot akhir neuron ke-1 GRU (setelah pelatihan)**

| Variabel | W_z[:,1] | W_r[:,1] | W_h[:,1] |
|---|---|---|---|
| X1 Close Price (USD) | -0.004894 | -0.466489 | 0.398500 |
| X2 Miners Revenue (USD) | 0.005574 | -0.363806 | -0.016380 |
| X3 Difficulty | -0.035411 | -0.198020 | 0.119456 |
| X4 Hash Rate | -0.024312 | -0.155058 | 0.269774 |
| X5 Median Confirmation Time | 0.053985 | -0.359717 | -0.034389 |
| X6 Average Block Size | -0.125109 | -0.476919 | -0.098092 |
| X7 Total Unique Addresses | -0.004999 | -0.418683 | -0.104376 |
| X8 Transaction per Block | -0.387106 | -0.101683 | -0.147950 |
| X9 Confirmed Transaction | 0.032508 | -0.253078 | -0.151735 |
| X10 Cost % per Transaction | -0.225453 | -0.349738 | 0.017280 |
| Bias b_in[1] | 0.211800 | -0.249689 | 0.000974 |
| Bias b_rec[1] | 0.211800 | -0.249689 | 0.007737 |

**Mengapa b_in dan b_rec gerbang z dan r bernilai sama?** Pada gerbang z dan r, kedua bias langsung dijumlahkan di dalam sigmoid, sehingga gradien keduanya selalu sama. Karena sama-sama berawal dari 0, keduanya bergeser bersamaan dan tetap kembar. Pada kandidat h, b_rec dikalikan dengan r terlebih dahulu, sehingga gradiennya berbeda dan nilainya berpisah (selisih terbesar b_in − b_rec gerbang z dan r = 1.8e-07).

**Pemeriksaan:** bobot hasil pelatihan ulang dibandingkan dengan bobot model terbaik dari Tahap 8.1 untuk seluruh 3,780 parameter lapisan rekuren. Selisih mutlak terbesarnya 0.0e+00 → identik. Tabel di atas sama persis dengan bobot yang dipakai pada *forward pass* Tahap 8.

## Kesimpulan

Bobot GRU neuron ke-1 berasal dari bilangan acak Glorot uniform U(−0.244949, +0.244949) dengan n_in = 10 dan n_out = 3 × 30 = 90, yang dibangkitkan dengan seed 42. Bobot itu kemudian diperbarui oleh Adam sebanyak 26 batch × 500 epoch = 13,000 kali berdasarkan gradien BPTT dari loss MSE. Satu langkah Adam yang dihitung manual sama dengan hasil Keras, dan pelatihan ulang menghasilkan bobot yang identik dengan model terbaik, sehingga seluruh nilai bobot pada Bab III dapat ditelusuri asal-usulnya.
