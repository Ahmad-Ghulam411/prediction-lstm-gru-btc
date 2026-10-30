# Perhitungan Manual - Tahap 7.6: Fungsi Aktivasi dan Alur Sel LSTM pada Model Utama (Gambar 1-6)

Bagian ini menjelaskan alur **Gambar 1 sampai Gambar 6** (subbab 1.5.8) memakai model LSTM
yang benar-benar dipakai penelitian. Kita jalankan satu jendela data asli melewati sel LSTM,
hari demi hari, lalu melihat apa yang dikerjakan setiap gerbang.

**Semua angka di sini berasal dari model LSTM utama penelitian**, yaitu model terbaik hasil
tuning: **40 neuron**, masukan **7 hari × 10 fitur**, batch
**32**, dan dilatih 500 epoch. Tidak ada model mini atau angka
karangan. Karena angkanya banyak (8,201 parameter), triknya begini: kita ikuti
**satu neuron (neuron ke-1)** dan **satu sampel** secara rinci sampai bisa dicek dengan
kalkulator, sedangkan neuron dan sampel lainnya dihitung komputer dengan rumus yang sama
lalu dicocokkan dengan Keras.

**Daftar isi**

- [1. Model dan Data yang Dipakai](#1-model-dan-data-yang-dipakai)
- [2. Cara Membaca Gambar](#2-cara-membaca-gambar)
- [3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)](#3-sigmoid-dan-tanh-penjabaran-persamaan-11-13)
- [4. Gambar 1: Satu Sel LSTM Menerima Apa Saja](#4-gambar-1-satu-sel-lstm-menerima-apa-saja)
- [5. Time Step t = 1, Neuron ke-1](#5-time-step-t--1-neuron-ke-1)
- [6. Time Step t = 2: Memori Mulai Bekerja](#6-time-step-t--2-memori-mulai-bekerja)
- [7. Tujuh Time Step Neuron ke-1](#7-tujuh-time-step-neuron-ke-1)
- [8. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)](#8-dari-seluruh-neuron-ke-prediksi-lapisan-dense)
- [9. Ringkasan Satu Langkah LSTM](#9-ringkasan-satu-langkah-lstm)
- [10. Catatan untuk Naskah Skripsi](#10-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Model dan Data yang Dipakai

**Arsitektur:** Input (7 × 10) → LSTM (40 neuron) → Dense (1), total
**8,201 parameter** (rinciannya di [Tahap 7.4 (Jumlah Parameter Model LSTM)](tahap_07_04_jumlah_parameter_lstm.md)).

**Bobot yang dipakai adalah bobot awal**, yaitu bobot tepat sebelum epoch pertama dimulai.
Bobot ini dibuat ulang dengan `bangun_model("LSTM", 40, seed=42)`. Seed dikunci sebelum
bobot diundi, jadi hasilnya sama persis dengan bobot awal model terbaik waktu dilatih di
Tahap 7.1 (notebook memeriksanya dengan membangun model dua kali dan membandingkan semua
bobotnya). Kenapa bobot awal, bukan bobot hasil latihan? Karena kita ingin melihat apa yang
terjadi **saat pelatihan dimulai**. Bobot hasil latihan dipakai di [Tahap 7.11 (Forward Pass LSTM (Jendela Pertama Data Uji))](tahap_07_11_forward_pass_lstm.md).

| Bobot | Bentuk | Cara Keras mengisinya | Nilai pada model ini |
|---|---|---|---|
| kernel W (masukan) | (10, 160) = 10 fitur × 4 gerbang × 40 neuron | acak Glorot uniform, rentang ±√(6/(fan_in+fan_out)) = ±0.1879 | -0.1878 s.d. 0.1878 |
| recurrent kernel U | (40, 160) | acak ortogonal (baris-barisnya saling tegak lurus) | -0.2657 s.d. 0.3101; cek ortogonal: selisih 5.4e-07 |
| bias b | (160,) | nol, kecuali forget gate = 1 (*unit forget bias*) | b_f = 1 untuk semua neuron; b_i = b_c = b_o = 0 |
| Dense W_y | (40, 1) | Glorot uniform, ±0.3825 | -0.3518 s.d. 0.3681 |
| Dense b_y | (1,) | nol | 0 |

Jadi bobot awal memang acak, tetapi tidak sembarangan: rentangnya dibatasi supaya nilai
gerbang tidak langsung jenuh di 0 atau 1, dan bias forget gate dibuat 1 supaya di awal
pelatihan sel cenderung **menyimpan** memori, bukan membuangnya.

**Data:** jendela pertama data latih (`X_latih[0]`, sama dengan Tahap 6.2), nilai
ternormalisasi:

| Hari | Tanggal | X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 | X10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 1 | 2023-07-01 | 0.0541 | 0.0778 | 0.0000 | 0.0408 | 0.5139 | 0.4820 | 0.4001 | 0.2466 | 0.1195 | 0.1251 |
| t = 2 | 2023-07-02 | 0.0553 | 0.1639 | 0.0000 | 0.1535 | 0.3832 | 0.6123 | 0.2669 | 0.0935 | 0.0986 | 0.2833 |
| t = 3 | 2023-07-03 | 0.0556 | 0.0985 | 0.0000 | 0.0527 | 0.6993 | 0.5825 | 0.4506 | 0.2360 | 0.1252 | 0.1170 |
| t = 4 | 2023-07-04 | 0.0610 | 0.1059 | 0.0000 | 0.0675 | 0.6256 | 0.5379 | 0.4357 | 0.3023 | 0.1988 | 0.1418 |
| t = 5 | 2023-07-05 | 0.0572 | 0.1247 | 0.0000 | 0.1060 | 0.5074 | 0.5117 | 0.5434 | 0.4290 | 0.3667 | 0.1533 |
| t = 6 | 2023-07-06 | 0.0544 | 0.0888 | 0.0000 | 0.0586 | 0.5287 | 0.4781 | 0.5320 | 0.3412 | 0.2195 | 0.0741 |
| t = 7 | 2023-07-07 | 0.0484 | 0.1666 | 0.0000 | 0.1653 | 0.4277 | 0.5982 | 0.6295 | 0.4274 | 0.4581 | 0.1722 |

Targetnya adalah Close Price ternormalisasi tanggal **2023-07-08**:
y′ = **0.052770**, setara 0.052770 × 98,196.75 + 25,162.70 =
**30,344.53 USD**. Keadaan awal memori mengikuti Keras: h₀ = 0
dan c₀ = 0 untuk semua neuron.

## 2. Cara Membaca Gambar

Gambar 1-6 berasal dari Olah (2015). Simbolnya:

| Simbol | Arti |
|---|---|
| Kotak kuning (σ atau tanh) | lapisan yang **punya bobot dan bias** (W, U, b) dan ikut dilatih |
| Lingkaran merah muda (×, +) | operasi elemen demi elemen **tanpa bobot** (× = perkalian Hadamard ⊙) |
| Oval "tanh" (Gambar 1 dan 6) | fungsi tanh biasa tanpa bobot, hanya memampatkan nilai ke −1 sampai 1 |
| Dua garis menyatu | hₜ₋₁ dan xₜ masuk bersama ke satu gerbang (di skripsi: xₜW + hₜ₋₁U + b) |
| Garis paling atas | cell state cₜ (memori jangka panjang) |

Gampangnya: **σ (0 sampai 1) itu keran**, menentukan berapa persen sesuatu boleh lewat.
**tanh (−1 sampai 1) itu isi informasinya**, bisa positif atau negatif, dan tidak bisa
meledak karena dibatasi.

## 3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)

Kotak kuning σ dan tanh pada Gambar 1-6 memakai dua fungsi aktivasi, yaitu
persamaan (11) dan (12). Persamaan (13) menghubungkan keduanya. Bagian ini
menunjukkan cara **memperoleh persamaan (13) sendiri** dari (11) dan (12), seolah-olah
kita belum pernah melihatnya.

**Penting:** persamaan (12) **bukan** hasil olahan persamaan (11). Keduanya definisi
yang berdiri sendiri. Persamaan (13) adalah **jembatan** yang membuktikan bahwa tanh
sebenarnya sigmoid yang dipercuram, direntangkan, lalu digeser.

### 3.1 Sigmoid, Persamaan (11)

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Perilakunya ditentukan oleh e^(−z):

| z | e^(−z) | σ(z) = 1 / (1 + e^(−z)) | Keterangan |
|---|---|---|---|
| 10 | 0.000045 | 0.999955 | z besar → e^(−z) ≈ 0 → σ ≈ 1 |
| 2 | 0.135335 | 0.880797 |  |
| 0 | 1.000000 | 0.500000 | tepat di tengah |
| -2 | 7.389056 | 0.119203 |  |
| -10 | 22026.465795 | 0.000045 | z sangat negatif → e^(−z) sangat besar → σ ≈ 0 |

Berapa pun nilai z, hasil sigmoid selalu di antara **0 dan 1**, dan σ(0) = 0.5.

### 3.2 Tanh, Persamaan (12)

$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$

| z | e^z | e^(−z) | tanh(z) |
|---|---|---|---|
| 10 | 22026.465795 | 0.000045 | 1.000000 |
| 2 | 7.389056 | 0.135335 | 0.964028 |
| 0 | 1.000000 | 1.000000 | 0.000000 |
| -2 | 0.135335 | 7.389056 | -0.964028 |
| -10 | 0.000045 | 22026.465795 | -1.000000 |

Hasil tanh selalu di antara **−1 dan 1**, tanh(0) = 0, dan simetris: tanh(−z) = −tanh(z).
Bentuk kurvanya sama-sama huruf S seperti sigmoid; yang berbeda hanya rentangnya.

### 3.3 Cara 1: Mulai dari Tanh, Cari Bentuk Sigmoid

Ciri khas sigmoid adalah pola **1 / (1 + e^(−sesuatu))**. Tugasnya: ubah rumus tanh
sedikit demi sedikit sampai pola itu muncul di dalamnya.

**Langkah 1 — tulis rumus tanh (12).**

$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$

**Langkah 2 — buat penyebut diawali angka 1.** Penyebut sigmoid berbentuk "1 + …",
sedangkan penyebut tanh diawali e^z. Karena itu **pembilang dan penyebut dibagi e^z**.
Ini boleh, karena membagi atas dan bawah pecahan dengan bilangan yang sama tidak
mengubah nilainya, dan e^z tidak pernah 0. Aturan yang dipakai: e^p / e^q = e^(p−q),
sehingga e^z / e^z = e^0 = 1 dan e^(−z) / e^z = e^(−z−z) = e^(−2z).

$$\tanh(z) = \frac{\dfrac{e^{z}}{e^{z}} - \dfrac{e^{-z}}{e^{z}}}{\dfrac{e^{z}}{e^{z}} + \dfrac{e^{-z}}{e^{z}}} = \frac{1-e^{-2z}}{1+e^{-2z}}$$

**Langkah 3 — kenali pola sigmoid.** Penyebut 1 + e^(−2z) persis sama dengan penyebut
sigmoid (11), asalkan z di sigmoid diganti **2z**. Jadi yang akan muncul adalah σ(2z),
bukan σ(z). **Inilah asal angka 2 di dalam kurung** pada persamaan (13).

$$\sigma(2z) = \frac{1}{1+e^{-2z}}$$

Agar ringkas, misalkan **u = e^(−2z)**, sehingga:

$$\tanh(z) = \frac{1-u}{1+u}, \qquad \sigma(2z) = \frac{1}{1+u}$$

**Langkah 4 — munculkan (1 + u) di pembilang.** Pembilang tanh adalah 1 − u, sedangkan
pembilang sigmoid adalah 1. Agar bisa dicoret dengan penyebut, (1 + u) harus muncul di
pembilang. Caranya dengan **trik tambah-kurang**: tulis 1 sebagai 2 − 1, sehingga
1 − u = 2 − 1 − u = 2 − (1 + u).

$$\tanh(z) = \frac{2-(1+u)}{1+u}$$

**Langkah 5 — pecah menjadi dua pecahan** (penyebutnya sama), lalu coret (1 + u)/(1 + u) = 1.

$$\tanh(z) = \frac{2}{1+u} - \frac{1+u}{1+u} = 2\cdot\frac{1}{1+u} - 1$$

**Langkah 6 — ganti 1/(1 + u) dengan σ(2z)** dari Langkah 3. Persamaan (13) muncul:

$$\tanh(z) = 2\sigma(2z) - 1 \qquad (13)$$

| Langkah | Yang dilakukan | Alasannya |
|---|---|---|
| 1 | Tulis tanh (12) | titik awal |
| 2 | Bagi pembilang dan penyebut dengan e^z | agar penyebut diawali 1, seperti sigmoid |
| 3 | Kenali 1 + e^(−2z) | itu penyebut σ(2z); asal angka 2 |
| 4 | 1 − u = 2 − (1 + u) | agar (1 + u) muncul di pembilang |
| 5 | Pecah pecahan | (1 + u)/(1 + u) = 1 |
| 6 | Ganti 1/(1 + u) dengan σ(2z) | hasil akhir: persamaan (13) |

### 3.4 Cara 2: Mulai dari Sigmoid, Cari Bentuk Tanh

Arahnya dibalik. Pangkat pada tanh berpasangan (+z dan −z), jadi sigmoid dibuat
berpasangan juga.

**Langkah 1 — kalikan pembilang dan penyebut sigmoid dengan e^(z/2).** Penyebutnya
menjadi e^(z/2) + e^(−z) · e^(z/2) = e^(z/2) + e^(−z/2), yang pangkatnya berpasangan
seperti tanh.

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^{z/2}}{e^{z/2}+e^{-z/2}}$$

**Langkah 2 — ubah pembilang menjadi selisih.** Pembilang tanh berupa selisih, sedangkan
pembilang di atas hanya satu suku. Rentang nilainya memberi petunjuk: sigmoid bernilai
0 sampai 1, tanh −1 sampai 1. Untuk memindahkan 0..1 ke −1..1, **kalikan 2** (menjadi
0..2) lalu **kurangi 1** (menjadi −1..1). Jadi hitung 2σ(z) − 1:

$$2\sigma(z) - 1 = \frac{2e^{z/2} - \left(e^{z/2}+e^{-z/2}\right)}{e^{z/2}+e^{-z/2}} = \frac{e^{z/2}-e^{-z/2}}{e^{z/2}+e^{-z/2}} = \tanh\left(\frac{z}{2}\right)$$

**Langkah 3 — ganti z dengan 2z** di kedua ruas. Hasilnya sama dengan Cara 1:

$$2\sigma(2z) - 1 = \tanh(z)$$

### 3.5 Cara 3: Tebak dari Bentuk Grafik, Lalu Buktikan

Cara ini kira-kira yang dilakukan seseorang yang belum tahu rumusnya.

```
1. Samakan rentang. Sigmoid 0..1, tanh −1..1 → tebakan pertama: 2σ(z) − 1.

2. Uji dengan angka, z = 0.5:
     2σ(0.5) − 1 = 2 × 0.622459 − 1 = 0.244919
     tanh(0.5)   = 0.462117          → tidak sama!

3. Cari penyebabnya lewat kemiringan (turunan) di z = 0:
     kemiringan σ(z)        = σ(0)(1 − σ(0)) = 0.5 × 0.5 = 0.25
     kemiringan 2σ(z) − 1   = 2 × 0.25 = 0.50
     kemiringan tanh(z)     = 1 − tanh²(0) = 1.00
   Tebakan pertama 2 kali kurang curam → ganti z dengan 2z.

4. Tebakan kedua: 2σ(2z) − 1
     2σ(1) − 1 = 2 × 0.731059 − 1 = 0.462117 = tanh(0.5) ✓

5. Tebakan yang cocok ini lalu dibuktikan secara aljabar dengan Cara 1.
```

### 3.6 Arti Persamaan (13)

Persamaan (13) mengubah sigmoid menjadi tanh dengan tiga operasi:

| Operasi | Bentuk | Rentang | Nilai di z = 0 | Kemiringan di z = 0 |
|---|---|---|---|---|
| mulai dari sigmoid | σ(z) | 0 sampai 1 | 0.5 | 0.25 |
| ① ganti z dengan 2z | σ(2z) | 0 sampai 1, 2 kali lebih curam | 0.5 | 0.50 |
| ② kalikan 2 | 2σ(2z) | 0 sampai 2 | 1.0 | 1.00 |
| ③ kurangi 1 | 2σ(2z) − 1 | **−1 sampai 1** | **0.0** = tanh(0) | **1.00** = kemiringan tanh |

Singkatnya, tanh adalah sigmoid yang **dibuat 2 kali lebih curam, direntangkan 2 kali ke
atas, lalu digeser turun 1**.

### 3.7 Cek dengan Angka Model Utama

Kandidat cell state neuron ke-1 pada t = 1 (bagian 5) memakai tanh dengan pra-aktivasi
a = 0.013076. Kedua rumus memberi hasil yang sama:

```
Persamaan (12): (e^a − e^(−a)) / (e^a + e^(−a))
  = (1.013162 − 0.987009) / (1.013162 + 0.987009) = 0.013076
Persamaan (13): 2σ(2a) − 1 = 2 × σ(0.026153) − 1 = 2 × 0.506538 − 1 = 0.013076
Nilai c̃₁ dari model = 0.013076 ✓
```

### 3.8 Kaitan dengan Turunan pada BPTT

Dengan s = σ(2z), persamaan (13) memberi 1 − tanh²(z) = 4s(1 − s). Artinya turunan kedua
fungsi cukup dihitung dari nilai keluarannya sendiri: σ′ = σ(1 − σ) dan tanh′ = 1 − tanh².
Dua rumus ini dipakai terus di [Tahap 7.8 (Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time)](tahap_07_08_pelatihan_lstm_bptt.md).

## 4. Gambar 1: Satu Sel LSTM Menerima Apa Saja

Pada setiap hari t, setiap neuron menerima dua hal: **xₜ** (10 nilai fitur hari itu) dan
**hₜ₋₁** (40 nilai hidden state dari hari sebelumnya, satu dari setiap neuron). Keduanya
masuk ke empat kotak kuning: forget gate, input gate, kandidat, dan output gate. Untuk
neuron ke-j, isi kotaknya selalu berpola sama:

$$a_j = \underbrace{\sum_{k=1}^{10} x_{t,k}\,W[k,j]}_{\text{bagian masukan}}
+ \underbrace{\sum_{m=1}^{40} h_{t-1,m}\,U[m,j]}_{\text{bagian memori}} + b_j$$

lalu a dimasukkan ke σ (untuk gerbang) atau tanh (untuk kandidat). Satu neuron per gerbang
butuh 10 + 40 = 50 perkalian, jadi satu hari untuk seluruh sel butuh
4 × 40 × 50 = **8,000 perkalian**. Inilah sebabnya
kita merinci satu neuron saja; neuron lain caranya persis sama.

## 5. Time Step t = 1, Neuron ke-1

Di hari pertama belum ada memori (h₀ = 0, c₀ = 0), jadi bagian memori ΣhU bernilai 0.
Contoh lengkap untuk **forget gate** (Gambar 3) neuron ke-1: kalikan setiap fitur hari
pertama (2023-07-01) dengan bobotnya, lalu jumlahkan.

| Kode | Fitur | x₁ | W_f[k, 1] | x₁ × W_f |
|---|---|---|---|---|
| X1 | Close Price (USD) | 0.054063 | (-0.171849) | (-0.009291) |
| X2 | Miners Revenue (USD) | 0.077789 | (-0.114586) | (-0.008914) |
| X3 | Difficulty | 0.000000 | (-0.039092) | 0.000000 |
| X4 | Hash Rate | 0.040839 | 0.003295 | 0.000135 |
| X5 | Median Confirmation Time | 0.513902 | 0.035027 | 0.018000 |
| X6 | Average Block Size | 0.481952 | (-0.041692) | (-0.020094) |
| X7 | Total Unique Addresses | 0.400071 | (-0.047612) | (-0.019048) |
| X8 | Transaction per Block | 0.246577 | 0.039421 | 0.009720 |
| X9 | Confirmed Transaction | 0.119541 | 0.031904 | 0.003814 |
| X10 | Cost % per Transaction | 0.125072 | 0.104081 | 0.013018 |
|  | **jumlah** |  |  | **-0.012659** |

Tiga gerbang lainnya dihitung dengan cara yang sama persis, hanya bobotnya yang beda:

| Gerbang | Σ x₁·W | Σ h₀·U | b | a = jumlah | Hasil |
|---|---|---|---|---|---|
| forget f₁ | -0.012659 | 0 | 1 | 0.987341 | σ(0.987341) = **0.728562** |
| input i₁ | 0.207530 | 0 | 0 | 0.207530 | σ(0.207530) = **0.551697** |
| kandidat c̃₁ | 0.013076 | 0 | 0 | 0.013076 | tanh(0.013076) = **0.013076** |
| output o₁ | 0.057016 | 0 | 0 | 0.057016 | σ(0.057016) = **0.514250** |

```
Gambar 3 (forget): karena c₀ = 0, belum ada memori yang bisa disimpan atau dibuang.
  f₁ × c₀ = 0.728562 × 0 = 0
Gambar 4 (input): usulan isi baru c̃₁ hanya masuk sebagian, sebesar i₁.
  i₁ × c̃₁ = 0.551697 × 0.013076 = 0.007214
Gambar 5 (update cell state):
  c₁ = f₁ × c₀ + i₁ × c̃₁ = 0 + 0.007214 = 0.007214
Gambar 6 (output):
  tanh(c₁) = tanh(0.007214) = 0.007214
  h₁ = o₁ × tanh(c₁) = 0.514250 × 0.007214 = 0.003710
```

Perhatikan forget gate: biasnya 1, jadi meskipun masukannya kecil, f₁ = 0.728562 sudah lebih
dari 0.5. Sel "disetel" untuk menyimpan memori sejak awal. Semua 40 neuron mengerjakan hal
yang sama secara bersamaan, sehingga hasil t = 1 adalah vektor h₁ dan c₁ yang masing-masing
berisi 40 angka.

## 6. Time Step t = 2: Memori Mulai Bekerja

Mulai hari kedua, bagian memori tidak nol lagi. Neuron ke-1 sekarang menerima h₁ dari
**seluruh 40 neuron**: h₁ = [0.003710, 0.005407, -0.004322, 0.002929, …]. Contoh bagian memori untuk
forget gate:

```
  h₁·U_f[:,1] = 0.003710×0.063310 + 0.005407×(-0.061390) + (-0.004322)×(-0.017715) + … (37 suku lagi) = 0.024083
```

| Gerbang | Σ x₂·W (10 suku) | Σ h₁·U (40 suku) | b | a = jumlah | Hasil |
|---|---|---|---|---|---|
| forget f₂ | -0.016273 | 0.024083 | 1 | 1.007810 | σ(1.007810) = **0.732591** |
| input i₂ | 0.202496 | -0.021964 | 0 | 0.180532 | σ(0.180532) = **0.545011** |
| kandidat c̃₂ | -0.022783 | 0.007657 | 0 | -0.015125 | tanh(-0.015125) = **-0.015124** |
| output o₂ | 0.012791 | 0.006004 | 0 | 0.018795 | σ(0.018795) = **0.504699** |

```
Memori lama yang disimpan (Gambar 3): f₂ × c₁ = 0.732591 × 0.007214 = 0.005285
Informasi baru yang ditulis (Gambar 4): i₂ × c̃₂ = 0.545011 × (-0.015124) = -0.008243
Cell state baru (Gambar 5): c₂ = 0.005285 + (-0.008243) = -0.002958
Hidden state (Gambar 6): h₂ = o₂ × tanh(c₂) = 0.504699 × (-0.002958) = -0.001493
```

Di sini terlihat peran forget gate: 73.3% isi memori kemarin dibawa ke hari ini, lalu
ditambah informasi baru yang lolos lewat input gate sebesar 54.5%.

## 7. Tujuh Time Step Neuron ke-1

Hari ketiga sampai ketujuh dihitung dengan cara yang sama. Hasil neuron ke-1:

| t | Tanggal | fₜ | iₜ | c̃ₜ | cₜ | oₜ | hₜ |
|---|---|---|---|---|---|---|---|
| 1 | 2023-07-01 | 0.728562 | 0.551697 | 0.013076 | 0.007214 | 0.514250 | 0.003710 |
| 2 | 2023-07-02 | 0.732591 | 0.545011 | -0.015124 | -0.002958 | 0.504699 | -0.001493 |
| 3 | 2023-07-03 | 0.735446 | 0.555712 | 0.061326 | 0.031904 | 0.516365 | 0.016469 |
| 4 | 2023-07-04 | 0.740359 | 0.544609 | 0.043981 | 0.047573 | 0.520631 | 0.024749 |
| 5 | 2023-07-05 | 0.743716 | 0.533914 | 0.018632 | 0.045329 | 0.533627 | 0.024172 |
| 6 | 2023-07-06 | 0.744538 | 0.535738 | 0.032891 | 0.051370 | 0.530242 | 0.027214 |
| 7 | 2023-07-07 | 0.745666 | 0.531013 | -0.009690 | 0.033159 | 0.541088 | 0.017936 |

Forget gate neuron ke-1 berada di kisaran 0.733 sampai 0.746.
Kalau dikalikan terus dari hari ke-2 sampai ke-7, porsi isi memori hari pertama yang masih
tersisa di hari terakhir kira-kira 16.5%. Itulah "memori jangka
panjang" pada LSTM: informasi lama tidak hilang mendadak, tetapi meluruh pelan-pelan sesuai
nilai forget gate. Setelah dilatih, nilai-nilai gerbang ini akan berubah karena bobotnya
berubah.

## 8. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)

Setelah hari ke-7, hidden state h₇ dari **40 neuron** dikirim ke lapisan Dense
(persamaan 21). Dense hanya menjumlahkan h₇ yang sudah diberi bobot:

```
  ŷ′ = Σ h₇[j] × W_y[j] + b_y   (j = 1 … 40)
     h₇[1] × W_y[1] = 0.017936 × 0.241729 = 0.004336
     h₇[2] × W_y[2] = 0.039134 × 0.104388 = 0.004085
     h₇[3] × W_y[3] = 0.020056 × 0.000893 = 0.000018
     h₇[4] × W_y[4] = 0.048770 × 0.140193 = 0.006837
     … (36 suku lagi)
  Jumlah 40 suku = 0.041640
  ŷ′ = 0.041640 + 0.000000 = 0.041640

  Dalam USD: 0.041640 × 98,196.75 + 25,162.70 = 29,251.62 USD
  Target   : 0.052770  (30,344.53 USD)
  Selisih dengan model.predict() Keras: 9.65e-10
```

Hasil hitungan manual sama dengan `model.predict()` Keras (selisih 9.6e-10).
Prediksinya meleset 1,092.91 USD dari target. Besar kecilnya meleset di
sampel ini belum berarti apa-apa, karena bobotnya masih acak. Justru di sinilah pelatihan
mulai bekerja: selisih antara ŷ′ dan y′ akan diukur dengan loss
([Tahap 7.7 (Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss)](tahap_07_07_pelatihan_lstm_forward_pass_dan_loss.md)), lalu bobot diperbaiki sedikit demi sedikit.

## 9. Ringkasan Satu Langkah LSTM

| Urutan | Gambar | Rumus (persamaan) | Fungsinya |
|---|---|---|---|
| 1. Forget | Gambar 3 | fₜ = σ(xₜW_f + hₜ₋₁U_f + b_f) (15) | berapa persen memori lama disimpan |
| 2. Input dan kandidat | Gambar 4 | iₜ = σ(…), c̃ₜ = tanh(…) (16, 17) | informasi baru apa dan seberapa banyak |
| 3. Update memori | Gambar 5 | cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ (18) | memori jangka panjang yang baru |
| 4. Output | Gambar 6 | oₜ = σ(…), hₜ = oₜ ⊙ tanh(cₜ) (19, 20) | bagian memori yang dikeluarkan hari ini |
| 5. Prediksi | - | ŷ′ = h₇W_y + b_y (21) | dipakai sekali, setelah hari terakhir |

## 10. Catatan untuk Naskah Skripsi

Kalimat yang bisa dipakai di subbab 1.5.8 setelah Gambar 6:

> Pada penelitian ini setiap sel LSTM terdiri atas 40 neuron. Pada setiap hari t, setiap
> neuron menerima 10 nilai fitur xₜ dan 40 nilai hidden state hₜ₋₁, menghitung
> keempat gerbang menurut persamaan (15)-(20), lalu setelah hari ke-7 hidden state h₇
> diteruskan ke lapisan dense untuk menghasilkan prediksi harga penutupan hari berikutnya.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.4: Jumlah Parameter Model LSTM](tahap_07_04_jumlah_parameter_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.7: Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_07_07_pelatihan_lstm_forward_pass_dan_loss.md)
