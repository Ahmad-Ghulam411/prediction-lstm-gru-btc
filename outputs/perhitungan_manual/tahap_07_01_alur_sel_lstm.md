# Perhitungan Manual - Tahap 7.1: Alur Sel LSTM (Gambar 1-6) dengan Model Mini

Subbagian ini menjelaskan alur **Gambar 1 sampai Gambar 6** pada subbab 1.5.8
langkah demi langkah: apa yang mengalir di setiap garis, apa yang dikerjakan setiap
kotak, dan apa fungsinya. Sebelumnya, bagian 3 menjabarkan fungsi aktivasi sigmoid
dan tanh beserta hubungan keduanya (persamaan 11-13). Setiap gerbang diberi contoh
angka dari model mini pada **time step t = 2**, karena di sana memori dari t = 1
sudah ikut bekerja.

**Daftar isi**

- [1. Model Mini yang Dipakai](#1-model-mini-yang-dipakai)
- [2. Cara Membaca Gambar](#2-cara-membaca-gambar)
- [3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)](#3-sigmoid-dan-tanh-penjabaran-persamaan-11-13)
- [4. Gambar 1: Struktur LSTM](#4-gambar-1-struktur-lstm)
- [5. Gambar 2: Cell State](#5-gambar-2-cell-state)
- [6. Gambar 3: Forget Gate](#6-gambar-3-forget-gate)
- [7. Gambar 4: Input Gate](#7-gambar-4-input-gate)
- [8. Gambar 5: Pembaruan Cell State](#8-gambar-5-pembaruan-cell-state)
- [9. Gambar 6: Output Gate](#9-gambar-6-output-gate)
- [10. Ringkasan Satu Langkah LSTM](#10-ringkasan-satu-langkah-lstm)
- [11. Catatan untuk Naskah Skripsi](#11-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Model Mini yang Dipakai

Supaya setiap langkah punya angka nyata yang bisa diikuti dengan tangan, seluruh
subbagian perhitungan manual pelatihan LSTM memakai **model mini** yang sama:
**1 neuron, 1 fitur, 2 time step, dan 1 sampel**. Rumus dan urutan langkahnya
identik dengan model penelitian (10 fitur, 7 time step,
batch 32); yang berbeda hanya jumlah angkanya.

**Data** (harga penutupan ternormalisasi, angka ilustrasi):

- hari 1: x₁ = 0.5
- hari 2: x₂ = 0.6
- target hari 3: y′ = 0.7

**Keadaan awal:** h₀ = 0 dan c₀ = 0, sama seperti bawaan Keras.

**Bobot awal** dibuat bulat agar mudah dihitung. Keras sebenarnya mengisi bobot secara
acak (Glorot uniform untuk W, ortogonal untuk U), tetapi aturan biasnya diikuti:
**bias forget gate = 1**, bias lain = 0.

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b (bias) |
|---|---|---|---|
| forget (f) | 0.5 | 0.4 | 1 (unit forget bias) |
| input (i) | 0.6 | 0.3 | 0 |
| kandidat (c̃) | 0.7 | 0.2 | 0 |
| output (o) | 0.4 | 0.5 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 |

Hasil time step t = 1 yang dipakai di contoh berikut adalah h₁ = 0.104941 dan
c₁ = 0.193228. Perhitungan lengkap kedua time step ada di [Tahap 7.2 (Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss)](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md).

## 2. Cara Membaca Gambar

Gambar 1-6 berasal dari Olah (2015). Simbolnya:

| Simbol | Arti |
|---|---|
| Kotak kuning (σ atau tanh) | lapisan jaringan saraf yang **punya bobot dan bias** (W, U, b) dan dipelajari saat pelatihan |
| Lingkaran merah muda (×, +) | operasi elemen demi elemen **tanpa bobot**; × = perkalian Hadamard (⊙), + = penjumlahan |
| Oval merah muda "tanh" (Gambar 1 dan 6) | fungsi tanh biasa **tanpa bobot**, hanya memampatkan nilai ke rentang -1 sampai 1 |
| Dua garis menyatu | hₜ₋₁ dan xₜ menjadi masukan bersama sebuah gerbang (di skripsi ditulis xₜW + hₜ₋₁U) |
| Satu garis bercabang | nilai yang sama disalin ke dua tujuan |
| Cₜ (huruf besar) pada gambar Olah | sama dengan cₜ (cell state) pada persamaan skripsi |

**Mengapa σ dipakai untuk gerbang dan tanh untuk isi?**

- **Sigmoid (0 sampai 1)** berfungsi seperti **keran**: 0 berarti tertutup, 1 berarti
  terbuka penuh, 0.5 berarti setengah. Nilainya adalah *proporsi*, jadi selalu dipakai
  untuk **mengalikan** sesuatu.
- **Tanh (-1 sampai 1)** berfungsi sebagai **isi informasi**. Nilainya bisa positif
  atau negatif sehingga memori bisa dinaikkan atau diturunkan, dan tetap terbatas
  sehingga tidak meledak.

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

### 3.7 Cek dengan Angka Model Mini

Kandidat c̃₂ pada Gambar 4 (bagian 7) memakai tanh dengan pra-aktivasi a = 0.440988.
Ketiga bentuk rumus memberi hasil yang sama:

```
Lewat persamaan (12):
  e^a = e^0.440988 = 1.554242      e^(−a) = 0.643400
  tanh(a) = (1.554242 − 0.643400) / (1.554242 + 0.643400) = 0.910842 / 2.197643 = 0.414463

Lewat hasil Langkah 2 (Cara 1):
  u = e^(−2a) = e^(−0.881976) = 0.413964
  (1 − u) / (1 + u) = 0.586036 / 1.413964 = 0.414463

Lewat persamaan (13):
  σ(2a) = σ(0.881976) = 1 / (1 + 0.413964) = 0.707232
  2σ(2a) − 1 = 2 × 0.707232 − 1 = 0.414463

Ketiganya = c̃₂ = 0.414463 ✓
```

### 3.8 Kaitan dengan Turunan pada BPTT

Persamaan (13) juga menghubungkan turunan kedua fungsi. Dengan s = σ(2z):

1 − tanh²(z) = 1 − (2s − 1)² = 1 − (4s² − 4s + 1) = 4s(1 − s)

Jadi turunan tanh pun bisa dihitung dari sigmoid, dan keduanya cukup memakai nilai
keluarannya sendiri: σ′ = σ(1 − σ) dan tanh′ = 1 − tanh². Kedua rumus turunan ini
dipakai di setiap langkah backward pada [Tahap 7.3, bagian 1](tahap_07_03_simulasi_bptt_lstm.md#1-rumus-turunan-dasar).

## 4. Gambar 1: Struktur LSTM

**Alurnya:**

1. Ketiga kotak hijau berlabel **A** adalah **sel yang sama** (bobot yang sama) yang
   digambar berulang untuk setiap waktu: kiri untuk t − 1, tengah untuk t, kanan
   untuk t + 1. Cara menggambar ini disebut *unrolling*.
2. Setiap sel menerima **masukan dari bawah** (xₜ₋₁, xₜ, xₜ₊₁) dan menghasilkan
   **keluaran ke atas** (hₜ₋₁, hₜ, hₜ₊₁).
3. Di antara sel ada **dua garis horizontal** yang membawa memori ke langkah
   berikutnya: garis **atas** adalah **cell state cₜ** (memori jangka panjang), garis
   **bawah** adalah **hidden state hₜ** (memori jangka pendek sekaligus keluaran).
4. Isi sel tengah adalah 4 kotak kuning, dari kiri: **σ (forget gate), σ (input
   gate), tanh (kandidat), σ (output gate)**. Gambar 3-6 membedahnya satu per satu.

**Fungsinya:** RNN biasa (persamaan 10) hanya punya satu garis, yaitu h. LSTM
menambahkan garis cₜ dan tiga gerbang yang mengatur apa yang dibuang, disimpan, dan
dikeluarkan.

**Kaitan dengan penelitian:** dengan *window* 7 hari, sel A dijalankan
**7 kali berturut-turut** (xₜ₋₆ sampai xₜ), dan setiap xₜ berisi
10 fitur blockchain ternormalisasi. Di awal, h₀ = c₀ = 0 (bawaan Keras).
Hanya hidden state **langkah terakhir h_T** yang diteruskan ke lapisan dense
(persamaan 21). Model mini dijalankan 2 kali (t = 1 dan t = 2), lalu h₂ masuk ke dense.

## 5. Gambar 2: Cell State

**Alurnya:**

1. cₜ₋₁ masuk dari kiri.
2. cₜ₋₁ melewati lingkaran **×**: sebagian memori lama dihapus (dikalikan fₜ).
3. Hasilnya melewati lingkaran **+**: informasi baru ditambahkan (iₜ × c̃ₜ).
4. Keluar ke kanan sebagai **cₜ** dan masuk ke langkah waktu berikutnya.

**Fungsinya:** cell state adalah **jalan tol memori**. Di sepanjang garis ini hanya
ada perkalian dan penjumlahan sederhana; tidak ada perkalian matriks bobot dan tidak
ada fungsi aktivasi yang berulang.

Inilah alasan LSTM tahan terhadap *vanishing gradient*. Saat *backpropagation*,
gradien yang mengalir mundur di jalur ini hanya dikalikan fₜ. Jika model memutuskan
untuk mengingat (fₜ mendekati 1), gradien hampir tidak mengecil. Buktinya ada di
[Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md): **86.4%** sinyal kesalahan yang
sampai ke c₁ datang lewat jalur cell state ini.

## 6. Gambar 3: Forget Gate

**Alurnya:**

1. **hₜ₋₁** (masuk dari kiri) dan **xₜ** (masuk dari bawah) bertemu dan menjadi
   masukan bersama.
2. Keduanya masuk ke **kotak kuning σ pertama** dan dihitung pra-aktivasinya:
   xₜW_f + hₜ₋₁U_f + b_f (persamaan 15).
3. Hasilnya dilewatkan ke sigmoid sehingga menjadi **fₜ**, bernilai 0 sampai 1.
4. Panah fₜ naik ke lingkaran **×** di garis cell state dan **mengalikan cₜ₋₁**.

**Fungsinya:** menentukan **berapa banyak memori lama yang dipertahankan**. fₜ mendekati
0 berarti memori dihapus; fₜ mendekati 1 berarti memori dipertahankan utuh.

```
Contoh angka (t = 2), memakai h₁ = 0.104941 dan c₁ = 0.193228 dari t = 1:
  a  = W_f × x₂ + U_f × h₁ + b_f = 0.5 × 0.6 + 0.4 × 0.104941 + 1 = 1.341976
  f₂ = σ(1.341976) = 0.792815
  Memori lama yang dipertahankan: f₂ × c₁ = 0.792815 × 0.193228 = 0.153194
  Artinya 79.3% isi memori lama dipertahankan dan 20.7% dibuang.
```

Bias forget bernilai 1 (*unit forget bias* Keras). Karena itu, walaupun xₜ dan hₜ₋₁
bernilai nol, gerbang ini tetap bernilai σ(1) = 0.731: secara bawaan model
cenderung **mengingat**. Peran bias dijelaskan lengkap di [Tahap 7.12 (Bias pada LSTM: Peran dan Cara Menghitungnya)](tahap_07_12_bias_lstm.md).

## 7. Gambar 4: Input Gate

Gambar ini memiliki **dua cabang paralel** dari masukan yang sama (hₜ₋₁ dan xₜ).

**Alurnya:**

1. **Cabang kiri, kotak σ kedua**, menghasilkan **iₜ** (0 sampai 1, persamaan 16):
   keran yang menentukan **berapa banyak** informasi baru boleh masuk.
2. **Cabang kanan, kotak tanh**, menghasilkan **c̃ₜ** (-1 sampai 1, persamaan 17):
   **isi informasi baru** yang diusulkan.
3. Kedua cabang bertemu di lingkaran **×** sehingga menjadi iₜ × c̃ₜ.
4. Hasilnya naik ke lingkaran **+** di garis cell state.

**Fungsinya:** menentukan **informasi baru apa yang ditulis ke memori**. c̃ₜ menjawab
"apa isinya?", sedangkan iₜ menjawab "seberapa penting untuk disimpan?". Karena c̃ₜ
bisa negatif, informasi baru bisa menaikkan atau menurunkan memori.

```
Contoh angka (t = 2):
  i₂ = σ(W_i × x₂ + U_i × h₁ + b_i) = σ(0.6 × 0.6 + 0.3 × 0.104941 + 0) = σ(0.391482) = 0.596639
  c̃₂ = tanh(W_c × x₂ + U_c × h₁ + b_c) = tanh(0.7 × 0.6 + 0.2 × 0.104941 + 0) = tanh(0.440988) = 0.414463
  Informasi baru yang ditulis: i₂ × c̃₂ = 0.596639 × 0.414463 = 0.247285
  Artinya usulan isi 0.414463 hanya masuk 59.7%.
```

## 8. Gambar 5: Pembaruan Cell State

Gambar ini menggabungkan hasil Gambar 3 dan Gambar 4 di garis atas (persamaan 18):

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

**Alurnya:** cₜ₋₁ dikalikan fₜ (memori lama terseleksi), lalu ditambah iₜ × c̃ₜ (memori
baru terseleksi). Hasilnya adalah **cₜ**, memori jangka panjang yang sudah diperbarui,
yang dikirim ke langkah waktu berikutnya.

**Fungsinya:** di sinilah memori benar-benar ditulis ulang. Karena fₜ dan iₜ
**independen**, LSTM bisa sekaligus mempertahankan banyak memori lama dan menambah
banyak memori baru. Ini salah satu pembeda utama dengan GRU (lihat Tahap 9.2).

```
Contoh angka (t = 2), memakai hasil Gambar 3 dan Gambar 4:
  c₂ = f₂ × c₁ + i₂ × c̃₂ = 0.153194 + 0.247285 = 0.400479
  Memori berubah dari c₁ = 0.193228 menjadi c₂ = 0.400479.
```

## 9. Gambar 6: Output Gate

**Alurnya:**

1. hₜ₋₁ dan xₜ masuk ke **kotak σ keempat** dan menghasilkan **oₜ** (0 sampai 1,
   persamaan 19).
2. Sementara itu, **cₜ** dari garis atas bercabang turun ke **oval tanh** (tanpa bobot)
   yang memampatkan cₜ ke rentang -1 sampai 1.
3. tanh(cₜ) dan oₜ bertemu di lingkaran **×** sehingga menjadi **hₜ = oₜ ⊙ tanh(cₜ)**
   (persamaan 20).
4. hₜ **bercabang dua**: naik ke atas sebagai keluaran waktu t, dan ke kanan sebagai
   hₜ₋₁ untuk langkah berikutnya.

**Fungsinya:** menentukan **bagian memori mana yang ditampilkan** sebagai keluaran saat
ini. Tidak semua isi cₜ relevan untuk keluaran sekarang; sebagian cukup disimpan untuk
dipakai nanti.

```
Contoh angka (t = 2):
  o₂ = σ(W_o × x₂ + U_o × h₁ + b_o) = σ(0.4 × 0.6 + 0.5 × 0.104941 + 0) = σ(0.292470) = 0.572601
  tanh(c₂) = tanh(0.400479) = 0.380359        (oval tanh, tanpa bobot)
  h₂ = o₂ × tanh(c₂) = 0.572601 × 0.380359 = 0.217794
  t = 2 adalah langkah terakhir, jadi h₂ masuk ke lapisan dense (persamaan 21):
  ŷ' = W_y × h₂ + b_y = 0.8 × 0.217794 + 0 = 0.174235
```

## 10. Ringkasan Satu Langkah LSTM

Seluruh alur satu time step LSTM (t = 2, memori masuk c₁ = 0.193228 dan h₁ = 0.104941):

| Urutan | Gambar | Perhitungan | Hasil | Makna |
|---|---|---|---|---|
| 1. Forget | Gambar 3 | f₂ = σ(a); f₂ × c₁ | 0.792815; 0.153194 | 79.3% memori lama dipertahankan |
| 2. Input | Gambar 4 | i₂ = σ(a); c̃₂ = tanh(a); i₂ × c̃₂ | 0.596639; 0.414463; 0.247285 | usulan baru masuk 59.7% |
| 3. Update | Gambar 5 | c₂ = f₂c₁ + i₂c̃₂ | 0.400479 | memori jangka panjang baru |
| 4. Output | Gambar 6 | o₂ = σ(a); h₂ = o₂ tanh(c₂) | 0.572601; 0.217794 | 57.3% memori yang sudah dimampatkan dikeluarkan |
| 5. Dense | - | ŷ′ = W_y h₂ + b_y | 0.174235 | prediksi (skala ternormalisasi) |

Prediksi ŷ′ = 0.174235 masih jauh dari target 0.7. Seberapa salah prediksi
ini dan bagaimana bobot diperbaiki dihitung pada langkah berikutnya, mulai dari
[Tahap 7.2 (Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss)](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md).

## 11. Catatan untuk Naskah Skripsi

**Persamaan (11)-(13).**

1. Kalimat "Jika fungsi tanh dihubungkan dengan fungsi sigmoid…" sudah tepat, karena
   persamaan (13) memang **hubungan** kedua fungsi, bukan hasil olahan (11) menjadi (12).
2. Bagian *Keterangan* belum menjelaskan tanh(z), padahal simbol ini muncul di persamaan
   (12) dan (13). Sebaiknya ditambahkan:
   `tanh(z) = Fungsi aktivasi tangen hiperbolik`.
3. Jika penjabaran persamaan (13) ingin dicantumkan, versi ringkas Cara 1 (bagian 3.3)
   cukup ditulis dalam beberapa baris:

$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$

$$= \frac{1-e^{-2z}}{1+e^{-2z}} \qquad \text{(pembilang dan penyebut dibagi } e^{z}\text{)}$$

$$= \frac{2-\left(1+e^{-2z}\right)}{1+e^{-2z}}$$

$$= 2\cdot\frac{1}{1+e^{-2z}} - 1$$

$$= 2\sigma(2z) - 1 \qquad \text{(karena } \sigma(2z) = \tfrac{1}{1+e^{-2z}}\text{)}$$

Contoh kalimat pengantarnya:

> Dengan membagi pembilang dan penyebut persamaan (12) dengan e^z, penyebutnya berbentuk
> sama dengan penyebut fungsi sigmoid pada persamaan (11) untuk masukan 2z, sehingga
> diperoleh persamaan (13).

**Notasi Cₜ pada Gambar 1-6.** Gambar Olah (2015) memakai huruf besar Cₜ untuk cell
state, sedangkan persamaan skripsi memakai cₜ. Satu kalimat penjelas dapat mencegah
pertanyaan penguji, misalnya:

> Notasi Cₜ pada Gambar 1-6 sama dengan cₜ pada persamaan (18)-(20).

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 6.2: Pembentukan Sliding Window](tahap_06_02_sliding_window.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.2: Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md)
