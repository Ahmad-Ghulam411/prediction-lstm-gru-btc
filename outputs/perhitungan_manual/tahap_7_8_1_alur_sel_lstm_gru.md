# Perhitungan Manual - Tahap 7 & 8 (Bagian 1 dari 4): Alur Sel LSTM dan GRU (Gambar 1-7)

> Berkas ini dibuat otomatis oleh `tools/simulasi_pelatihan_manual.py`. Semua
> angka dihitung ulang oleh skrip itu dan diperiksa dengan `assert`. Untuk membuat
> ulang seluruh seri: `python tools/simulasi_pelatihan_manual.py`.

**Seri perhitungan manual pelatihan LSTM dan GRU** (baca berurutan):

1. **Alur Sel LSTM dan GRU (Gambar 1-7)** ← sedang dibaca
2. [Fungsi Loss dan Adam](tahap_7_8_2_fungsi_loss_dan_adam.md)
3. [Simulasi Satu Siklus Pelatihan LSTM dan GRU](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md)
4. [Bias pada LSTM dan GRU](tahap_7_8_4_bias_lstm_gru.md)

Bagian ini menjelaskan alur **Gambar 1 sampai Gambar 7** pada subbab 1.5.8 (LSTM)
dan 1.5.9 (GRU) langkah demi langkah: apa yang mengalir di setiap garis, apa yang
dikerjakan setiap kotak, dan apa fungsinya.

Supaya setiap langkah punya angka nyata, contoh angka diambil dari **simulasi
model mini** yang dihitung lengkap di [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md): 1 neuron, 1 fitur, data
x₁ = 0.5 dan x₂ = 0.6, bobot awal bulat (W_f = 0.5, U_f = 0.4,
b_f = 1, dan seterusnya; tabel lengkapnya di [Bagian 3, subbagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md#3-lstm-satu-siklus-pelatihan-langkah-demi-langkah)).
Contoh memakai **time step t = 2**, karena di sana memori dari t = 1 sudah ikut
bekerja. Angka yang sama dipakai lagi di Bagian 2 dan 3 saat menghitung loss,
gradien, dan Adam, sehingga seluruh seri saling menyambung.

**Daftar isi**

- [1. Cara Membaca Gambar](#1-cara-membaca-gambar)
- [2. Gambar 1: Struktur LSTM](#2-gambar-1-struktur-lstm)
- [3. Gambar 2: Cell State](#3-gambar-2-cell-state)
- [4. Gambar 3: Forget Gate](#4-gambar-3-forget-gate)
- [5. Gambar 4: Input Gate](#5-gambar-4-input-gate)
- [6. Gambar 5: Pembaruan Cell State](#6-gambar-5-pembaruan-cell-state)
- [7. Gambar 6: Output Gate](#7-gambar-6-output-gate)
- [8. Ringkasan Satu Langkah LSTM](#8-ringkasan-satu-langkah-lstm)
- [9. Gambar 7: Struktur Sel GRU](#9-gambar-7-struktur-sel-gru)
- [10. Hubungan LSTM dan GRU](#10-hubungan-lstm-dan-gru)
- [11. Catatan untuk Naskah Skripsi](#11-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Seperti berkas perhitungan manual lainnya, angka memakai
titik sebagai pemisah desimal dan koma sebagai pemisah ribuan. Angka ditampilkan
6 desimal, tetapi skrip menghitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Cara Membaca Gambar

Gambar 1-6 berasal dari Olah (2015), dan Gambar 7 mengikuti gaya yang sama. Simbolnya:

| Simbol | Arti |
|---|---|
| Kotak kuning (σ atau tanh) | lapisan jaringan saraf yang **punya bobot dan bias** (W, U, b) dan dipelajari saat pelatihan |
| Lingkaran merah muda (×, +, 1−) | operasi elemen demi elemen **tanpa bobot**; × = perkalian Hadamard (⊙), + = penjumlahan |
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

## 2. Gambar 1: Struktur LSTM

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
menambahkan garis cₜ dan tiga gerbang yang mengatur apa yang dibuang, disimpan,
dan dikeluarkan.

**Kaitan dengan penelitian:** dengan *window* 7 hari, sel A dijalankan
**7 kali berturut-turut** (xₜ₋₆ sampai xₜ), dan setiap xₜ berisi
10 fitur blockchain ternormalisasi. Di awal, h₀ = c₀ = 0 (bawaan Keras).
Hanya hidden state **langkah terakhir h_T** yang diteruskan ke lapisan dense
(persamaan 21). Model mini di seri ini sama, hanya dijalankan 2 kali (t = 1 dan
t = 2), lalu h₂ masuk ke dense.

## 3. Gambar 2: Cell State

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
untuk mengingat (fₜ mendekati 1), gradien hampir tidak mengecil. Buktinya ada pada
perhitungan [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md) subbagian 3.7: **86.4%** sinyal kesalahan yang
sampai ke c₁ datang lewat jalur cell state ini.

## 4. Gambar 3: Forget Gate

**Alurnya:**

1. **hₜ₋₁** (masuk dari kiri) dan **xₜ** (masuk dari bawah) bertemu dan menjadi
   masukan bersama.
2. Keduanya masuk ke **kotak kuning σ pertama** dan dihitung pra-aktivasinya:
   xₜW_f + hₜ₋₁U_f + b_f (persamaan 15).
3. Hasilnya dilewatkan ke sigmoid sehingga menjadi **fₜ**, bernilai 0 sampai 1.
4. Panah fₜ naik ke lingkaran **×** di garis cell state dan **mengalikan cₜ₋₁**.

**Fungsinya:** menentukan **berapa banyak memori lama yang dipertahankan**. fₜ
mendekati 0 berarti memori dihapus; fₜ mendekati 1 berarti memori dipertahankan utuh.

```
Contoh angka (t = 2). Dari t = 1 sudah diperoleh h₁ = 0.104941 dan c₁ = 0.193228.
  a  = W_f × x₂ + U_f × h₁ + b_f = 0.5 × 0.6 + 0.4 × 0.104941 + 1 = 1.341976
  f₂ = σ(1.341976) = 0.792815
  Memori lama yang dipertahankan: f₂ × c₁ = 0.792815 × 0.193228 = 0.153194
  Artinya 79.3% isi memori lama dipertahankan dan 20.7% dibuang.
```

Bias forget bernilai 1 (*unit forget bias* Keras). Karena itu, walaupun xₜ dan hₜ₋₁
bernilai nol, gerbang ini tetap bernilai σ(1) = 0.731: secara bawaan model
cenderung **mengingat**. Peran bias dijelaskan lengkap di [Bagian 4](tahap_7_8_4_bias_lstm_gru.md).

## 5. Gambar 4: Input Gate

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

## 6. Gambar 5: Pembaruan Cell State

Gambar ini menggabungkan hasil Gambar 3 dan Gambar 4 di garis atas (persamaan 18):

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

**Alurnya:** cₜ₋₁ dikalikan fₜ (memori lama terseleksi), lalu ditambah iₜ × c̃ₜ
(memori baru terseleksi). Hasilnya adalah **cₜ**, memori jangka panjang yang sudah
diperbarui, yang dikirim ke langkah waktu berikutnya.

**Fungsinya:** di sinilah memori benar-benar ditulis ulang. Karena fₜ dan iₜ
**independen**, LSTM bisa sekaligus mempertahankan banyak memori lama dan menambah
banyak memori baru. Ini salah satu pembeda utama dengan GRU (subbagian 10).

```
Contoh angka (t = 2), memakai hasil Gambar 3 dan Gambar 4:
  c₂ = f₂ × c₁ + i₂ × c̃₂ = 0.153194 + 0.247285 = 0.400479
  Memori berubah dari c₁ = 0.193228 menjadi c₂ = 0.400479.
```

## 7. Gambar 6: Output Gate

**Alurnya:**

1. hₜ₋₁ dan xₜ masuk ke **kotak σ keempat** dan menghasilkan **oₜ** (0 sampai 1,
   persamaan 19).
2. Sementara itu, **cₜ** dari garis atas bercabang turun ke **oval tanh** (tanpa bobot)
   yang memampatkan cₜ ke rentang -1 sampai 1.
3. tanh(cₜ) dan oₜ bertemu di lingkaran **×** sehingga menjadi **hₜ = oₜ ⊙ tanh(cₜ)**
   (persamaan 20).
4. hₜ **bercabang dua**: naik ke atas sebagai keluaran waktu t, dan ke kanan sebagai
   hₜ₋₁ untuk langkah berikutnya.

**Fungsinya:** menentukan **bagian memori mana yang ditampilkan** sebagai keluaran
saat ini. Tidak semua isi cₜ relevan untuk keluaran sekarang; sebagian cukup
disimpan untuk dipakai nanti.

```
Contoh angka (t = 2):
  o₂ = σ(W_o × x₂ + U_o × h₁ + b_o) = σ(0.4 × 0.6 + 0.5 × 0.104941 + 0) = σ(0.292470) = 0.572601
  tanh(c₂) = tanh(0.400479) = 0.380359        (oval tanh, tanpa bobot)
  h₂ = o₂ × tanh(c₂) = 0.572601 × 0.380359 = 0.217794
  t = 2 adalah langkah terakhir, jadi h₂ masuk ke lapisan dense (persamaan 21):
  ŷ' = W_y × h₂ + b_y = 0.8 × 0.217794 + 0 = 0.174235
```

## 8. Ringkasan Satu Langkah LSTM

Seluruh alur satu time step LSTM (contoh t = 2, memori masuk c₁ = 0.193228 dan h₁ = 0.104941):

| Urutan | Gambar | Perhitungan | Hasil | Makna |
|---|---|---|---|---|
| 1. Forget | Gambar 3 | f₂ = σ(a); f₂ × c₁ | 0.792815; 0.153194 | 79.3% memori lama dipertahankan |
| 2. Input | Gambar 4 | i₂ = σ(a); c̃₂ = tanh(a); i₂ × c̃₂ | 0.596639; 0.414463; 0.247285 | usulan baru masuk 59.7% |
| 3. Update | Gambar 5 | c₂ = f₂c₁ + i₂c̃₂ | 0.400479 | memori jangka panjang baru |
| 4. Output | Gambar 6 | o₂ = σ(a); h₂ = o₂ tanh(c₂) | 0.572601; 0.217794 | 57.3% memori yang sudah dimampatkan dikeluarkan |
| 5. Dense | - | ŷ′ = W_y h₂ + b_y | 0.174235 | prediksi (skala ternormalisasi) |

Prediksi ŷ′ = 0.174235 masih jauh dari target 0.7. Seberapa salah
prediksi ini dan bagaimana bobot diperbaiki dijelaskan di [Bagian 2](tahap_7_8_2_fungsi_loss_dan_adam.md) (konsep loss
dan Adam) dan [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md) (perhitungan lengkapnya).

## 9. Gambar 7: Struktur Sel GRU

GRU **hanya punya satu garis memori**, yaitu hₜ (garis atas). Tidak ada cₜ
terpisah dan tidak ada output gate. Gambar 7 dibaca dalam enam langkah berikut;
contoh angkanya memakai time step t = 2 dari simulasi GRU mini.

**Langkah 0 — masukan.**

- hₜ₋₁ masuk dari **kiri atas**, lalu bercabang: tetap di garis atas, turun lewat
  garis vertikal kiri ke jalur bawah, dan bercabang ke tengah menuju lingkaran ×
  milik rₜ.
- xₜ masuk dari **bawah** ke jalur horizontal bawah.
- Jalur bawah membawa hₜ₋₁ dan xₜ ke ketiga kotak kuning: σ (z), σ (r), dan tanh.
- Contoh: h₁ = 0.147273 (hasil t = 1) dan x₂ = 0.6.

**Langkah 1 — update gate (kotak σ kiri) menghasilkan zₜ** (persamaan 23).

- z₂ = σ(W_z × x₂ + U_z × h₁ + b_z(in) + b_z(rec)) = σ(0.358909) = **0.588776**.
- Panah zₜ naik dan **bercabang dua**: ke **lingkaran × kiri atas**, mengalikan
  hₜ₋₁ menjadi z₂ × h₁ = 0.588776 × 0.147273 = **0.086711**;
  dan ke kanan menuju **lingkaran "1−"**, menghasilkan 1 − z₂ = **0.411224**.

**Langkah 2 — reset gate (kotak σ tengah) menghasilkan rₜ** (persamaan 24).

- r₂ = σ(W_r × x₂ + U_r × h₁ + b_r(in) + b_r(rec)) = σ(0.404182) = **0.599692**.
- Panah rₜ naik ke **lingkaran × tengah**, tempat ia mengalikan memori lama yang datang
  dari kiri.

**Langkah 3 — kandidat (kotak tanh) menghasilkan h̃ₜ** (persamaan 25).

- Cabang memori lama melewati lingkaran × milik rₜ sebelum masuk tanh. Pada Keras
  (`reset_after=True`), yang dikalikan rₜ adalah bagian rekuren
  qₜ = hₜ₋₁U_h + b_h(rec): q₂ = 0.2 × 0.147273 + 0 = 0.029455,
  sehingga r₂ × q₂ = 0.017664.
- xₜ masuk dari bawah: x₂W_h + b_h(in) = 0.7 × 0.6 + 0 = 0.420000.
- h̃₂ = tanh(0.420000 + 0.017664) = tanh(0.437664) = **0.411706**.
- Fungsi rₜ: menentukan **seberapa banyak masa lalu dipakai untuk menyusun usulan
  baru**. rₜ mendekati 0 berarti kandidat disusun hampir hanya dari xₜ; rₜ mendekati 1
  berarti masa lalu ikut diperhitungkan penuh. Pada t = 1, q₁ = 0 karena h₀ = 0, jadi
  reset gate belum berpengaruh.

**Langkah 4 — pencampuran** (persamaan 26).

- h̃ₜ naik ke **lingkaran × kanan** dan dikalikan (1 − zₜ):
  0.411224 × 0.411706 = 0.169303.
- Hasilnya naik ke **lingkaran +** dan dijumlahkan dengan zₜ × hₜ₋₁ dari kiri:
  h₂ = 0.086711 + 0.169303 = **0.256014**.

**Langkah 5 — keluaran.** h₂ keluar ke **kanan** (ke langkah berikutnya) dan ke
**atas** (keluaran waktu t). Karena t = 2 adalah langkah terakhir, h₂ masuk ke dense:
ŷ′ = 0.8 × 0.256014 + 0 = **0.204811**.

**Fungsi utama zₜ:** zₜ bekerja seperti **penggeser (slider) pencampur**. zₜ mendekati 1
berarti hₜ hampir sama dengan hₜ₋₁ (memori lama dipertahankan); zₜ mendekati 0 berarti hₜ
hampir sama dengan h̃ₜ (diganti informasi baru). Porsi lama dan porsi baru **selalu
berjumlah 1**. Pada contoh di atas: 58.9% memori lama + 41.1%
kandidat baru.

| Langkah | Bagian Gambar 7 | Hasil (t = 2) |
|---|---|---|
| 1. Update gate | kotak σ kiri | z₂ = 0.588776 |
| 2. Reset gate | kotak σ tengah | r₂ = 0.599692 |
| 3. Kandidat | lingkaran × tengah, kotak tanh | r₂q₂ = 0.017664; h̃₂ = 0.411706 |
| 4. Pencampuran | lingkaran × kiri, 1−, × kanan, + | 0.086711 + 0.169303 = 0.256014 |
| 5. Dense | - | ŷ′ = 0.204811 |

## 10. Hubungan LSTM dan GRU

| LSTM | GRU | Penjelasan |
|---|---|---|
| fₜ (forget) dan iₜ (input), **independen** | zₜ dan (1 − zₜ), **terikat** | GRU menggabungkan keduanya menjadi satu update gate: menyimpan lebih banyak yang lama berarti menerima lebih sedikit yang baru |
| cₜ dan hₜ (dua memori) | hanya hₜ | memori langsung disimpan di hidden state |
| oₜ (output gate) | tidak ada | seluruh hₜ langsung dikeluarkan |
| tidak ada padanan langsung | rₜ (reset gate) | mengatur peran masa lalu saat menyusun kandidat |
| jalur cell state (Gambar 2) | jalur langsung zₜ ⊙ hₜ₋₁ | "jalan tol gradien"; pada simulasi, 86.4% (LSTM) dan 97.3% (GRU) sinyal kesalahan mengalir lewat jalur ini |
| 4 himpunan bobot (i, f, c, o) | 3 himpunan bobot (z, r, h) | parameter GRU lebih sedikit |

Jumlah parameter lapisan rekuren dengan 10 fitur (persamaan 22 dan 27, tanpa dense):

| Neuron n_u | LSTM: 4[n_u(n_u + 10) + n_u] | GRU: 3[n_u(n_u + 10) + 2n_u] |
|---|---|---|
| 30 | 4,920 | 3,780 ← GRU terbaik |
| 40 | 8,160 ← LSTM terbaik | 6,240 |

Pada jumlah neuron yang sama, GRU selalu memerlukan parameter lebih sedikit.

## 11. Catatan untuk Naskah Skripsi

1. **Klaim tentang rumus asli Cho et al. (2014) di halaman 17 perlu diperbaiki.**
   Persamaan (7) pada makalah aslinya berbunyi
   hⱼ⟨t⟩ = zⱼ hⱼ⟨t−1⟩ + (1 − zⱼ) h̃ⱼ⟨t⟩, yaitu **sama** dengan persamaan (26) (konvensi
   Keras), sehingga peran zₜ **tidak tertukar**. Bentuk dengan peran zₜ tertukar,
   hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, berasal dari Chung et al. (2014). Perbedaan dengan
   Cho et al. (2014) hanya pada posisi reset gate. Saran kalimat:

   > Rumus GRU pada makalah asli Cho et al. (2014) sedikit berbeda, yaitu gerbang
   > reset diterapkan sebelum perkalian dengan matriks bobot rekuren,
   > h̃ₜ = tanh(xₜW_h + (rₜ ⊙ hₜ₋₁)U_h). Adapun bentuk
   > hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, dengan peran zₜ tertukar, digunakan oleh
   > Chung et al. (2014).

2. **Gambar 7 dan pengaturan `reset_after=True`.** Gambar 7 menggambar rₜ
   mengalikan hₜ₋₁ sebelum kotak tanh (bentuk Cho et al.), sedangkan persamaan (25)
   mengalikan rₜ dengan (hₜ₋₁U_h + b_h(rec)) setelah perkalian matriks. Fungsinya
   sama, jadi gambar tidak perlu diubah. Cukup tambahkan kalimat berikut, atau beri
   label garis dari lingkaran × rₜ ke kotak tanh dengan rₜ ⊙ (hₜ₋₁U_h + b_h(rec)) dan
   garis xₜ ke kotak tanh dengan xₜW_h + b_h(in):

   > Gambar 7 merupakan ilustrasi konseptual; urutan perhitungan yang digunakan
   > mengikuti persamaan (25).

3. **Notasi Cₜ pada Gambar 1-6.** Gambar Olah (2015) memakai huruf besar Cₜ untuk
   cell state, sedangkan persamaan skripsi memakai cₜ. Satu kalimat penjelas dapat
   mencegah pertanyaan penguji.

---

→ Berikutnya: [Bagian 2. Fungsi Loss dan Adam](tahap_7_8_2_fungsi_loss_dan_adam.md)
