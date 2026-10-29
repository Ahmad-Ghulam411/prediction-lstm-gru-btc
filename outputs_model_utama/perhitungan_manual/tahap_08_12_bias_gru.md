# Perhitungan Manual - Tahap 8.12: Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias

Subbagian ini merangkum pembahasan **bias pada GRU**: kenapa bias perlu ada dan bagaimana
nilainya diperoleh, termasuk kenapa GRU punya dua jenis bias. Semua contoh
angka diambil dari model GRU utama (30 neuron): bobot awal dan satu siklus
pelatihannya (Tahap 8.6 sampai Tahap 8.9) serta bobot akhirnya.

**Daftar isi**

- [1. Intinya](#1-intinya)
- [2. Mengapa Bias Diperlukan](#2-mengapa-bias-diperlukan)
- [3. Cara Menghitung Bias Langkah demi Langkah](#3-cara-menghitung-bias-langkah-demi-langkah)
- [4. Dua Bias pada GRU: Asal dan Buktinya](#4-dua-bias-pada-gru-asal-dan-buktinya)
- [5. Catatan untuk Naskah Skripsi](#5-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Intinya

Bias itu **titik awal (intercept)** setiap gerbang dan neuron, mirip intercept *a* pada
regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa bias, setiap kali
masukannya nol, gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu setengah terbuka;
tanh(0) = 0). Dengan bias, setiap gerbang bisa punya **posisi bawaannya sendiri**. Nilainya
tidak dihitung dengan satu rumus, tetapi **dipelajari** lewat siklus yang sama dengan bobot
(bagian 3).

## 2. Mengapa Bias Diperlukan

**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Kondisi ini sering
terjadi: h₀ = 0 di awal setiap jendela, dan normalisasi min-max membuat fitur bernilai dekat
0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman, sedangkan bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:

| x | Dengan bias: σ(5x − 2.5) | Tanpa bias: σ(5x) |
|---|---|---|
| 0 | 0.076 | 0.500 |
| 0.25 | 0.223 | 0.777 |
| 0.5 | 0.500 | 0.924 |
| 0.75 | 0.777 | 0.977 |
| 1 | 0.924 | 0.993 |

Tanpa bias, gerbang **tidak pernah di bawah 0.5** untuk data ternormalisasi yang positif,
sehingga model tidak bisa membuat aturan "tutup gerbang saat fitur rendah, buka saat tinggi".

**(c) Bias menentukan sikap bawaan setiap gerbang.** Pada GRU, bias update gate menentukan
campuran bawaan memori lama dan baru, sedangkan bias reset gate menentukan seberapa banyak
masa lalu dipakai untuk kandidat. Semua bias GRU mulai dari 0 (sikap netral, σ = 0.5), lalu
bergeser selama pelatihan. Pada **model GRU terlatih** (neuron ke-1, saat xW + hU = 0;
nilai bias yang sama tampil di Tahap 8.11):

| Gerbang | Bias masukan + bias rekuren | Nilai bawaan | Artinya |
|---|---|---|---|
| update z | 0.211800 + 0.211800 = 0.423600 | σ = 0.604 | cenderung mempertahankan 60% memori lama |
| reset r | -0.249689 + (-0.249689) = -0.499377 | σ = 0.378 | memakai sekitar 38% masa lalu untuk kandidat |
| kandidat h̃ | b_h(in) = 0.000974, b_h(rec) = 0.007737 | - | b_h(rec) ikut dikalikan rₜ (bagian 4) |

Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari mengikuti
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menentukan tingkat dasar prediksi.** h_T selalu di rentang (−1, 1), jadi b_y
yang mengatur "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada model GRU
terlatih, b_y = 0.042936, setara 0.042936 × 98,196.75 ≈
**4,216.21 USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih (25,162.70 USD) setiap kali h_T = 0.

**(e) Bias selalu mendapat sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai nol
kalau masukannya nol (di Tahap 8.8, U tidak mendapat sumbangan dari t = 1 karena
h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa-apa.

**(f) Biayanya kecil.** Model GRU terlatih memiliki 181 bias (termasuk b_y) dari 3,811 parameter, atau 4.7%.

## 3. Cara Menghitung Bias Langkah demi Langkah

Bias tidak punya rumus langsung; bias **dipelajari** lewat siklus yang sama dengan bobot.
Berikut lima langkahnya pada model GRU utama, memakai angka yang sudah dihitung di
subbagian sebelumnya. Contohnya **b_y** dan **b_h(in)[1]** (bias neuron ke-1).

```
Langkah 1 — Nilai awal (Tahap 8.6)
  semua bias gerbang (masukan dan rekuren) = 0, dan b_y = 0.

Langkah 2 — Dipakai di forward pass (Tahap 8.6)
  Gerbang : a = x W + h U + b
  Dense   : ŷ′ = h_T W_y + b_y

Langkah 3 — Gradien (Tahap 8.8)
  Karena ∂a/∂b = 1, gradien bias = jumlah δ untuk semua sampel dan semua hari:
  ∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = -0.224177
  ∂L/∂b_h(in)[1] = Σₜ Σₙ δa_h  (neuron ke-1), per hari:
    t=1: 1.5872e-05, t=2: -0.000239, t=3: -0.000982, t=4: -0.002869, t=5: -0.007216, t=6: -0.016288, t=7: -0.032967
    jumlah = -0.060545

Langkah 4 — Update Adam (Tahap 8.9)
  b_y : 0.000000 → 0.001000 (k = 1) → 0.001861 (k = 2)
  b_h(in)[1]: 0.000000 → 0.001000 (k = 1) → 0.001859 (k = 2)

Langkah 5 — Ulangi 13,000 kali (26 batch × 500 epoch)
  b_y akhir = 0.042936,  b_h(in)[1] akhir = 0.000974
```

## 4. Dua Bias pada GRU: Asal dan Buktinya

**Letaknya pada Gambar 7.** Bias tidak digambar; bias ada di dalam setiap kotak kuning (σ, σ,
tanh). Setiap kotak menerima dua garis masuk, yaitu xₜ dari bawah dan hₜ₋₁ dari kiri. Dengan
`reset_after=True`, Keras menghitung kedua garis itu terpisah, masing-masing dengan biasnya
sendiri: garis xₜ membawa **bias masukan** (xₜW + b(in)) dan garis hₜ₋₁ membawa **bias
rekuren** (hₜ₋₁U + b(rec)).

```
Kotak σ untuk z (dan r, sama persis): kedua cabang langsung dijumlahkan
  h_(t-1) ──► h_(t-1) × U_z + b_z(rec) ──┐
                                         (+) ──► σ ──► z_t
  x_t ─────► x_t × W_z + b_z(in) ────────┘

Kotak tanh untuk kandidat: cabang h_(t-1) dikali r_t dulu
  h_(t-1) ──► h_(t-1) × U_h + b_h(rec) ──► (× r_t) ──┐
                                                     (+) ──► tanh ──► h̃_t
  x_t ─────► x_t × W_h + b_h(in) ─────────────────────┘
```

**Pada z dan r, dua bias sebenarnya berlebih.** b(in) dan b(rec) langsung dijumlahkan, jadi
gradien keduanya selalu sama. Di iterasi pertama model ini, ∂L/∂b_z(in)[1] = ∂L/∂b_z(rec)[1] =
-0.000220. Karena nilai awalnya juga sama (0) dan gradiennya selalu kembar, keduanya
tetap kembar sampai akhir pelatihan (selisihnya hanya sisa pembulatan float32): selisih
terbesar antara b_z(in) dan b_z(rec) di model terlatih = 1.8e-07, dan antara b_r(in) dan
b_r(rec) = 1.1e-07.

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikali rₜ (kalau rₜ mendekati 0, bias ini
ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu aktif. Gradiennya pun berbeda:
iterasi pertama memberi ∂L/∂b_h(in)[1] = -0.060545 dan ∂L/∂b_h(rec)[1] =
-0.024833. Di model terlatih, neuron ke-1 punya b_h(in) = 0.000974 dan
b_h(rec) = 0.007737 (selisih terbesar di antara semua neuron 0.053780).

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak menuliskan bias sama sekali
(*"to make the equations uncluttered, we omit biases"*). Bentuk dua bias berasal dari
implementasi GRU pada pustaka **NVIDIA cuDNN** yang memisahkan bagian masukan dan rekuren
setiap gerbang, dan Keras memakainya sebagai bawaan (`reset_after=True`). LSTM cukup satu
bias karena semua biasnya hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU 30 neuron memiliki
2 × 3 × 30 = 180 bias gerbang. Dengan satu bias jumlahnya hanya
90, sehingga total parameter menjadi 3,721,
bukan 3,811.

## 5. Catatan untuk Naskah Skripsi

**Penjelasan dua bias GRU** (setelah Gambar 7 atau persamaan 26):

> Pada Gambar 7, bias tidak digambarkan secara eksplisit karena termuat di dalam setiap
> lapisan (kotak σ dan tanh). Pada implementasi Keras dengan pengaturan reset_after=True,
> setiap lapisan memisahkan kontribusi masukan xₜW + b⁽ⁱⁿ⁾ dan kontribusi rekuren
> hₜ₋₁U + b⁽ʳᵉᶜ⁾, masing-masing dengan biasnya sendiri. Pada update gate dan reset gate
> kedua bias hanya dijumlahkan, sedangkan pada kandidat hidden state bias rekuren
> b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga keduanya tidak dapat digabung.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.11: Forward Pass GRU (Jendela Pertama Data Uji)](tahap_08_11_forward_pass_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 9.2: Perbandingan Struktur LSTM dan GRU](tahap_09_02_perbandingan_struktur_lstm_gru.md)
