# Perhitungan Manual - Tahap 7.9: Pelatihan LSTM Model Utama, Langkah 4: Update Adam

Langkah terakhir siklus pelatihan: memakai gradien dari [Tahap 7.8 (Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time)](tahap_07_08_pelatihan_lstm_bptt.md) untuk
memperbaiki **seluruh 8,201 parameter** model LSTM utama dengan optimizer Adam,
lalu memeriksa bahwa hasilnya sama dengan yang dilakukan `model.fit()`.

**Daftar isi**

- [1. Rumus Adam](#1-rumus-adam)
- [2. Iterasi k = 1, Satu Parameter (b_y)](#2-iterasi-k--1-satu-parameter-b_y)
- [3. Iterasi k = 1, Semua Parameter](#3-iterasi-k--1-semua-parameter)
- [4. Apakah Loss Langsung Turun?](#4-apakah-loss-langsung-turun)
- [5. Iterasi k = 2: Momentum Mulai Bekerja](#5-iterasi-k--2-momentum-mulai-bekerja)
- [6. Dicocokkan dengan model.fit Keras](#6-dicocokkan-dengan-modelfit-keras)
- [7. Siklus Diulang Sampai Selesai](#7-siklus-diulang-sampai-selesai)
- [8. Catatan untuk Naskah Skripsi](#8-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Rumus Adam

Adam bekerja **per parameter**: setiap bobot dan bias punya catatan m dan v sendiri, yang
mulai dari nol. Pada iterasi ke-k dengan gradien gₖ:

- **Momen pertama** (persamaan 29): mₖ = β₁mₖ₋₁ + (1 − β₁)gₖ, rata-rata bergerak arah gradien
  (β₁ = 0.9, kira-kira merangkum 10 gradien terakhir). Fungsinya seperti momentum, supaya
  langkah tidak gampang terombang-ambing oleh satu batch.
- **Momen kedua** (persamaan 29): vₖ = β₂vₖ₋₁ + (1 − β₂)gₖ², rata-rata bergerak besarnya
  gradien (β₂ = 0.999). Dipakai sebagai pembagi supaya ukuran langkah tiap parameter seimbang.
- **Koreksi bias** (persamaan 30): m̂ₖ = mₖ/(1 − β₁ᵏ), v̂ₖ = vₖ/(1 − β₂ᵏ), karena m dan v mulai
  dari nol sehingga di awal nilainya terlalu kecil.
- **Update** (persamaan 31): θₖ = θₖ₋₁ − η m̂ₖ/(√v̂ₖ + ε), dengan η = 0.001 dan ε = 10⁻⁷.

Keras menulis rumus yang sama dalam bentuk ringkas: α = η√(1 − β₂ᵏ)/(1 − β₁ᵏ), lalu
θ ← θ − α m/(√v + ε). Bedanya hanya letak ε, dan pengaruhnya sangat kecil. Perhitungan di
bawah mengikuti bentuk Keras agar hasilnya bisa dicocokkan sampai digit terakhir.

## 2. Iterasi k = 1, Satu Parameter (b_y)

```
θ₀ = b_y awal = 0.000000      g₁ = ∂L/∂b_y = -0.007474   (Tahap 7.8)

m₁ = 0.9 × 0 + 0.1 × (-0.007474) = -0.000747
v₁ = 0.999 × 0 + 0.001 × (-0.007474)² = 5.5864e-08
α₁ = 0.001 × √(1 − 0.999) / (1 − 0.9) = 0.001 × 0.031623 / 0.1 = 0.00031623
Δ  = α₁ × m₁ / (√v₁ + ε) = 0.00031623 × (-0.000747) / (0.000236 + 1e-7) = -0.001000
θ₁ = θ₀ − Δ = 0.000000 − (-0.001000) = 0.001000
```

Gradien b_y negatif, artinya prediksi rata-rata terlalu
rendah, jadi b_y dinaikkan. Besar langkahnya
hampir tepat η = 0.001. Ini bukan kebetulan. Kalau rumus di atas disederhanakan untuk k = 1
(m₁ = 0.1g, √v₁ = 0.0316|g|), hasilnya:

$$\Delta_1 = \eta\,\frac{g}{|g| + 3.16 \times 10^{-6}}$$

Angka 3.16 × 10⁻⁶ = ε/√(1 − β₂) adalah "ambang" kecil. Kalau |g| jauh lebih besar dari ambang
ini, langkahnya ≈ η × tanda(g), berapa pun besar gradiennya. Kalau |g| sekecil ambang atau
lebih kecil, langkahnya ikut mengecil.

## 3. Iterasi k = 1, Semua Parameter

| Parameter | θ₀ | g₁ | m₁ | v₁ | Δ | θ₁ = θ₀ − Δ |
|---|---|---|---|---|---|---|
| b_y[1] | 0.000000 | -0.007474 | -0.000747 | 5.5864e-08 | (-0.001000) | 0.001000 |
| W_y[1, 1] | 0.241729 | 9.7972e-05 | 9.7972e-06 | 9.5986e-12 | 0.000969 | 0.240760 |
| b_f[1] | 1.000000 | -5.0019e-06 | -5.0019e-07 | 2.5019e-14 | (-0.000613) | 1.000613 |
| W_f[1, 1] | -0.171849 | -2.3416e-07 | -2.3416e-08 | 5.4832e-17 | (-6.8944e-05) | -0.171780 |
| U_f[1, 1] | 0.063310 | -2.0085e-07 | -2.0085e-08 | 4.0340e-17 | (-5.9721e-05) | 0.063369 |
| b_i[1] | 0.000000 | 1.1947e-05 | 1.1947e-06 | 1.4273e-13 | 0.000791 | -0.000791 |

Sebaran besar langkah untuk seluruh 8,201 parameter pada iterasi pertama:

| Besar langkah \|Δ\| | Jumlah parameter | Porsi |
|---|---|---|
| ≥ 0.99η (hampir tepat η) | 233 | 2.8% |
| 0.5η sampai 0.99η | 4,119 | 50.2% |
| 0.1η sampai 0.5η | 2,580 | 31.5% |
| < 0.1η | 1,269 | 15.5% |

Jadi 53.1% parameter bergeser lebih dari separuh η, padahal gradiennya bisa berbeda
ribuan kali lipat (lihat tabel gradien di Tahap 7.8). Di iterasi pertama, Adam belum
"tahu" parameter mana yang penting; ia hanya melihat tanda gradien.

## 4. Apakah Loss Langsung Turun?

Hitung ulang forward pass batch pertama dengan bobot θ₁:

Loss batch pertama justru **naik**, dari 0.000198 menjadi 0.001783. Ini wajar di awal pelatihan dengan Adam. Seperti terlihat di bagian 3, pada iterasi pertama banyak parameter digeser hampir sama jauh, padahal sebagian besar gradiennya sangat kecil. Untuk parameter seperti itu langkahnya kebesaran, dan kalau ribuan parameter bergeser bersamaan, efek gabungannya bisa melewati titik terbaik (overshoot). Begitu m dan v sudah merekam beberapa gradien, ukuran langkah tiap parameter menyesuaikan diri, dan loss turun dengan jelas dalam beberapa epoch (lihat Tahap 7.10).

## 5. Iterasi k = 2: Momentum Mulai Bekerja

Iterasi kedua memakai **batch kedua** (target 2023-08-09 s.d. 2023-09-09) dan
bobot θ₁. Forward pass dan BPTT diulang dengan cara yang sama: loss batch kedua =
0.004550. Untuk b_y:

```
g₂ = 0.130597
m₂ = 0.9 × (-0.000747) + 0.1 × 0.130597 = 0.012387
v₂ = 0.999 × 5.5864e-08 + 0.001 × 0.130597² = 1.7111e-05
α₂ = 0.001 × √(1 − 0.999²) / (1 − 0.9²) = 0.00023532
Δ  = α₂ × m₂ / (√v₂ + ε) = 0.000705
θ₂ = 0.001000 − 0.000705 = 0.000295
```

Gradien kedua berlawanan arah dengan yang pertama. Momentum m₂ meredam perubahan arah ini, sehingga b_y tidak langsung berbalik arah sepenuhnya.
Mulai iterasi ini ukuran langkah tidak lagi seragam: setiap parameter punya m dan v sendiri.

## 6. Dicocokkan dengan model.fit Keras

Sebagai pembuktian, notebook membangun model LSTM yang sama (seed 42), menjalankan
`model.fit()` dengan pengaturan penelitian (batch 32, `shuffle=False`), lalu mengambil
bobot Keras tepat setelah batch ke-1 dan ke-2. Hasilnya dibandingkan dengan perhitungan
manual di atas untuk seluruh 8,201 parameter:

| Setelah | Selisih terbesar manual vs Keras |
|---|---|
| iterasi k = 1 (batch 1) | 5.7e-08 |
| iterasi k = 2 (batch 2) | 1.3e-07 |

Selisihnya hanya sisa pembulatan (Keras menghitung dengan float32), jadi siklus forward → loss → BPTT → Adam yang dihitung manual memang siklus yang dijalankan Keras.

## 7. Siklus Diulang Sampai Selesai

Siklus yang sama diulang untuk batch ke-3, ke-4, dan seterusnya sampai batch ke-26
(satu epoch), lalu diulang lagi selama 500 epoch. Totalnya **13,000 kali update**.
Perjalanan beberapa parameter:

| Parameter | Awal | k = 1 | k = 2 | Akhir (k = 13,000) |
|---|---|---|---|---|
| b_y[1] | 0.000000 | 0.001000 | 0.000295 | 0.081419 |
| W_y[1, 1] | 0.241729 | 0.240760 | 0.241425 | 0.164646 |
| b_f[1] | 1.000000 | 1.000613 | 1.001371 | 0.681141 |

Bobot akhir inilah bobot model LSTM terbaik yang dipakai untuk prediksi (bentuknya di
Tahap 7.5 dan forward pass-nya diverifikasi di Tahap 7.11).

## 8. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11, setelah persamaan 31):

> Pada iterasi pertama, Adam menggeser setiap parameter ke arah yang berlawanan dengan
> tanda gradiennya dengan besar langkah mendekati learning rate η = 0.001, kecuali untuk
> parameter yang gradiennya sangat kecil. Pada iterasi berikutnya,
> momen pertama dan kedua yang terakumulasi membuat ukuran langkah setiap parameter
> menyesuaikan diri dengan riwayat gradiennya masing-masing. Model LSTM terbaik
> diperbarui sebanyak 26 × 500 = 13,000 kali.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.8: Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time](tahap_07_08_pelatihan_lstm_bptt.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.10: Membaca Kurva Loss Model LSTM](tahap_07_10_kurva_loss_lstm.md)
