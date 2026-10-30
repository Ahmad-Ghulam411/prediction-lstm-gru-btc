# Perhitungan Manual - Tahap 8.9: Pelatihan GRU Model Utama, Langkah 4: Update Adam

Langkah terakhir siklus pelatihan: memakai gradien dari [Tahap 8.8 (Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time)](tahap_08_08_pelatihan_gru_bptt.md) untuk
memperbaiki **seluruh 3,811 parameter** model GRU utama dengan optimizer Adam,
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
θ₀ = b_y awal = 0.000000      g₁ = ∂L/∂b_y = -0.224177   (Tahap 8.8)

m₁ = 0.9 × 0 + 0.1 × (-0.224177) = -0.022418
v₁ = 0.999 × 0 + 0.001 × (-0.224177)² = 5.0255e-05
α₁ = 0.001 × √(1 − 0.999) / (1 − 0.9) = 0.001 × 0.031623 / 0.1 = 0.00031623
Δ  = α₁ × m₁ / (√v₁ + ε) = 0.00031623 × (-0.022418) / (0.007089 + 1e-7) = -0.001000
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
| b_y[1] | 0.000000 | -0.224177 | -0.022418 | 5.0255e-05 | (-0.001000) | 0.001000 |
| W_y[1, 1] | 0.277997 | 0.044590 | 0.004459 | 1.9882e-06 | 0.001000 | 0.276997 |
| b_z(in)[1] | 0.000000 | -0.000220 | -2.1970e-05 | 4.8270e-11 | (-0.000986) | 0.000986 |
| W_z[1, 1] | 0.175476 | -1.0759e-05 | -1.0759e-06 | 1.1576e-13 | (-0.000773) | 0.176249 |
| U_z[1, 1] | 0.002427 | 2.6822e-05 | 2.6822e-06 | 7.1941e-13 | 0.000895 | 0.001532 |
| b_h(rec)[1] | 0.000000 | -0.024833 | -0.002483 | 6.1669e-07 | (-0.001000) | 0.001000 |

Sebaran besar langkah untuk seluruh 3,811 parameter pada iterasi pertama:

| Besar langkah \|Δ\| | Jumlah parameter | Porsi |
|---|---|---|
| ≥ 0.99η (hampir tepat η) | 1,478 | 38.8% |
| 0.5η sampai 0.99η | 2,146 | 56.3% |
| 0.1η sampai 0.5η | 170 | 4.5% |
| < 0.1η | 17 | 0.4% |

Jadi 95.1% parameter bergeser lebih dari separuh η, padahal gradiennya bisa berbeda
ribuan kali lipat (lihat tabel gradien di Tahap 8.8). Di iterasi pertama, Adam belum
"tahu" parameter mana yang penting; ia hanya melihat tanda gradien.

## 4. Apakah Loss Langsung Turun?

Hitung ulang forward pass batch pertama dengan bobot θ₁:

Loss batch pertama turun dari 0.013489 menjadi 0.004921 (63.5% lebih kecil). Satu langkah kecil pada semua parameter sudah cukup memperbaiki prediksi batch ini.

## 5. Iterasi k = 2: Momentum Mulai Bekerja

Iterasi kedua memakai **batch kedua** (target 2023-08-09 s.d. 2023-09-09) dan
bobot θ₁. Forward pass dan BPTT diulang dengan cara yang sama: loss batch kedua =
0.002340. Untuk b_y:

```
g₂ = -0.069853
m₂ = 0.9 × (-0.022418) + 0.1 × (-0.069853) = -0.027161
v₂ = 0.999 × 5.0255e-05 + 0.001 × (-0.069853)² = 5.5084e-05
α₂ = 0.001 × √(1 − 0.999²) / (1 − 0.9²) = 0.00023532
Δ  = α₂ × m₂ / (√v₂ + ε) = -0.000861
θ₂ = 0.001000 − (-0.000861) = 0.001861
```

Gradien kedua searah dengan yang pertama, jadi momentum m₂ makin kuat dan b_y terus bergerak ke arah yang sama.
Mulai iterasi ini ukuran langkah tidak lagi seragam: setiap parameter punya m dan v sendiri.

## 6. Dicocokkan dengan model.fit Keras

Sebagai pembuktian, notebook membangun model GRU yang sama (seed 42), menjalankan
`model.fit()` dengan pengaturan penelitian (batch 32, `shuffle=False`), lalu mengambil
bobot Keras tepat setelah batch ke-1 dan ke-2. Hasilnya dibandingkan dengan perhitungan
manual di atas untuk seluruh 3,811 parameter:

| Setelah | Selisih terbesar manual vs Keras |
|---|---|
| iterasi k = 1 (batch 1) | 2.0e-08 |
| iterasi k = 2 (batch 2) | 4.2e-08 |

Selisihnya hanya sisa pembulatan (Keras menghitung dengan float32), jadi siklus forward → loss → BPTT → Adam yang dihitung manual memang siklus yang dijalankan Keras.

## 7. Siklus Diulang Sampai Selesai

Siklus yang sama diulang untuk batch ke-3, ke-4, dan seterusnya sampai batch ke-26
(satu epoch), lalu diulang lagi selama 500 epoch. Totalnya **13,000 kali update**.
Perjalanan beberapa parameter:

| Parameter | Awal | k = 1 | k = 2 | Akhir (k = 13,000) |
|---|---|---|---|---|
| b_y[1] | 0.000000 | 0.001000 | 0.001861 | 0.042936 |
| W_y[1, 1] | 0.277997 | 0.276997 | 0.276137 | 0.256136 |
| b_z(in)[1] | 0.000000 | 0.000986 | 0.001845 | 0.211800 |

Bobot akhir inilah bobot model GRU terbaik yang dipakai untuk prediksi (bentuknya di
Tahap 8.5 dan forward pass-nya diverifikasi di Tahap 8.11).

## 8. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11, setelah persamaan 31):

> Pada iterasi pertama, Adam menggeser setiap parameter ke arah yang berlawanan dengan
> tanda gradiennya dengan besar langkah mendekati learning rate η = 0.001, kecuali untuk
> parameter yang gradiennya sangat kecil. Pada iterasi berikutnya,
> momen pertama dan kedua yang terakumulasi membuat ukuran langkah setiap parameter
> menyesuaikan diri dengan riwayat gradiennya masing-masing. Model GRU terbaik
> diperbarui sebanyak 26 × 500 = 13,000 kali.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.8: Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time](tahap_08_08_pelatihan_gru_bptt.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.10: Membaca Kurva Loss Model GRU](tahap_08_10_kurva_loss_gru.md)
