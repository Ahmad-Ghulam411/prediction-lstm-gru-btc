# Perhitungan Manual - Tahap 7.2: Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss

Subbagian ini memulai **simulasi satu siklus pelatihan LSTM** pada model mini. Di sini
dihitung dua langkah pertama: **forward pass** (berapa prediksinya) dan **loss**
(seberapa salah prediksinya). Langkah 3 dan 4 dilanjutkan di [Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md) dan
[Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md).

**Daftar isi**

- [1. Gambaran Satu Siklus Pelatihan](#1-gambaran-satu-siklus-pelatihan)
- [2. Kapan Loss dan Adam Dipakai dalam Penelitian](#2-kapan-loss-dan-adam-dipakai-dalam-penelitian)
- [3. Model Mini dan Notasi](#3-model-mini-dan-notasi)
- [4. Langkah 1: Forward Pass](#4-langkah-1-forward-pass)
- [5. Langkah 2: Loss MSE](#5-langkah-2-loss-mse)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Gambaran Satu Siklus Pelatihan

Pelatihan model bisa dibayangkan seperti **orang yang menuruni gunung dalam kabut**
untuk mencari titik terendah:

| Komponen | Analogi | Tugasnya |
|---|---|---|
| **Fungsi loss (MSE)** | ketinggian posisi saat ini | mengukur **seberapa salah** prediksi; makin kecil makin baik |
| **Gradien gₖ** | kemiringan tanah di bawah kaki | menunjukkan **arah** perubahan bobot yang menaikkan loss; dihitung dengan *backpropagation through time* (BPTT) |
| **Adam** | strategi melangkah | memutuskan **seberapa jauh dan ke mana** setiap bobot digeser agar loss turun |

Urutannya selalu: **loss dihitung → gradien dihitung dari loss → Adam memakai gradien
untuk memperbarui bobot**. Satu putaran disebut **satu iterasi k**:

```
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
 Bobot & bias θ ─► (1) FORWARD ─► (2) LOSS ─► (3) BACKWARD (BPTT) ─► (4) ADAM
                    prediksi ŷ'    seberapa     gradien g = ∂L/∂θ      θ baru
                                   salah?       untuk setiap θ
```

| Langkah | Pertanyaan yang dijawab | Persamaan skripsi | Dihitung di |
|---|---|---|---|
| 1. Forward pass | Dengan bobot sekarang, berapa prediksinya? | (15)-(21) | subbagian ini (Tahap 7.2) |
| 2. Loss | Seberapa salah prediksinya? | (28) | subbagian ini (Tahap 7.2) |
| 3. Backward (BPTT) | Bobot mana yang menyebabkan salah, ke arah mana? | aturan rantai | [Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md) |
| 4. Update Adam | Seberapa jauh setiap bobot digeser? | (29)-(31) | [Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md) |

## 2. Kapan Loss dan Adam Dipakai dalam Penelitian

Di penelitian, data latih berisi **804 sampel** (jendela 7 hari ×
10 fitur) dengan **batch size 32**. Jadi satu epoch terdiri dari
⌈804 / 32⌉ = **26 batch**: 25 batch
berisi 32 sampel dan 1 batch terakhir berisi 4 sampel. Untuk
**setiap kombinasi neuron × epoch** pada grid tuning (misalnya 30 neuron,
1,000 epoch), pelatihan berjalan sebagai berikut:

**Langkah 0 — persiapan model.** Bobot diisi acak (Glorot uniform untuk W, ortogonal
untuk U); bias = 0 kecuali bias forget LSTM = 1. Memori Adam di-nol-kan: m₀ = 0,
v₀ = 0, penghitung k = 0.

**Langkah 1-5 — diulang untuk setiap batch**, berurutan secara kronologis karena
`shuffle=False`:

1. **Forward pass.** 32 jendela masuk ke lapisan rekuren lalu dense,
   menghasilkan 32 prediksi ŷ′.
2. **Hitung loss.** MSE dari 32 prediksi itu terhadap nilai aktual y′.
3. **Hitung gradien.** BPTT menghasilkan gₖ untuk **setiap parameter**.
4. **Update Adam.** Setiap parameter digeser memakai persamaan (29)-(31).
5. k bertambah 1, lalu lanjut ke batch berikutnya.

**Langkah 6 — akhir setiap epoch.** Keras mencatat **loss latih** (rata-rata loss dari
26 batch selama epoch itu) dan menghitung **loss validasi** pada
90 sampel validasi memakai bobot akhir epoch. Loss validasi **hanya diukur;
tidak ada update bobot**. Kedua angka ini digambar sebagai kurva loss.

**Langkah 7 — setelah semua epoch selesai.** Bobot pada **epoch terakhir** yang dipakai
(penelitian tidak memakai *early stopping*). Prediksi data validasi didenormalisasi ke
USD, lalu **RMSE validasi** dipakai untuk memilih kombinasi neuron × epoch terbaik.

**Jumlah update Adam per model** (26 per epoch):

| Epoch pada grid | Jumlah update |
|---|---|
| 100 | 26 × 100 = 2,600 |
| 500 | 26 × 500 = 13,000 |
| 1,000 | 26 × 1,000 = 26,000 |

**Kapan loss dan Adam TIDAK dipakai:** saat memilih model terbaik (dipakai RMSE
validasi dalam USD), saat memprediksi data uji (hanya forward pass, bobot sudah beku),
serta saat evaluasi akhir (RMSE, MAE, MAPE, akurasi arah) dan uji Diebold-Mariano.
Singkatnya, **loss dan Adam hanya bekerja di fase pelatihan**.

## 3. Model Mini dan Notasi

| Komponen | Model mini | Model penelitian |
|---|---|---|
| Neuron | 1 | grid 10, 20, 30, 40, 50 |
| Fitur per hari | 1 (harga penutupan ternormalisasi) | 10 |
| Time step (window) | 2 | 7 |
| Sampel per iterasi | 1 (N = 1) | 32 (batch size) |
| Optimizer | Adam, η = 0.001 | Adam, η = 0.001 |

Data model mini: x₁ = 0.5, x₂ = 0.6, target y′ = 0.7; keadaan awal
h₀ = 0 dan c₀ = 0. Bobot awal (sama dengan [Tahap 7.1 (Alur Sel LSTM (Gambar 1-6) dengan Model Mini)](tahap_07_01_alur_sel_lstm.md)):

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b (bias) |
|---|---|---|---|
| forget (f) | 0.5 | 0.4 | 1 (unit forget bias) |
| input (i) | 0.6 | 0.3 | 0 |
| kandidat (c̃) | 0.7 | 0.2 | 0 |
| output (o) | 0.4 | 0.5 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 |

```
Jumlah parameter model mini (persamaan 22), n_u = 1, n_f = 1:
  P = 4 × [n_u × (n_u + n_f) + n_u] + (n_u + 1) = 4 × [1 × (1 + 1) + 1] + (1 + 1) = 4 × 3 + 2 = 14 parameter
```

**Notasi yang dipakai:**

| Simbol | Arti |
|---|---|
| a | pra-aktivasi: nilai sebelum σ atau tanh, misalnya a = W·x + U·h + b |
| σ, tanh | fungsi aktivasi sigmoid (persamaan 11) dan tanh (persamaan 12) |
| ∂L/∂θ | turunan parsial loss terhadap θ; variabel lain dianggap konstan |
| δ | sinyal kesalahan sebuah gerbang = ∂L/∂a (turunan loss terhadap pra-aktivasinya) |
| g | gradien sebuah parameter = ∂L/∂θ |
| m, v | momen pertama dan kedua Adam (persamaan 29) |
| k | nomor iterasi (satu kali update = satu batch) |

## 4. Langkah 1: Forward Pass

Rumus versi 1 neuron (persamaan 15-21). Karena hanya ada satu neuron, semua bobot
berupa angka tunggal (skalar), bukan matriks. Makna setiap gerbang dijelaskan di
[Tahap 7.1 (Alur Sel LSTM (Gambar 1-6) dengan Model Mini)](tahap_07_01_alur_sel_lstm.md); di sini dihitung angkanya untuk kedua time step.

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \qquad
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \qquad
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

$$c_t = f_t\,c_{t-1} + i_t\,\tilde{c}_t \qquad
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \qquad
h_t = o_t \tanh(c_t) \qquad
\hat{y}' = W_y h_T + b_y$$

```
======================================================================
TIME STEP t = 1   (x₁ = 0.500000, h₀ = 0.000000, c₀ = 0.000000)
======================================================================
Forget gate (persamaan 15, Gambar 3)
  a = W_f × x₁ + U_f × h₀ + b_f
    = 0.5 × 0.500000 + 0.4 × 0.000000 + 1
    = 0.250000 + 0.000000 + 1.000000
    = 1.250000
  f₁ = σ(1.250000) = 1 / (1 + e^(-1.250000)) = 0.777300
Input gate (persamaan 16, Gambar 4)
  a = W_i × x₁ + U_i × h₀ + b_i
    = 0.6 × 0.500000 + 0.3 × 0.000000 + 0
    = 0.300000 + 0.000000 + 0.000000
    = 0.300000
  i₁ = σ(0.300000) = 1 / (1 + e^(-0.300000)) = 0.574443
Kandidat cell state (persamaan 17, Gambar 4)
  a = W_c × x₁ + U_c × h₀ + b_c
    = 0.7 × 0.500000 + 0.2 × 0.000000 + 0
    = 0.350000 + 0.000000 + 0.000000
    = 0.350000
  c̃₁ = tanh(0.350000) = 0.336376
Output gate (persamaan 19, Gambar 6)
  a = W_o × x₁ + U_o × h₀ + b_o
    = 0.4 × 0.500000 + 0.5 × 0.000000 + 0
    = 0.200000 + 0.000000 + 0.000000
    = 0.200000
  o₁ = σ(0.200000) = 1 / (1 + e^(-0.200000)) = 0.549834
Pembaruan cell state (persamaan 18, Gambar 5)
  c₁ = f₁ × c₀ + i₁ × c̃₁
     = 0.777300 × 0.000000 + 0.574443 × 0.336376
     = 0.000000 + 0.193228
     = 0.193228
Hidden state (persamaan 20, Gambar 6)
  h₁ = o₁ × tanh(c₁) = 0.549834 × tanh(0.193228) = 0.549834 × 0.190859 = 0.104941

======================================================================
TIME STEP t = 2   (x₂ = 0.600000, h₁ = 0.104941, c₁ = 0.193228)
======================================================================
Forget gate (persamaan 15, Gambar 3)
  a = W_f × x₂ + U_f × h₁ + b_f
    = 0.5 × 0.600000 + 0.4 × 0.104941 + 1
    = 0.300000 + 0.041976 + 1.000000
    = 1.341976
  f₂ = σ(1.341976) = 1 / (1 + e^(-1.341976)) = 0.792815
Input gate (persamaan 16, Gambar 4)
  a = W_i × x₂ + U_i × h₁ + b_i
    = 0.6 × 0.600000 + 0.3 × 0.104941 + 0
    = 0.360000 + 0.031482 + 0.000000
    = 0.391482
  i₂ = σ(0.391482) = 1 / (1 + e^(-0.391482)) = 0.596639
Kandidat cell state (persamaan 17, Gambar 4)
  a = W_c × x₂ + U_c × h₁ + b_c
    = 0.7 × 0.600000 + 0.2 × 0.104941 + 0
    = 0.420000 + 0.020988 + 0.000000
    = 0.440988
  c̃₂ = tanh(0.440988) = 0.414463
Output gate (persamaan 19, Gambar 6)
  a = W_o × x₂ + U_o × h₁ + b_o
    = 0.4 × 0.600000 + 0.5 × 0.104941 + 0
    = 0.240000 + 0.052470 + 0.000000
    = 0.292470
  o₂ = σ(0.292470) = 1 / (1 + e^(-0.292470)) = 0.572601
Pembaruan cell state (persamaan 18, Gambar 5)
  c₂ = f₂ × c₁ + i₂ × c̃₂
     = 0.792815 × 0.193228 + 0.596639 × 0.414463
     = 0.153194 + 0.247285
     = 0.400479
Hidden state (persamaan 20, Gambar 6)
  h₂ = o₂ × tanh(c₂) = 0.572601 × tanh(0.400479) = 0.572601 × 0.380359 = 0.217794

======================================================================
LAPISAN DENSE (persamaan 21) — hanya h₂ (time step terakhir) yang dipakai
======================================================================
  ŷ' = W_y × h₂ + b_y = 0.8 × 0.217794 + 0 = 0.174235
```

| t | x | f | i | c̃ | o | c | h |
|---|---|---|---|---|---|---|---|
| 1 | 0.500000 | 0.777300 | 0.574443 | 0.336376 | 0.549834 | 0.193228 | 0.104941 |
| 2 | 0.600000 | 0.792815 | 0.596639 | 0.414463 | 0.572601 | 0.400479 | 0.217794 |

## 5. Langkah 2: Loss MSE

Persamaan (28):

$$\mathcal{L} = \frac{1}{N}\sum_{k=1}^{N}\left(y'_k - \hat{y}'_k\right)^2$$

**Cara kerjanya**, dengan contoh 3 sampel:

| Sampel | Aktual y′ | Prediksi ŷ′ | Selisih | Kuadrat |
|---|---|---|---|---|
| 1 | 0.50 | 0.48 | 0.02 | 0.0004 |
| 2 | 0.52 | 0.53 | -0.01 | 0.0001 |
| 3 | 0.55 | 0.51 | 0.04 | 0.0016 |

```
MSE = (0.0004 + 0.0001 + 0.0016) / 3 = 0.0021 / 3 = 0.0007
```

**Mengapa dikuadratkan?**

1. Selisih positif dan negatif tidak saling meniadakan.
2. Kesalahan besar dihukum jauh lebih berat: selisih 0.04 menyumbang
   16 kali lebih besar daripada selisih 0.01, sehingga model
   "dipaksa" menghindari meleset jauh.
3. Fungsi kuadrat **mulus dan bisa diturunkan di semua titik**. Turunannya,
   ∂L/∂ŷ′ = −2(y′ − ŷ′), menjadi titik awal BPTT ([Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md)). MAE (nilai
   mutlak) tidak mulus di titik nol.

**Mengapa dihitung pada skala ternormalisasi, bukan USD?** Harga Bitcoin bernilai
puluhan ribu USD. Selisih 2,500.00 USD jika dikuadratkan menjadi
6,250,000.00, sehingga gradien sangat besar dan pelatihan tidak stabil. Dengan
rentang harga data latih 98,196.75 USD, selisih yang sama pada skala 0-1 hanya
0.025459 dan kuadratnya 0.000648.

**Nilai N dalam praktik:** saat pelatihan N = 32 (ukuran batch; batch
terakhir N = 4), loss validasi dihitung pada N = 90 sampel
validasi, dan pada model mini N = 1.

```
Persamaan (28) dengan N = 1 sampel:
  L = (y' - ŷ')²
    = (0.7 - 0.174235)²
    = 0.525765²
    = 0.276429
```

Prediksi masih **terlalu rendah** (0.174235 padahal seharusnya 0.7). Langkah
berikutnya mencari tahu bobot mana yang perlu diubah agar loss ini turun:
[Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md).

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.1: Alur Sel LSTM (Gambar 1-6) dengan Model Mini](tahap_07_01_alur_sel_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.3: Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time](tahap_07_03_simulasi_bptt_lstm.md)
