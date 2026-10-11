# Perhitungan Manual - Tahap 8.2: Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss

Subbagian ini memulai **simulasi satu siklus pelatihan GRU** pada model mini. Di sini
dihitung dua langkah pertama: **forward pass** (berapa prediksinya) dan **loss**
(seberapa salah prediksinya). Langkah 3 dan 4 dilanjutkan di [Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md) dan
[Tahap 8.4 (Simulasi Pelatihan GRU, Langkah 4: Update Adam)](tahap_08_04_simulasi_update_adam_gru.md).

**Daftar isi**

- [1. Gambaran Satu Siklus Pelatihan](#1-gambaran-satu-siklus-pelatihan)
- [2. Model Mini dan Notasi](#2-model-mini-dan-notasi)
- [3. Langkah 1: Forward Pass](#3-langkah-1-forward-pass)
- [4. Langkah 2: Loss MSE](#4-langkah-2-loss-mse)

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
| 1. Forward pass | Dengan bobot sekarang, berapa prediksinya? | (23)-(26), (21) | subbagian ini (Tahap 8.2) |
| 2. Loss | Seberapa salah prediksinya? | (28) | subbagian ini (Tahap 8.2) |
| 3. Backward (BPTT) | Bobot mana yang menyebabkan salah, ke arah mana? | aturan rantai | [Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md) |
| 4. Update Adam | Seberapa jauh setiap bobot digeser? | (29)-(31) | [Tahap 8.4 (Simulasi Pelatihan GRU, Langkah 4: Update Adam)](tahap_08_04_simulasi_update_adam_gru.md) |

Prosedur pelatihan GRU di penelitian **identik** dengan LSTM: data, batch size,
optimizer Adam, learning rate, dan seed sama persis; yang berbeda hanya jenis lapisan
rekuren. Kapan loss dan Adam dipakai serta berapa kali update terjadi per epoch
dijelaskan di [Tahap 7.2, bagian 2](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md#2-kapan-loss-dan-adam-dipakai-dalam-penelitian).

## 2. Model Mini dan Notasi

| Komponen | Model mini | Model penelitian |
|---|---|---|
| Neuron | 1 | grid 10, 20, 30, 40, 50 |
| Fitur per hari | 1 (harga penutupan ternormalisasi) | 10 |
| Time step (window) | 2 | 7 |
| Sampel per iterasi | 1 (N = 1) | 32 (batch size) |
| Optimizer | Adam, η = 0.001 | Adam, η = 0.001 |

Data model mini: x₁ = 0.5, x₂ = 0.6, target y′ = 0.7; keadaan awal
h₀ = 0. Bobot awal (sama dengan [Tahap 8.1 (Alur Sel GRU (Gambar 7) dengan Model Mini)](tahap_08_01_alur_sel_gru.md)):

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b(in) (bias masukan) | b(rec) (bias rekuren) |
|---|---|---|---|---|
| update (z) | 0.5 | 0.4 | 0 | 0 |
| reset (r) | 0.6 | 0.3 | 0 | 0 |
| kandidat (h̃) | 0.7 | 0.2 | 0 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 | - |

```
Jumlah parameter model mini (persamaan 27), n_u = 1, n_f = 1:
  P = 3 × [n_u × (n_u + n_f) + 2 × n_u] + (n_u + 1) = 3 × [1 × (1 + 1) + 2] + (1 + 1) = 3 × 4 + 2 = 14 parameter
```

**Notasi yang dipakai:**

| Simbol | Arti |
|---|---|
| a | pra-aktivasi: nilai sebelum σ atau tanh, misalnya a = W·x + U·h + b |
| q | bagian rekuren kandidat GRU: qₜ = U_h·hₜ₋₁ + b_h(rec) |
| σ, tanh | fungsi aktivasi sigmoid (persamaan 11) dan tanh (persamaan 12) |
| ∂L/∂θ | turunan parsial loss terhadap θ; variabel lain dianggap konstan |
| δ | sinyal kesalahan sebuah gerbang = ∂L/∂a (turunan loss terhadap pra-aktivasinya) |
| g | gradien sebuah parameter = ∂L/∂θ |
| m, v | momen pertama dan kedua Adam (persamaan 29) |
| k | nomor iterasi (satu kali update = satu batch) |

## 3. Langkah 1: Forward Pass

Rumus versi 1 neuron (persamaan 23-26, konvensi Keras `reset_after=True`). Untuk
kandidat, bagian rekurennya diberi nama $q_t$ supaya terlihat jelas apa yang dikalikan
reset gate. Alur setiap langkah pada Gambar 7 dijelaskan di [Tahap 8.1 (Alur Sel GRU (Gambar 7) dengan Model Mini)](tahap_08_01_alur_sel_gru.md);
di sini dihitung angkanya untuk kedua time step.

$$z_t = \sigma\bigl(W_z x_t + U_z h_{t-1} + b_z^{(in)} + b_z^{(rec)}\bigr) \qquad
r_t = \sigma\bigl(W_r x_t + U_r h_{t-1} + b_r^{(in)} + b_r^{(rec)}\bigr)$$

$$q_t = U_h h_{t-1} + b_h^{(rec)} \qquad
\tilde{h}_t = \tanh\bigl(W_h x_t + b_h^{(in)} + r_t\,q_t\bigr) \qquad
h_t = z_t\,h_{t-1} + (1-z_t)\,\tilde{h}_t \qquad
\hat{y}' = W_y h_T + b_y$$

```
======================================================================
TIME STEP t = 1   (x₁ = 0.500000, h₀ = 0.000000)
======================================================================
Update gate (persamaan 23, Gambar 7: kotak σ kiri)
  a = W_z × x₁ + U_z × h₀ + b_z(in) + b_z(rec)
    = 0.5 × 0.500000 + 0.4 × 0.000000 + 0 + 0
    = 0.250000 + 0.000000 + 0.000000
    = 0.250000
  z₁ = σ(0.250000) = 1 / (1 + e^(-0.250000)) = 0.562177
Reset gate (persamaan 24, Gambar 7: kotak σ tengah)
  a = W_r × x₁ + U_r × h₀ + b_r(in) + b_r(rec)
    = 0.6 × 0.500000 + 0.3 × 0.000000 + 0 + 0
    = 0.300000 + 0.000000 + 0.000000
    = 0.300000
  r₁ = σ(0.300000) = 1 / (1 + e^(-0.300000)) = 0.574443
Kandidat hidden state (persamaan 25, Gambar 7: kotak tanh)
  bagian rekuren  q₁ = U_h × h₀ + b_h(rec) = 0.2 × 0.000000 + 0 = 0.000000
  bagian masukan       = W_h × x₁ + b_h(in) = 0.7 × 0.500000 + 0 = 0.350000
  a = bagian masukan + r₁ × q₁ = 0.350000 + 0.574443 × 0.000000 = 0.350000 + 0.000000 = 0.350000
  h̃₁ = tanh(0.350000) = 0.336376
Hidden state baru (persamaan 26, Gambar 7: lingkaran ×, 1−, dan +)
  h₁ = z₁ × h₀ + (1 - z₁) × h̃₁
     = 0.562177 × 0.000000 + 0.437823 × 0.336376
     = 0.000000 + 0.147273
     = 0.147273
  (q₁ = 0 karena h₀ = 0, jadi reset gate belum berpengaruh pada t = 1)

======================================================================
TIME STEP t = 2   (x₂ = 0.600000, h₁ = 0.147273)
======================================================================
Update gate (persamaan 23, Gambar 7: kotak σ kiri)
  a = W_z × x₂ + U_z × h₁ + b_z(in) + b_z(rec)
    = 0.5 × 0.600000 + 0.4 × 0.147273 + 0 + 0
    = 0.300000 + 0.058909 + 0.000000
    = 0.358909
  z₂ = σ(0.358909) = 1 / (1 + e^(-0.358909)) = 0.588776
Reset gate (persamaan 24, Gambar 7: kotak σ tengah)
  a = W_r × x₂ + U_r × h₁ + b_r(in) + b_r(rec)
    = 0.6 × 0.600000 + 0.3 × 0.147273 + 0 + 0
    = 0.360000 + 0.044182 + 0.000000
    = 0.404182
  r₂ = σ(0.404182) = 1 / (1 + e^(-0.404182)) = 0.599692
Kandidat hidden state (persamaan 25, Gambar 7: kotak tanh)
  bagian rekuren  q₂ = U_h × h₁ + b_h(rec) = 0.2 × 0.147273 + 0 = 0.029455
  bagian masukan       = W_h × x₂ + b_h(in) = 0.7 × 0.600000 + 0 = 0.420000
  a = bagian masukan + r₂ × q₂ = 0.420000 + 0.599692 × 0.029455 = 0.420000 + 0.017664 = 0.437664
  h̃₂ = tanh(0.437664) = 0.411706
Hidden state baru (persamaan 26, Gambar 7: lingkaran ×, 1−, dan +)
  h₂ = z₂ × h₁ + (1 - z₂) × h̃₂
     = 0.588776 × 0.147273 + 0.411224 × 0.411706
     = 0.086711 + 0.169303
     = 0.256014

======================================================================
LAPISAN DENSE (persamaan 21)
======================================================================
  ŷ' = W_y × h₂ + b_y = 0.8 × 0.256014 + 0 = 0.204811
```

| t | x | z | r | q | h̃ | h |
|---|---|---|---|---|---|---|
| 1 | 0.500000 | 0.562177 | 0.574443 | 0.000000 | 0.336376 | 0.147273 |
| 2 | 0.600000 | 0.588776 | 0.599692 | 0.029455 | 0.411706 | 0.256014 |

## 4. Langkah 2: Loss MSE

Fungsi loss sama dengan LSTM, yaitu MSE (persamaan 28). Alasan dikuadratkan dan alasan
dihitung pada skala ternormalisasi dijelaskan di
[Tahap 7.2, bagian 5](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md#5-langkah-2-loss-mse). Pada model mini N = 1:

```
Persamaan (28) dengan N = 1 sampel:
  L = (y' - ŷ')²
    = (0.7 - 0.204811)²
    = 0.495189²
    = 0.245212
```

Prediksi masih **terlalu rendah** (0.204811 padahal seharusnya 0.7). Langkah
berikutnya mencari tahu bobot mana yang perlu diubah agar loss ini turun:
[Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md).

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.1: Alur Sel GRU (Gambar 7) dengan Model Mini](tahap_08_01_alur_sel_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.3: Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time](tahap_08_03_simulasi_bptt_gru.md)
