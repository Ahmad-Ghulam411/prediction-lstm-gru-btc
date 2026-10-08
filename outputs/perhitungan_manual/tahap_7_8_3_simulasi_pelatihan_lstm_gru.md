# Perhitungan Manual - Tahap 7 & 8 (Bagian 3 dari 4): Simulasi Satu Siklus Pelatihan LSTM dan GRU

> Berkas ini dibuat otomatis oleh `tools/simulasi_pelatihan_manual.py`. Semua
> angka dihitung ulang oleh skrip itu dan diperiksa dengan `assert`. Untuk membuat
> ulang seluruh seri: `python tools/simulasi_pelatihan_manual.py`.

**Seri perhitungan manual pelatihan LSTM dan GRU** (baca berurutan):

1. [Alur Sel LSTM dan GRU (Gambar 1-7)](tahap_7_8_1_alur_sel_lstm_gru.md)
2. [Fungsi Loss dan Adam](tahap_7_8_2_fungsi_loss_dan_adam.md)
3. **Simulasi Satu Siklus Pelatihan LSTM dan GRU** ← sedang dibaca
4. [Bias pada LSTM dan GRU](tahap_7_8_4_bias_lstm_gru.md)

[Bagian 1](tahap_7_8_1_alur_sel_lstm_gru.md) menjelaskan makna setiap gerbang pada Gambar 1-7, dan [Bagian 2](tahap_7_8_2_fungsi_loss_dan_adam.md)
menjelaskan konsep fungsi loss serta Adam. Bagian ini **menghitung satu siklus
pelatihan secara lengkap**: *forward pass* → *loss* → *backpropagation through time*
(BPTT) → Adam, sampai bobot dan bias berubah dan loss turun.

Berkas `tahap_7_forward_pass_lstm.md` dan `tahap_8_forward_pass_gru.md`
menunjukkan bagaimana model yang **sudah dilatih** menghasilkan prediksi. Bagian
ini menjawab pertanyaan sebelumnya: **bagaimana bobot dan bias itu diperoleh?**

Agar setiap angka bisa diikuti dengan tangan, simulasi memakai **model mini**:
1 neuron, 1 fitur, 2 time step, dan 1 sampel. Rumus dan urutan langkahnya
**identik** dengan model penelitian; yang berbeda hanya jumlah angkanya.
Subbagian 5 menjelaskan cara memperbesarnya ke model penelitian
(10 fitur, 7 time step, batch 32,
40/30 neuron).

**Daftar isi**

- [1. Asumsi Simulasi, Notasi, dan Siklus yang Dihitung](#1-asumsi-simulasi-notasi-dan-siklus-yang-dihitung)
- [2. Rumus Turunan Dasar yang Dipakai](#2-rumus-turunan-dasar-yang-dipakai)
- [3. LSTM: Satu Siklus Pelatihan Langkah demi Langkah](#3-lstm-satu-siklus-pelatihan-langkah-demi-langkah)
- [4. GRU: Satu Siklus Pelatihan Langkah demi Langkah](#4-gru-satu-siklus-pelatihan-langkah-demi-langkah)
- [5. Dari Model Mini ke Model Penelitian](#5-dari-model-mini-ke-model-penelitian)
- [6. Ringkasan Alur dan Hasil Verifikasi](#6-ringkasan-alur-dan-hasil-verifikasi)

**Cara membaca angka.** Seperti berkas perhitungan manual lainnya, angka memakai
titik sebagai pemisah desimal dan koma sebagai pemisah ribuan. Angka ditampilkan
6 desimal, tetapi skrip menghitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Asumsi Simulasi, Notasi, dan Siklus yang Dihitung

| Komponen | Model mini (berkas ini) | Model penelitian |
|---|---|---|
| Neuron | 1 | LSTM 40, GRU 30 |
| Fitur per hari | 1 (harga penutupan ternormalisasi) | 10 |
| Time step (window) | 2 | 7 |
| Sampel per iterasi | 1 (N = 1) | 32 (batch size) |
| Optimizer | Adam, η = 0.001 | Adam, η = 0.001 |

**Data.** Harga penutupan ternormalisasi dua hari dipakai untuk memprediksi hari
ketiga (angka ilustrasi):

- hari 1: x₁ = 0.5
- hari 2: x₂ = 0.6
- target hari 3: y′ = 0.7

**Keadaan awal.** h₀ = 0 dan c₀ = 0, sama seperti bawaan Keras.

**Adam.** η = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁷ (bawaan Keras, sama dengan penelitian).

**Bobot awal.** Dibuat bulat agar mudah dihitung. Keras sebenarnya mengisi
bobot secara acak (Glorot uniform untuk W, ortogonal untuk U), tetapi aturan
biasnya diikuti: **bias forget gate LSTM = 1**, bias lain = 0.

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

**Siklus yang dihitung.** Satu siklus di bawah ini disebut **satu iterasi k**.
Model penelitian mengulangnya ribuan kali; berapa kali dan kapan persisnya
dijelaskan di [Bagian 2, subbagian 2](tahap_7_8_2_fungsi_loss_dan_adam.md#2-kapan-dipakai-alur-pelatihan-di-penelitian).

```
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
 Bobot & bias θ ─► (1) FORWARD ─► (2) LOSS ─► (3) BACKWARD (BPTT) ─► (4) ADAM
                    prediksi ŷ'    seberapa     gradien g = ∂L/∂θ      θ baru
                                   salah?       untuk setiap θ
```

| Langkah | Pertanyaan yang dijawab | Persamaan skripsi | LSTM | GRU |
|---|---|---|---|---|
| (1) Forward pass | Dengan bobot sekarang, berapa prediksinya? | (15)-(21) LSTM, (23)-(26) GRU | 3.2 | 4.2 |
| (2) Loss | Seberapa salah prediksinya? | (28) | 3.3 | 4.3 |
| (3) Backward (BPTT) | Bobot mana yang menyebabkan salah, ke arah mana? | aturan rantai (subbagian 2) | 3.4-3.9 | 4.4-4.9 |
| (4) Adam | Seberapa jauh setiap bobot digeser? | (29)-(31) | 3.10-3.13 | 4.10-4.13 |

## 2. Rumus Turunan Dasar yang Dipakai

Seluruh backward pass hanya mengulang beberapa rumus turunan berikut. Polanya
selalu sama: **kalikan sinyal kesalahan dengan turunan lokal, lalu teruskan ke
kiri (ke langkah sebelumnya).**

| Rumus | Bentuk | Contoh | Dipakai di |
|---|---|---|---|
| Aturan pangkat | d(u²)/du = 2u | d(x²)/dx = 2x | ∂L/∂ŷ′ |
| Fungsi linear | d(a·u + b)/du = a | d(3x + 5)/dx = 3 | pra-aktivasi dan dense: ∂a/∂W = x, ∂a/∂U = hₜ₋₁, ∂a/∂b = 1 |
| Aturan perkalian | ∂(u·v)/∂u = v | ∂(3u)/∂u = 3 | h = o·tanh(c), c = f·cₜ₋₁ + i·c̃, GRU h = z·hₜ₋₁ + (1−z)·h̃ |
| Aturan rantai | ∂L/∂a = ∂L/∂b · ∂b/∂a | - | semua langkah |
| Aturan penjumlahan | variabel yang dipakai di beberapa tempat: gradiennya dijumlah | - | h₁ dipakai 4 gerbang; bobot dipakai di setiap t; c₁ punya 2 jalur |
| Turunan sigmoid | σ′(a) = σ(a)·(1 − σ(a)) | σ(1.25) = 0.777300 → σ′ = 0.173105 | gerbang f, i, o, z, r |
| Turunan tanh | tanh′(a) = 1 − tanh²(a) | tanh(0.35) = 0.336376 → 0.886851 | c̃, h̃, tanh(c) |

**Asal turunan sigmoid** (dari persamaan 11):

$$\sigma(a) = \frac{1}{1+e^{-a}}
\;\Rightarrow\;
\sigma'(a) = \frac{e^{-a}}{(1+e^{-a})^2}
= \frac{1}{1+e^{-a}}\cdot\frac{e^{-a}}{1+e^{-a}}
= \sigma(a)\,\bigl(1-\sigma(a)\bigr)$$

**Asal turunan tanh** (dari persamaan 13, $\tanh(a) = 2\sigma(2a) - 1$):

$$\tanh'(a) = 4\,\sigma(2a)\bigl(1-\sigma(2a)\bigr)
= 4\cdot\frac{1+\tanh(a)}{2}\cdot\frac{1-\tanh(a)}{2}
= 1-\tanh^2(a)$$

## 3. LSTM: Satu Siklus Pelatihan Langkah demi Langkah

### 3.1 Bobot Awal dan Jumlah Parameter

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b (bias) |
|---|---|---|---|
| forget (f) | 0.5 | 0.4 | 1 (unit forget bias) |
| input (i) | 0.6 | 0.3 | 0 |
| kandidat (c̃) | 0.7 | 0.2 | 0 |
| output (o) | 0.4 | 0.5 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 |

```
Jumlah parameter (persamaan 22) dengan n_u = 1 neuron dan n_f = 1 fitur:
  P_LSTM = 4 × [n_u × (n_u + n_f) + n_u] + (n_u + 1)
         = 4 × [1 × (1 + 1) + 1] + (1 + 1)
         = 4 × 3 + 2
         = 14 parameter  (12 di lapisan LSTM + 2 di lapisan dense)
```

### 3.2 Langkah 1: Forward Pass

Makna setiap gerbang dan alurnya pada Gambar 1-6 dijelaskan di [Bagian 1](tahap_7_8_1_alur_sel_lstm_gru.md); di sini dihitung angkanya untuk kedua time step.

Rumus versi 1 neuron (persamaan 15-21). Karena hanya ada satu neuron, semua
bobot berupa angka tunggal (skalar), bukan matriks:

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

### 3.3 Langkah 2: Loss

```
Persamaan (28) dengan N = 1 sampel:
  L = (y' - ŷ')²
    = (0.7 - 0.174235)²
    = 0.525765²
    = 0.276429
```

Prediksi masih **terlalu rendah** (0.174235 padahal seharusnya 0.7).
Langkah berikutnya mencari tahu bobot mana yang perlu diubah agar loss ini turun.
Konsep MSE dan alasan pemakaiannya dibahas di [Bagian 2, subbagian 3](tahap_7_8_2_fungsi_loss_dan_adam.md#3-fungsi-loss-mean-squared-error).

### 3.4 Langkah 3a: Backward di Lapisan Dense

Backward berjalan dari kanan ke kiri, mulai dari loss. Rantainya:
`h₂ → ŷ' = W_y·h₂ + b_y → L = (y' − ŷ')²`. Setiap turunan memakai rumus
dari bagian 2.

```
Diketahui: h₂ = 0.217794, W_y = 0.8, b_y = 0, ŷ' = 0.174235, y' = 0.7

(1) ∂L/∂ŷ' — seberapa besar loss berubah bila prediksi berubah
    L = (y' - ŷ')²
    misalkan u = y' - ŷ'           →  L = u²
    dL/du  = 2u                     (aturan pangkat)
    du/dŷ' = 0 - 1 = -1             (y' adalah data, turunannya 0)
    ∂L/∂ŷ' = 2u × (-1) = -2(y' - ŷ')   (aturan rantai)
           = -2 × (0.7 - 0.174235)
           = -2 × 0.525765
           = -1.051530
    Cara lain tanpa aturan rantai: L = y'² - 2y'ŷ' + ŷ'²
                                   ∂L/∂ŷ' = -2y' + 2ŷ' = 2(ŷ' - y')  (sama)

(2) ∂L/∂W_y — gradien bobot dense
    ŷ' = W_y × h₂ + b_y  →  ∂ŷ'/∂W_y = h₂   (fungsi linear; h₂ dan b_y dianggap konstan)
    ∂L/∂W_y = ∂L/∂ŷ' × h₂ = (-1.051530) × 0.217794 = -0.229017

(3) ∂L/∂b_y — gradien bias dense
    ∂ŷ'/∂b_y = 1
    ∂L/∂b_y = ∂L/∂ŷ' × 1 = -1.051530

(4) ∂L/∂h₂ — sinyal kesalahan yang dikirim MASUK ke sel LSTM (awal BPTT)
    ∂ŷ'/∂h₂ = W_y            (kali ini h₂ yang menjadi variabel)
    ∂L/∂h₂ = ∂L/∂ŷ' × W_y = (-1.051530) × 0.8 = -0.841224
```

**Makna setiap angka:**

- **∂L/∂ŷ′ = -1.051530**: tanda negatif berarti menaikkan prediksi akan
  menurunkan loss (sesuai, karena prediksi masih terlalu rendah). Jika ŷ′ naik
  0.001, loss turun sekitar 0.0010515. Nilai ini adalah **sinyal
  kesalahan keluaran** yang dipakai ulang oleh tiga turunan berikutnya.
- **∂L/∂W_y = -0.229017**: pengaruh W_y terhadap prediksi "dikali" h₂.
  Karena h₂ hanya 0.217794, gradien W_y lebih kecil daripada gradien ŷ′.
- **∂L/∂b_y = -1.051530**: bias langsung ditambahkan ke prediksi, jadi
  gradiennya sama dengan sinyal kesalahan keluaran.
- **∂L/∂h₂ = -0.841224**: h₂ **bukan parameter**, jadi tidak
  diupdate Adam. Nilai ini adalah pintu masuk BPTT ke dalam sel LSTM.

**Bukti dengan uji geser.** Turunan berarti "perubahan loss bila variabel
digeser sedikit". Setiap besaran digeser +0.001, lalu loss dihitung ulang:

| Yang digeser +0.001 | Perkiraan dari gradien (gradien × 0.001) | Perubahan loss sebenarnya |
|---|---|---|
| ŷ' | (-1.051530) × 0.001 = -0.0010515 | -0.0010505 |
| W_y (0.8 → 0.801) | (-0.229017) × 0.001 = -0.0002290 | -0.0002290 |
| b_y (0 → 0.001) | (-1.051530) × 0.001 = -0.0010515 | -0.0010505 |
| h₂ | (-0.841224) × 0.001 = -0.0008412 | -0.0008406 |

Selisih kecil (sekitar 0.000001) muncul karena turunan adalah pendekatan garis
lurus, sedangkan loss berbentuk kuadrat; sisanya sebesar (0.001 × ∂ŷ′/∂θ)².

### 3.5 Langkah 3b: Backward di Dalam Sel, t = 2

Sinyal ∂L/∂h₂ masuk ke sel lewat dua rumus forward yang diturunkan:
`h₂ = o₂ × tanh(c₂)` (menuju output gate dan cell state), lalu
`c₂ = f₂ × c₁ + i₂ × c̃₂` (menuju forget gate, input gate, dan kandidat).
Setiap gerbang lalu melewati turunan aktivasinya (σ′ atau tanh′) sehingga
diperoleh sinyal kesalahan gerbang **δ**.

```
Masukan dari 3.4: ∂L/∂h₂ = -0.841224

(a) Output gate — turunkan h₂ = o₂ × tanh(c₂) terhadap o₂ (aturan perkalian)
    ∂L/∂o₂ = ∂L/∂h₂ × tanh(c₂) = (-0.841224) × 0.380359 = -0.319967
    δo₂    = ∂L/∂o₂ × o₂(1 - o₂)                     (turunan sigmoid)
           = (-0.319967) × 0.572601 × 0.427399
           = (-0.319967) × 0.244729
           = -0.078305

(b) Cell state — turunkan h₂ = o₂ × tanh(c₂) terhadap c₂ (perkalian + turunan tanh)
    ∂L/∂c₂ = ∂L/∂h₂ × o₂ × (1 - tanh²(c₂))
           = (-0.841224) × 0.572601 × (1 - 0.380359²)
           = (-0.841224) × 0.572601 × 0.855327
           = -0.411999
    (t = 2 adalah time step terakhir, jadi belum ada kiriman dari t = 3)

(c) Forget gate — turunkan c₂ = f₂ × c₁ + i₂ × c̃₂ terhadap f₂
    ∂L/∂f₂ = ∂L/∂c₂ × c₁ = (-0.411999) × 0.193228 = -0.079610
    δf₂    = ∂L/∂f₂ × f₂(1 - f₂) = (-0.079610) × 0.164260 = -0.013077

(d) Input gate — turunkan c₂ terhadap i₂
    ∂L/∂i₂ = ∂L/∂c₂ × c̃₂ = (-0.411999) × 0.414463 = -0.170758
    δi₂    = ∂L/∂i₂ × i₂(1 - i₂) = (-0.170758) × 0.240661 = -0.041095

(e) Kandidat — turunkan c₂ terhadap c̃₂
    ∂L/∂c̃₂ = ∂L/∂c₂ × i₂ = (-0.411999) × 0.596639 = -0.245815
    δc̃₂    = ∂L/∂c̃₂ × (1 - c̃₂²) = (-0.245815) × 0.828220 = -0.203589
```

### 3.6 Langkah 3c: Mengirim Kesalahan ke t = 1 (Inti "Through Time")

Kesalahan di t = 2 sebagian disebabkan oleh keadaan hari sebelumnya, jadi
sinyalnya dikirim mundur ke t = 1 lewat **dua jalur**:

1. **Jalur hidden state h₁.** h₁ dipakai oleh keempat gerbang di t = 2
   (lewat bobot U), jadi keempat kontribusinya dijumlahkan (aturan penjumlahan).
2. **Jalur cell state c₁.** Dari `c₂ = f₂ × c₁ + ...`, turunannya terhadap c₁
   adalah f₂. Jalur ini hanya berupa perkalian sederhana, itulah sebabnya
   disebut "jalan tol gradien".

```
Jalur 1 — hidden state:
  ∂L/∂h₁ = U_f × δf₂ + U_i × δi₂ + U_c × δc̃₂ + U_o × δo₂
         = 0.4 × (-0.013077) + 0.3 × (-0.041095) + 0.2 × (-0.203589) + 0.5 × (-0.078305)
         = (-0.005231) + (-0.012328) + (-0.040718) + (-0.039153)
         = -0.097429

Jalur 2 — cell state:
  kiriman ke c₁ = ∂L/∂c₂ × f₂ = (-0.411999) × 0.792815 = -0.326639
```

### 3.7 Langkah 3d: Backward di Dalam Sel, t = 1

Di t = 1 rumusnya sama dengan 3.5. Bedanya, cell state c₁ menerima **dua
kiriman** (dari jalur cell state dan dari h₁), lalu keduanya dijumlahkan.

```
Masukan dari 3.6: ∂L/∂h₁ = -0.097429, kiriman jalur cell state = -0.326639

(a) Output gate
    ∂L/∂o₁ = ∂L/∂h₁ × tanh(c₁) = (-0.097429) × 0.190859 = -0.018595
    δo₁    = (-0.018595) × 0.247517 = -0.004603

(b) Cell state c₁ = [kiriman jalur cell state] + [kiriman lewat h₁ = o₁ × tanh(c₁)]
    ∂L/∂c₁ = ∂L/∂c₂ × f₂ + ∂L/∂h₁ × o₁ × (1 - tanh²(c₁))
           = (-0.326639) + (-0.097429) × 0.549834 × 0.963573
           = (-0.326639) + (-0.051619)
           = -0.378257
    Porsi jalur cell state = -0.326639 / -0.378257 = 86.4%

(c) Forget gate
    ∂L/∂f₁ = ∂L/∂c₁ × c₀ = (-0.378257) × 0.000000 = 0.000000
    δf₁    = 0.000000     (nol karena c₀ = 0: belum ada memori untuk dilupakan)

(d) Input gate
    ∂L/∂i₁ = ∂L/∂c₁ × c̃₁ = (-0.378257) × 0.336376 = -0.127236
    δi₁    = (-0.127236) × 0.244458 = -0.031104

(e) Kandidat
    ∂L/∂c̃₁ = ∂L/∂c₁ × i₁ = (-0.378257) × 0.574443 = -0.217287
    δc̃₁    = (-0.217287) × 0.886851 = -0.192701
```

Sekitar **86.4%** sinyal kesalahan ke c₁ datang lewat jalur cell
state. Inilah wujud nyata alasan LSTM tahan terhadap *vanishing gradient*: di
jalur ini gradien hanya dikalikan f (bukan melewati tanh dan matriks bobot
berulang kali seperti pada RNN biasa).

### 3.8 Langkah 3e: Gradien Total Setiap Bobot dan Bias

Bobot yang sama dipakai di t = 1 **dan** t = 2, sehingga kontribusi kedua time
step dijumlahkan (aturan penjumlahan). Dari $a_t = W x_t + U h_{t-1} + b$
diperoleh $\partial a/\partial W = x_t$, $\partial a/\partial U = h_{t-1}$, dan
$\partial a/\partial b = 1$, sehingga:

$$\frac{\partial L}{\partial W} = \sum_t \delta_t\,x_t \qquad
\frac{\partial L}{\partial U} = \sum_t \delta_t\,h_{t-1} \qquad
\frac{\partial L}{\partial b} = \sum_t \delta_t$$

```
Forget gate:  δf₁ = 0.000000,  δf₂ = -0.013077
  ∂L/∂W_f = δf₁ × x₁ + δf₂ × x₂ = 0.000000 × 0.5 + (-0.013077) × 0.6 = 0.000000 + (-0.007846) = -0.007846
  ∂L/∂U_f = δf₁ × h₀ + δf₂ × h₁ = 0.000000 × 0.000000 + (-0.013077) × 0.104941 = -0.001372
  ∂L/∂b_f = δf₁ + δf₂ = 0.000000 + (-0.013077) = -0.013077

Input gate:  δi₁ = -0.031104,  δi₂ = -0.041095
  ∂L/∂W_i = δi₁ × x₁ + δi₂ × x₂ = (-0.031104) × 0.5 + (-0.041095) × 0.6 = (-0.015552) + (-0.024657) = -0.040209
  ∂L/∂U_i = δi₁ × h₀ + δi₂ × h₁ = (-0.031104) × 0.000000 + (-0.041095) × 0.104941 = -0.004313
  ∂L/∂b_i = δi₁ + δi₂ = (-0.031104) + (-0.041095) = -0.072199

Kandidat (c̃):  δc̃₁ = -0.192701,  δc̃₂ = -0.203589
  ∂L/∂W_c = δc̃₁ × x₁ + δc̃₂ × x₂ = (-0.192701) × 0.5 + (-0.203589) × 0.6 = (-0.096351) + (-0.122153) = -0.218504
  ∂L/∂U_c = δc̃₁ × h₀ + δc̃₂ × h₁ = (-0.192701) × 0.000000 + (-0.203589) × 0.104941 = -0.021365
  ∂L/∂b_c = δc̃₁ + δc̃₂ = (-0.192701) + (-0.203589) = -0.396290

Output gate:  δo₁ = -0.004603,  δo₂ = -0.078305
  ∂L/∂W_o = δo₁ × x₁ + δo₂ × x₂ = (-0.004603) × 0.5 + (-0.078305) × 0.6 = (-0.002301) + (-0.046983) = -0.049284
  ∂L/∂U_o = δo₁ × h₀ + δo₂ × h₁ = (-0.004603) × 0.000000 + (-0.078305) × 0.104941 = -0.008217
  ∂L/∂b_o = δo₁ + δo₂ = (-0.004603) + (-0.078305) = -0.082908

Lapisan dense (dari 3.4):
  ∂L/∂W_y = -0.229017
  ∂L/∂b_y = -1.051530
```

**Semua gradien bernilai negatif.** Prediksi terlalu rendah, dan setiap bobot
awal bernilai positif, sehingga menaikkan bobot mana pun akan menaikkan ŷ′.
Karena itu langkah Adam berikutnya akan menaikkan semua parameter. Perhatikan
juga **∂L/∂U = δ₂ × h₁ saja**: pada t = 1 bobot U tidak mendapat gradien karena
h₀ = 0, sedangkan bias tetap mendapat sinyal dari kedua time step.

### 3.9 Pemeriksaan Gradien dengan Turunan Numerik

Untuk memastikan seluruh rantai turunan di atas benar, setiap gradien
dibandingkan dengan turunan numerik (beda pusat):
$\dfrac{L(\theta+10^{-6}) - L(\theta-10^{-6})}{2\cdot10^{-6}}$.

| Parameter | Gradien BPTT (manual) | Gradien numerik | Selisih |
|---|---|---|---|
| W_f | -0.00784600 | -0.00784600 | 4.6e-11 |
| U_f | -0.00137228 | -0.00137228 | 2.3e-11 |
| b_f | -0.01307667 | -0.01307667 | 2.2e-11 |
| W_i | -0.04020889 | -0.04020889 | 1.1e-11 |
| U_i | -0.00431252 | -0.00431252 | 9.3e-11 |
| b_i | -0.07219882 | -0.07219882 | 3.0e-11 |
| W_c | -0.21850381 | -0.21850381 | 2.9e-11 |
| U_c | -0.02136474 | -0.02136474 | 4.9e-11 |
| b_c | -0.39628990 | -0.39628990 | 8.4e-12 |
| W_o | -0.04928448 | -0.04928448 | 1.6e-11 |
| U_o | -0.00821741 | -0.00821741 | 2.3e-11 |
| b_o | -0.08290791 | -0.08290791 | 5.1e-12 |
| W_y | -0.22901679 | -0.22901679 | 6.8e-12 |
| b_y | -1.05152971 | -1.05152971 | 3.1e-11 |

### 3.10 Langkah 4: Update Adam, Iterasi k = 1

Rumus Adam (persamaan 29-31) diterapkan **untuk setiap parameter secara
terpisah**; setiap parameter punya m dan v sendiri yang diawali 0:

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\,g_k \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)\,g_k^2$$

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad
\hat{v}_k = \frac{v_k}{1-\beta_2^k} \qquad
\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}$$

```
Contoh 1 — W_c  (g₁ = -0.218504, nilai awal W_c = 0.7)
  m₁  = 0.9 × 0 + 0.1 × (-0.218504) = -0.0218504
  v₁  = 0.999 × 0 + 0.001 × (-0.218504)² = 0.0000477439
  m̂₁  = -0.0218504 / (1 - 0.9¹) = -0.0218504 / 0.1 = -0.218504
  v̂₁  = 0.0000477439 / (1 - 0.999¹) = 0.0000477439 / 0.001 = 0.0477439
  √v̂₁ = √0.0477439 = 0.218504
  Δ   = η × m̂₁ / (√v̂₁ + ε) = 0.001 × (-0.218504) / (0.218504 + 0.0000001) = -0.0010000
  W_c baru = W_c lama - Δ = 0.7 - (-0.0010000) = 0.701000

Contoh 2 — b_y  (g₁ = -1.051530, nilai awal b_y = 0)
  m₁  = 0.9 × 0 + 0.1 × (-1.051530) = -0.1051530
  v₁  = 0.999 × 0 + 0.001 × (-1.051530)² = 0.0011057147
  m̂₁  = -0.1051530 / (1 - 0.9¹) = -0.1051530 / 0.1 = -1.051530
  v̂₁  = 0.0011057147 / (1 - 0.999¹) = 0.0011057147 / 0.001 = 1.1057147
  √v̂₁ = √1.1057147 = 1.051530
  Δ   = η × m̂₁ / (√v̂₁ + ε) = 0.001 × (-1.051530) / (1.051530 + 0.0000001) = -0.0010000
  b_y baru = b_y lama - Δ = 0 - (-0.0010000) = 0.001000
```

**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias
membuat m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap
parameter bergeser sebesar 0.001** (= η) berlawanan arah dengan tanda
gradiennya, berapa pun besar gradiennya. Selisih kecil pada digit ke-7 (misalnya
pada U_f) berasal dari ε yang ditambahkan ke penyebut. Semua gradien negatif, sehingga **semua parameter naik 0.001**.

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding,
SGD biasa (Δ = η × g) akan menggeser b_y sebesar
0.0010515 tetapi U_f hanya 0.0000014,
yaitu 766 kali lebih kecil. Adam menyamakan
kecepatan belajar semua parameter.

| Parameter | Nilai lama | Gradien g₁ | m₁ | v₁ | Δ = η·m̂₁/(√v̂₁+ε) | Nilai baru |
|---|---|---|---|---|---|---|
| W_f | 0.500000 | -0.007846 | -0.0007846 | 0.0000000616 | -0.0010000 | 0.501000 |
| U_f | 0.400000 | -0.001372 | -0.0001372 | 0.0000000019 | -0.0009999 | 0.401000 |
| b_f | 1.000000 | -0.013077 | -0.0013077 | 0.0000001710 | -0.0010000 | 1.001000 |
| W_i | 0.600000 | -0.040209 | -0.0040209 | 0.0000016168 | -0.0010000 | 0.601000 |
| U_i | 0.300000 | -0.004313 | -0.0004313 | 0.0000000186 | -0.0010000 | 0.301000 |
| b_i | 0.000000 | -0.072199 | -0.0072199 | 0.0000052127 | -0.0010000 | 0.001000 |
| W_c | 0.700000 | -0.218504 | -0.0218504 | 0.0000477439 | -0.0010000 | 0.701000 |
| U_c | 0.200000 | -0.021365 | -0.0021365 | 0.0000004565 | -0.0010000 | 0.201000 |
| b_c | 0.000000 | -0.396290 | -0.0396290 | 0.0001570457 | -0.0010000 | 0.001000 |
| W_o | 0.400000 | -0.049284 | -0.0049284 | 0.0000024290 | -0.0010000 | 0.401000 |
| U_o | 0.500000 | -0.008217 | -0.0008217 | 0.0000000675 | -0.0010000 | 0.501000 |
| b_o | 0.000000 | -0.082908 | -0.0082908 | 0.0000068737 | -0.0010000 | 0.001000 |
| W_y | 0.800000 | -0.229017 | -0.0229017 | 0.0000524487 | -0.0010000 | 0.801000 |
| b_y | 0.000000 | -1.051530 | -0.1051530 | 0.0011057147 | -0.0010000 | 0.001000 |

### 3.11 Forward Ulang dengan Bobot Baru

```
Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:
  ŷ' = 0.176325     (sebelumnya 0.174235)
  L  = (0.7 - 0.176325)² = 0.274235     (sebelumnya 0.276429)
  Loss turun 0.002193 hanya dengan satu kali update.
```

### 3.12 Iterasi k = 2: Momentum Mulai Bekerja

Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT
menghasilkan gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol,
sehingga langkah Adam merupakan **campuran** gradien sekarang dan gradien
sebelumnya. Berikut perhitungan Adam untuk dua parameter yang sama:

```
Contoh 1 — W_c  (g₂ = -0.217767, nilai sekarang = 0.701000, m₁ = -0.0218504, v₁ = 0.0000477439)
  m₂  = 0.9 × m₁ + 0.1 × g₂ = 0.9 × (-0.0218504) + 0.1 × (-0.217767) = (-0.0196653) + (-0.0217767) = -0.0414420
  v₂  = 0.999 × v₁ + 0.001 × g₂² = 0.999 × 0.0000477439 + 0.001 × (-0.217767)² = 0.0000476962 + 0.0000474223 = 0.0000951184
  m̂₂  = -0.0414420 / (1 - 0.9²) = -0.0414420 / 0.19 = -0.218116
  v̂₂  = 0.0000951184 / (1 - 0.999²) = 0.0000951184 / 0.001999 = 0.0475830
  √v̂₂ = 0.218135
  Δ   = 0.001 × (-0.218116) / (0.218135 + 0.0000001) = -0.0009999
  W_c baru = 0.701000 - (-0.0009999) = 0.702000

Contoh 2 — b_y  (g₂ = -1.047350, nilai sekarang = 0.001000, m₁ = -0.1051530, v₁ = 0.0011057147)
  m₂  = 0.9 × m₁ + 0.1 × g₂ = 0.9 × (-0.1051530) + 0.1 × (-1.047350) = (-0.0946377) + (-0.1047350) = -0.1993726
  v₂  = 0.999 × v₁ + 0.001 × g₂² = 0.999 × 0.0011057147 + 0.001 × (-1.047350)² = 0.0011046090 + 0.0010969415 = 0.0022015505
  m̂₂  = -0.1993726 / (1 - 0.9²) = -0.1993726 / 0.19 = -1.049330
  v̂₂  = 0.0022015505 / (1 - 0.999²) = 0.0022015505 / 0.001999 = 1.1013259
  √v̂₂ = 1.049441
  Δ   = 0.001 × (-1.049330) / (1.049441 + 0.0000001) = -0.0009999
  b_y baru = 0.001000 - (-0.0009999) = 0.002000
```

m₂ menggabungkan arah gradien iterasi 1 dan 2. Jika suatu saat gradien berbalik arah
(misalnya ketika bobot melewati titik minimum), m akan mengecil dan langkahnya
otomatis melambat. Koreksi bias (1 − βᵏ) makin lama makin mendekati 1, sehingga
pengaruhnya hilang setelah banyak iterasi; tabelnya ada di [Bagian 2, subbagian 8](tahap_7_8_2_fungsi_loss_dan_adam.md#8-koreksi-bias-seiring-iterasi).

### 3.13 Siklus Diulang: Perjalanan Loss

Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah
update. Bobot berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom
b_y dan b_f memperlihatkan bias ikut dipelajari seperti bobot.

| Setelah k update | Loss L | Prediksi ŷ′ | b_y | b_f |
|---|---|---|---|---|
| 0 | 0.276429 | 0.174235 | 0.000000 | 1.000000 |
| 1 | 0.274235 | 0.176325 | 0.001000 | 1.001000 |
| 2 | 0.272048 | 0.178418 | 0.002000 | 1.002000 |
| 3 | 0.269866 | 0.180513 | 0.003000 | 1.003000 |
| 5 | 0.265522 | 0.184712 | 0.004998 | 1.005000 |
| 10 | 0.254774 | 0.195249 | 0.009987 | 1.010000 |
| 20 | 0.233823 | 0.216448 | 0.019900 | 1.019995 |
| 50 | 0.176240 | 0.280190 | 0.048663 | 1.049741 |
| 100 | 0.100633 | 0.382773 | 0.091802 | 1.096754 |
| 200 | 0.022099 | 0.551344 | 0.155760 | 1.170481 |
| 300 | 0.002626 | 0.648754 | 0.189478 | 1.210478 |
| 400 | 0.000164 | 0.687176 | 0.202179 | 1.225637 |
| 500 | 0.0000055602 | 0.697642 | 0.205581 | 1.229700 |

Nilai seluruh parameter sebelum dan sesudah 500 iterasi:

| Parameter | Awal | Setelah 500 iterasi | Perubahan |
|---|---|---|---|
| W_f | 0.5 | 0.729698 | +0.229698 |
| U_f | 0.4 | 0.679413 | +0.279413 |
| b_f | 1 | 1.229700 | +0.229700 |
| W_i | 0.6 | 0.838608 | +0.238608 |
| U_i | 0.3 | 0.579916 | +0.279916 |
| b_i | 0 | 0.239535 | +0.239535 |
| W_c | 0.7 | 0.902863 | +0.202863 |
| U_c | 0.2 | 0.438643 | +0.238643 |
| b_c | 0 | 0.203810 | +0.203810 |
| W_o | 0.4 | 0.652157 | +0.252157 |
| U_o | 0.5 | 0.800588 | +0.300588 |
| b_o | 0 | 0.252786 | +0.252786 |
| W_y | 0.8 | 1.052395 | +0.252395 |
| b_y | 0 | 0.205581 | +0.205581 |

## 4. GRU: Satu Siklus Pelatihan Langkah demi Langkah

Urutan langkahnya sama persis dengan LSTM. Yang berbeda hanya rumus di dalam
sel: GRU tidak punya cell state, punya tiga himpunan bobot (z, r, kandidat), dan
setiap gerbang punya **dua bias** (bias masukan dan bias rekuren) karena Keras
memakai `reset_after=True` (lihat [Bagian 4, subbagian 4](tahap_7_8_4_bias_lstm_gru.md#4-dua-bias-pada-gru-asal-dan-buktinya)).

### 4.1 Bobot Awal dan Jumlah Parameter

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b(in) (bias masukan) | b(rec) (bias rekuren) |
|---|---|---|---|---|
| update (z) | 0.5 | 0.4 | 0 | 0 |
| reset (r) | 0.6 | 0.3 | 0 | 0 |
| kandidat (h̃) | 0.7 | 0.2 | 0 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 | - |

```
Jumlah parameter (persamaan 27) dengan n_u = 1 dan n_f = 1:
  P_GRU = 3 × [n_u × (n_u + n_f) + 2 × n_u] + (n_u + 1)
        = 3 × [1 × (1 + 1) + 2] + (1 + 1)
        = 3 × 4 + 2
        = 14 parameter  (12 di lapisan GRU + 2 di lapisan dense)
  Suku 2 × n_u = dua bias per gerbang (bias masukan dan bias rekuren).
```

### 4.2 Langkah 1: Forward Pass

Alur Gambar 7 langkah demi langkah dijelaskan di [Bagian 1, subbagian 9](tahap_7_8_1_alur_sel_lstm_gru.md#9-gambar-7-struktur-sel-gru); di sini dihitung angkanya untuk kedua time step.

Rumus versi 1 neuron (persamaan 23-26). Untuk kandidat, bagian rekurennya
diberi nama $q_t$ supaya terlihat jelas apa yang dikalikan reset gate:

$$z_t = \sigma\bigl(W_z x_t + U_z h_{t-1} + b_z^{(in)} + b_z^{(rec)}\bigr) \qquad
r_t = \sigma\bigl(W_r x_t + U_r h_{t-1} + b_r^{(in)} + b_r^{(rec)}\bigr)$$

$$q_t = U_h h_{t-1} + b_h^{(rec)} \qquad
\tilde{h}_t = \tanh\bigl(W_h x_t + b_h^{(in)} + r_t\,q_t\bigr) \qquad
h_t = z_t\,h_{t-1} + (1-z_t)\,\tilde{h}_t$$

Catatan konvensi: pada Keras (dan pada persamaan 7 makalah Cho et al., 2014),
**zₜ mengalikan memori lama hₜ₋₁**. Bentuk dengan peran z tertukar,
hₜ = (1 − zₜ)hₜ₋₁ + zₜh̃ₜ, berasal dari Chung et al. (2014).

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
Hidden state baru (persamaan 26, Gambar 7: lingkaran × , 1−, dan +)
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
Hidden state baru (persamaan 26, Gambar 7: lingkaran × , 1−, dan +)
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

### 4.3 Langkah 2: Loss

```
  L = (y' - ŷ')² = (0.7 - 0.204811)² = 0.495189² = 0.245212
```

### 4.4 Langkah 3a: Backward di Lapisan Dense

Rumusnya identik dengan 3.4; hanya nilai h₂ dan ŷ′ yang berbeda.

```
  ∂L/∂ŷ'  = -2(y' - ŷ') = -2 × (0.7 - 0.204811) = -0.990377
  ∂L/∂W_y = ∂L/∂ŷ' × h₂ = (-0.990377) × 0.256014 = -0.253551
  ∂L/∂b_y = ∂L/∂ŷ' × 1  = -0.990377
  ∂L/∂h₂  = ∂L/∂ŷ' × W_y = (-0.990377) × 0.8 = -0.792302   (masuk ke sel GRU)
```

### 4.5 Langkah 3b: Backward di Dalam Sel, t = 2

Rumus forward yang diturunkan: `h₂ = z₂ × h₁ + (1 − z₂) × h̃₂` (menuju update
gate dan kandidat), lalu `h̃₂ = tanh(W_h x₂ + b_h(in) + r₂ × q₂)` (menuju reset
gate dan bagian rekuren q₂).

```
Masukan dari 4.4: ∂L/∂h₂ = -0.792302

(a) Update gate — turunkan h₂ terhadap z₂: ∂h₂/∂z₂ = h₁ - h̃₂
    ∂L/∂z₂ = ∂L/∂h₂ × (h₁ - h̃₂) = (-0.792302) × (0.147273 - 0.411706) = (-0.792302) × (-0.264433) = 0.209511
    δz₂    = ∂L/∂z₂ × z₂(1 - z₂) = 0.209511 × 0.242119 = 0.050726

(b) Kandidat — turunkan h₂ terhadap h̃₂: ∂h₂/∂h̃₂ = 1 - z₂
    ∂L/∂h̃₂ = ∂L/∂h₂ × (1 - z₂) = (-0.792302) × 0.411224 = -0.325813
    δh̃₂    = ∂L/∂h̃₂ × (1 - h̃₂²) = (-0.325813) × 0.830498 = -0.270587

(c) Reset gate — dalam pra-aktivasi kandidat, r₂ dikalikan q₂
    ∂L/∂r₂ = δh̃₂ × q₂ = (-0.270587) × 0.029455 = -0.007970
    δr₂    = ∂L/∂r₂ × r₂(1 - r₂) = (-0.007970) × 0.240062 = -0.001913

(d) Bagian rekuren kandidat q₂ = U_h × h₁ + b_h(rec)
    ∂L/∂q₂ = δh̃₂ × r₂ = (-0.270587) × 0.599692 = -0.162269
```

**Makna tanda gradien z.** ∂L/∂z₂ bernilai **positif** karena kandidat baru
(h̃₂ = 0.411706) lebih besar daripada memori lama (h₁ = 0.147273),
sedangkan prediksi masih terlalu rendah. Adam nanti akan **menurunkan** z,
artinya GRU belajar untuk lebih mempercayai informasi baru.

### 4.6 Langkah 3c: Mengirim Kesalahan ke t = 1

h₁ dipakai di empat tempat pada t = 2: langsung di `z₂ × h₁`, di update gate
(lewat U_z), di reset gate (lewat U_r), dan di bagian rekuren q₂ (lewat U_h).
Keempat kontribusinya dijumlahkan.

```
  ∂L/∂h₁ = ∂L/∂h₂ × z₂ + U_z × δz₂ + U_r × δr₂ + U_h × ∂L/∂q₂
         = (-0.792302) × 0.588776 + 0.4 × 0.050726 + 0.3 × (-0.001913) + 0.2 × (-0.162269)
         = (-0.466489) + 0.020291 + (-0.000574) + (-0.032454)
         = -0.479226
  Porsi jalur langsung z₂ × h₁ = -0.466489 / -0.479226 = 97.3%
```

Sekitar **97.3%** sinyal mengalir lewat jalur langsung `zₜ × hₜ₋₁`.
Jalur ini adalah "jalan tol gradien" pada GRU, padanan cell state pada LSTM.

### 4.7 Langkah 3d: Backward di Dalam Sel, t = 1

```
Masukan dari 4.6: ∂L/∂h₁ = -0.479226

(a) Update gate
    ∂L/∂z₁ = ∂L/∂h₁ × (h₀ - h̃₁) = (-0.479226) × (0.000000 - 0.336376) = 0.161200
    δz₁    = 0.161200 × 0.246134 = 0.039677

(b) Kandidat
    ∂L/∂h̃₁ = ∂L/∂h₁ × (1 - z₁) = (-0.479226) × 0.437823 = -0.209816
    δh̃₁    = (-0.209816) × 0.886851 = -0.186076

(c) Reset gate
    ∂L/∂r₁ = δh̃₁ × q₁ = (-0.186076) × 0.000000 = 0.000000
    δr₁    = 0.000000     (nol karena q₁ = 0: belum ada memori untuk di-reset)

(d) Bagian rekuren q₁
    ∂L/∂q₁ = δh̃₁ × r₁ = (-0.186076) × 0.574443 = -0.106890
```

### 4.8 Langkah 3e: Gradien Total Setiap Bobot dan Bias

Sama seperti LSTM, kontribusi kedua time step dijumlahkan. Untuk kandidat,
bobot $U_h$ dan bias $b_h^{(rec)}$ berada di dalam $q_t$, sehingga gradiennya
memakai $\partial L/\partial q_t$; bobot $W_h$ dan bias $b_h^{(in)}$ memakai
$\delta\tilde{h}_t$.

```
Update gate:  δz₁ = 0.039677,  δz₂ = 0.050726
  ∂L/∂W_z = δz₁ × x₁ + δz₂ × x₂ = 0.039677 × 0.5 + 0.050726 × 0.6 = 0.019838 + 0.030436 = 0.050274
  ∂L/∂U_z = δz₁ × h₀ + δz₂ × h₁ = 0.039677 × 0.000000 + 0.050726 × 0.147273 = 0.007471
  ∂L/∂b_z(in)  = δz₁ + δz₂ = 0.039677 + 0.050726 = 0.090403
  ∂L/∂b_z(rec) = δz₁ + δz₂ = 0.090403   (sama persis dengan bias masukan)

Reset gate:  δr₁ = 0.000000,  δr₂ = -0.001913
  ∂L/∂W_r = δr₁ × x₁ + δr₂ × x₂ = 0.000000 × 0.5 + (-0.001913) × 0.6 = 0.000000 + (-0.001148) = -0.001148
  ∂L/∂U_r = δr₁ × h₀ + δr₂ × h₁ = 0.000000 × 0.000000 + (-0.001913) × 0.147273 = -0.000282
  ∂L/∂b_r(in)  = δr₁ + δr₂ = 0.000000 + (-0.001913) = -0.001913
  ∂L/∂b_r(rec) = δr₁ + δr₂ = -0.001913   (sama persis dengan bias masukan)

Kandidat:  δh̃₁ = -0.186076,  δh̃₂ = -0.270587,  ∂L/∂q₁ = -0.106890,  ∂L/∂q₂ = -0.162269
  ∂L/∂W_h      = δh̃₁ × x₁ + δh̃₂ × x₂ = (-0.186076) × 0.5 + (-0.270587) × 0.6 = (-0.093038) + (-0.162352) = -0.255390
  ∂L/∂b_h(in)  = δh̃₁ + δh̃₂ = (-0.186076) + (-0.270587) = -0.456663
  ∂L/∂U_h      = ∂L/∂q₁ × h₀ + ∂L/∂q₂ × h₁ = (-0.106890) × 0.000000 + (-0.162269) × 0.147273 = -0.023898
  ∂L/∂b_h(rec) = ∂L/∂q₁ + ∂L/∂q₂ = (-0.106890) + (-0.162269) = -0.269159
  (∂L/∂q = δh̃ × r, jadi bias rekuren kandidat = Σ δh̃ × r, BERBEDA dari bias masukan)

Lapisan dense (dari 4.4):
  ∂L/∂W_y = -0.253551
  ∂L/∂b_y = -0.990377
```

### 4.9 Pemeriksaan Gradien dengan Turunan Numerik

| Parameter | Gradien BPTT (manual) | Gradien numerik | Selisih |
|---|---|---|---|
| W_z | 0.05027427 | 0.05027427 | 4.7e-13 |
| U_z | 0.00747064 | 0.00747064 | 2.1e-11 |
| b_z(in) | 0.09040324 | 0.09040324 | 5.7e-12 |
| b_z(rec) | 0.09040324 | 0.09040324 | 5.7e-12 |
| W_r | -0.00114798 | -0.00114798 | 2.3e-11 |
| U_r | -0.00028178 | -0.00028178 | 1.1e-11 |
| b_r(in) | -0.00191330 | -0.00191330 | 1.0e-11 |
| b_r(rec) | -0.00191330 | -0.00191330 | 1.0e-11 |
| W_h | -0.25539034 | -0.25539034 | 2.2e-11 |
| U_h | -0.02389787 | -0.02389787 | 4.7e-12 |
| b_h(in) | -0.45666322 | -0.45666322 | 1.2e-11 |
| b_h(rec) | -0.26915896 | -0.26915896 | 1.8e-11 |
| W_y | -0.25355063 | -0.25355063 | 3.5e-11 |
| b_y | -0.99037732 | -0.99037732 | 2.3e-11 |

### 4.10 Langkah 4: Update Adam, Iterasi k = 1

Rumus Adam sama dengan bagian 3.10 (persamaan 29-31). Setiap parameter GRU juga
punya m dan v sendiri yang diawali 0.

```
Contoh 1 — W_h  (g₁ = -0.255390, nilai awal W_h = 0.7)
  m₁  = 0.9 × 0 + 0.1 × (-0.255390) = -0.0255390
  v₁  = 0.999 × 0 + 0.001 × (-0.255390)² = 0.0000652242
  m̂₁  = -0.0255390 / (1 - 0.9¹) = -0.0255390 / 0.1 = -0.255390
  v̂₁  = 0.0000652242 / (1 - 0.999¹) = 0.0000652242 / 0.001 = 0.0652242
  √v̂₁ = √0.0652242 = 0.255390
  Δ   = η × m̂₁ / (√v̂₁ + ε) = 0.001 × (-0.255390) / (0.255390 + 0.0000001) = -0.0010000
  W_h baru = W_h lama - Δ = 0.7 - (-0.0010000) = 0.701000

Contoh 2 — b_y  (g₁ = -0.990377, nilai awal b_y = 0)
  m₁  = 0.9 × 0 + 0.1 × (-0.990377) = -0.0990377
  v₁  = 0.999 × 0 + 0.001 × (-0.990377)² = 0.0009808472
  m̂₁  = -0.0990377 / (1 - 0.9¹) = -0.0990377 / 0.1 = -0.990377
  v̂₁  = 0.0009808472 / (1 - 0.999¹) = 0.0009808472 / 0.001 = 0.9808472
  √v̂₁ = √0.9808472 = 0.990377
  Δ   = η × m̂₁ / (√v̂₁ + ε) = 0.001 × (-0.990377) / (0.990377 + 0.0000001) = -0.0010000
  b_y baru = b_y lama - Δ = 0 - (-0.0010000) = 0.001000
```

**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias
membuat m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap
parameter bergeser sebesar 0.001** (= η) berlawanan arah dengan tanda
gradiennya, berapa pun besar gradiennya. Selisih kecil pada digit ke-7 (misalnya
pada U_r) berasal dari ε yang ditambahkan ke penyebut. Parameter dengan gradien **positif** (W_z, U_z, b_z(in), b_z(rec)) **turun** 0.001, sedangkan parameter lain (gradien negatif) naik 0.001.

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding,
SGD biasa (Δ = η × g) akan menggeser b_y sebesar
0.0009904 tetapi U_r hanya 0.0000003,
yaitu 3,515 kali lebih kecil. Adam menyamakan
kecepatan belajar semua parameter.

| Parameter | Nilai lama | Gradien g₁ | m₁ | v₁ | Δ = η·m̂₁/(√v̂₁+ε) | Nilai baru |
|---|---|---|---|---|---|---|
| W_z | 0.500000 | 0.050274 | 0.0050274 | 0.0000025275 | 0.0010000 | 0.499000 |
| U_z | 0.400000 | 0.007471 | 0.0007471 | 0.0000000558 | 0.0010000 | 0.399000 |
| b_z(in) | 0.000000 | 0.090403 | 0.0090403 | 0.0000081727 | 0.0010000 | -0.001000 |
| b_z(rec) | 0.000000 | 0.090403 | 0.0090403 | 0.0000081727 | 0.0010000 | -0.001000 |
| W_r | 0.600000 | -0.001148 | -0.0001148 | 0.0000000013 | -0.0009999 | 0.601000 |
| U_r | 0.300000 | -0.000282 | -0.0000282 | 0.0000000001 | -0.0009996 | 0.301000 |
| b_r(in) | 0.000000 | -0.001913 | -0.0001913 | 0.0000000037 | -0.0009999 | 0.001000 |
| b_r(rec) | 0.000000 | -0.001913 | -0.0001913 | 0.0000000037 | -0.0009999 | 0.001000 |
| W_h | 0.700000 | -0.255390 | -0.0255390 | 0.0000652242 | -0.0010000 | 0.701000 |
| U_h | 0.200000 | -0.023898 | -0.0023898 | 0.0000005711 | -0.0010000 | 0.201000 |
| b_h(in) | 0.000000 | -0.456663 | -0.0456663 | 0.0002085413 | -0.0010000 | 0.001000 |
| b_h(rec) | 0.000000 | -0.269159 | -0.0269159 | 0.0000724465 | -0.0010000 | 0.001000 |
| W_y | 0.800000 | -0.253551 | -0.0253551 | 0.0000642879 | -0.0010000 | 0.801000 |
| b_y | 0.000000 | -0.990377 | -0.0990377 | 0.0009808472 | -0.0010000 | 0.001000 |

### 4.11 Forward Ulang dengan Bobot Baru

```
Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:
  ŷ' = 0.207331     (sebelumnya 0.204811)
  L  = (0.7 - 0.207331)² = 0.242723     (sebelumnya 0.245212)
  Loss turun 0.002489 hanya dengan satu kali update.
```

### 4.12 Iterasi k = 2: Momentum Mulai Bekerja

Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT
menghasilkan gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol,
sehingga langkah Adam merupakan **campuran** gradien sekarang dan gradien
sebelumnya. Berikut perhitungan Adam untuk dua parameter yang sama:

```
Contoh 1 — W_h  (g₂ = -0.254254, nilai sekarang = 0.701000, m₁ = -0.0255390, v₁ = 0.0000652242)
  m₂  = 0.9 × m₁ + 0.1 × g₂ = 0.9 × (-0.0255390) + 0.1 × (-0.254254) = (-0.0229851) + (-0.0254254) = -0.0484105
  v₂  = 0.999 × v₁ + 0.001 × g₂² = 0.999 × 0.0000652242 + 0.001 × (-0.254254)² = 0.0000651590 + 0.0000646452 = 0.0001298042
  m̂₂  = -0.0484105 / (1 - 0.9²) = -0.0484105 / 0.19 = -0.254792
  v̂₂  = 0.0001298042 / (1 - 0.999²) = 0.0001298042 / 0.001999 = 0.0649346
  √v̂₂ = 0.254823
  Δ   = 0.001 × (-0.254792) / (0.254823 + 0.0000001) = -0.0009999
  W_h baru = 0.701000 - (-0.0009999) = 0.702000

Contoh 2 — b_y  (g₂ = -0.985339, nilai sekarang = 0.001000, m₁ = -0.0990377, v₁ = 0.0009808472)
  m₂  = 0.9 × m₁ + 0.1 × g₂ = 0.9 × (-0.0990377) + 0.1 × (-0.985339) = (-0.0891340) + (-0.0985339) = -0.1876678
  v₂  = 0.999 × v₁ + 0.001 × g₂² = 0.999 × 0.0009808472 + 0.001 × (-0.985339)² = 0.0009798664 + 0.0009708922 = 0.0019507586
  m̂₂  = -0.1876678 / (1 - 0.9²) = -0.1876678 / 0.19 = -0.987725
  v̂₂  = 0.0019507586 / (1 - 0.999²) = 0.0019507586 / 0.001999 = 0.9758672
  √v̂₂ = 0.987860
  Δ   = 0.001 × (-0.987725) / (0.987860 + 0.0000001) = -0.0009999
  b_y baru = 0.001000 - (-0.0009999) = 0.002000
```

Penjelasan momentum dan koreksi bias pada 3.12 berlaku sama untuk GRU.

### 4.13 Siklus Diulang: Perjalanan Loss

Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah
update. Bobot berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom
b_y dan b_z(in) memperlihatkan bias ikut dipelajari seperti bobot.

| Setelah k update | Loss L | Prediksi ŷ′ | b_y | b_z(in) |
|---|---|---|---|---|
| 0 | 0.245212 | 0.204811 | 0.000000 | 0.000000 |
| 1 | 0.242723 | 0.207331 | 0.001000 | -0.001000 |
| 2 | 0.240243 | 0.209854 | 0.002000 | -0.002000 |
| 3 | 0.237771 | 0.212382 | 0.002999 | -0.003000 |
| 5 | 0.232855 | 0.217450 | 0.004998 | -0.005000 |
| 10 | 0.220730 | 0.230181 | 0.009983 | -0.010002 |
| 20 | 0.197282 | 0.255835 | 0.019870 | -0.020007 |
| 50 | 0.134741 | 0.332930 | 0.048231 | -0.049824 |
| 100 | 0.060448 | 0.454138 | 0.088891 | -0.096424 |
| 200 | 0.005589 | 0.625239 | 0.139578 | -0.161695 |
| 300 | 0.000159 | 0.687388 | 0.156372 | -0.185319 |
| 400 | 0.0000013989 | 0.698817 | 0.159373 | -0.189662 |
| 500 | 0.0000000036 | 0.699940 | 0.159666 | -0.190088 |

Nilai seluruh parameter sebelum dan sesudah 500 iterasi:

| Parameter | Awal | Setelah 500 iterasi | Perubahan |
|---|---|---|---|
| W_z | 0.5 | 0.310097 | -0.189903 |
| U_z | 0.4 | 0.170506 | -0.229494 |
| b_z(in) | 0 | -0.190088 | -0.190088 |
| b_z(rec) | 0 | -0.190088 | -0.190088 |
| W_r | 0.6 | 0.848098 | +0.248098 |
| U_r | 0.3 | 0.577489 | +0.277489 |
| b_r(in) | 0 | 0.249159 | +0.249159 |
| b_r(rec) | 0 | 0.249159 | +0.249159 |
| W_h | 0.7 | 0.858738 | +0.158738 |
| U_h | 0.2 | 0.407983 | +0.207983 |
| b_h(in) | 0 | 0.158837 | +0.158837 |
| b_h(rec) | 0 | 0.168930 | +0.168930 |
| W_y | 0.8 | 0.994445 | +0.194445 |
| b_y | 0 | 0.159666 | +0.159666 |

Perhatikan pasangan bias pada tabel terakhir: **b_z(in) = b_z(rec) = -0.190088**
dan **b_r(in) = b_r(rec) = 0.249159** tetap kembar sampai akhir,
sedangkan **b_h(in) = 0.158837** dan **b_h(rec) = 0.168930**
berbeda. [Bagian 4, subbagian 4](tahap_7_8_4_bias_lstm_gru.md#4-dua-bias-pada-gru-asal-dan-buktinya) menjelaskan sebabnya.

## 5. Dari Model Mini ke Model Penelitian

| Aspek | Model mini | Model penelitian |
|---|---|---|
| Masukan per time step | 1 angka | vektor 10 fitur |
| Time step (BPTT mundur sejauh) | 2 | 7 |
| Bobot per gerbang | angka tunggal | matriks W (10 × n_u) dan U (n_u × n_u), vektor b (n_u) |
| Jumlah parameter | 14 | LSTM 8,201, GRU 3,811 |
| Loss per iterasi | 1 sampel | rata-rata 32 sampel (1 batch) |
| Jumlah iterasi | 500 | 26 per epoch × 500 epoch = 13,000 |

Yang sama: urutan forward → loss → BPTT → Adam, rumus setiap langkah, dan
pengaturan Adam. Setiap parameter (termasuk setiap elemen matriks) punya gradien,
m, dan v sendiri.

### 5.1 Jika Memakai Batch (N > 1)

Persamaan (28) memakai rata-rata $\mathcal{L} = \frac{1}{N}\sum_k (y'_k - \hat{y}'_k)^2$.
Setiap $\hat{y}'_k$ hanya muncul di suku ke-$k$, sehingga:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}'_k} = -\frac{2}{N}\,(y'_k - \hat{y}'_k) \qquad
\frac{\partial \mathcal{L}}{\partial W_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}\,h_{T,k} \qquad
\frac{\partial \mathcal{L}}{\partial b_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}$$

Setiap sampel menerima sinyal $\partial\mathcal{L}/\partial h_{T,k} =
\partial\mathcal{L}/\partial\hat{y}'_k \cdot W_y$ yang menjalankan BPTT-nya
sendiri, lalu gradien semua sampel dijumlahkan sebelum Adam dipanggil sekali.

```
Contoh N = 2 (sampel 1 = simulasi LSTM di atas; sampel 2 = ilustrasi h_T = 0.30, y' = 0.65)
  ŷ'₁ = 0.174235,   ŷ'₂ = 0.8 × 0.30 + 0 = 0.240000
  ∂L/∂ŷ'₁ = -(2/2) × (0.7 - 0.174235) = -0.525765
  ∂L/∂ŷ'₂ = -(2/2) × (0.65 - 0.240000) = -0.410000
  ∂L/∂W_y = (-0.525765) × 0.217794 + (-0.410000) × 0.3 = (-0.114508) + (-0.123000) = -0.237508
  ∂L/∂b_y = (-0.525765) + (-0.410000) = -0.935765
```

### 5.2 Versi Vektor pada Lapisan Dense

Pada model penelitian, h_T berisi n_u elemen dan W_y juga berisi n_u elemen,
sehingga ŷ′ = Σⱼ h_T,ⱼ·W_y,ⱼ + b_y. Rumus 3.4 berlaku untuk setiap elemen j:
∂L/∂W_y,ⱼ = Σₖ ∂L/∂ŷ′ₖ·h_T,ₖ,ⱼ. Hasilnya n_u gradien bobot + 1 gradien bias,
yaitu **41 parameter dense pada LSTM** dan **31 pada GRU** (suku
(n_u + 1) pada persamaan 22 dan 27). Sinyal yang masuk ke neuron j adalah
∂L/∂h_T,ⱼ = ∂L/∂ŷ′·W_y,ⱼ.

Hubungan loss MSE dengan RMSE dalam USD dibahas di [Bagian 2, subbagian 5](tahap_7_8_2_fungsi_loss_dan_adam.md#5-hubungan-mse-dengan-rmse-dalam-usd).

### 5.3 Batasan Simulasi Ini

1. Loss simulasi turun hampir ke nol karena hanya ada **1 sampel**, sehingga model
   bisa "menghafal" targetnya. Pada data penelitian dengan 804
   sampel, loss validasi LSTM terbaik berhenti di sekitar 0.00065
   karena model harus menemukan pola umum, bukan menghafal.
2. Angka akhir simulasi **tidak dapat dipakai untuk membandingkan** LSTM dan GRU.
   Bobot awalnya dipilih bulat agar mudah dihitung, sedangkan Keras memakai bobot
   acak. Perbandingan yang sah tetap hasil tahap 9-12.

## 6. Ringkasan Alur dan Hasil Verifikasi

```
SATU SIKLUS PELATIHAN (berlaku untuk LSTM dan GRU)

 1. Inisialisasi : bobot acak, bias 0 (forget LSTM = 1), Adam m = v = 0
 2. Forward      : x₁ → gerbang → h₁ (dan c₁) → x₂ → ... → h_T → ŷ' = W_y h_T + b_y
 3. Loss         : L = (y' - ŷ')²  (rata-rata satu batch)
 4. Dense        : ∂L/∂ŷ' = -2(y' - ŷ') → ∂L/∂W_y, ∂L/∂b_y, ∂L/∂h_T
 5. Sel, t = T   : ∂L/∂h_T → turunan setiap gerbang → δ setiap gerbang
 6. Through time : δ dikirim ke t-1 lewat U (dan lewat cell state / z × h_(t-1))
 7. Ulangi 5-6 sampai t = 1
 8. Gradien      : ∂L/∂W = Σ δx,  ∂L/∂U = Σ δh_(t-1),  ∂L/∂b = Σ δ
 9. Adam         : m, v → m̂, v̂ → θ baru = θ - η m̂/(√v̂ + ε)
10. Kembali ke langkah 2 dengan batch berikutnya

HASIL SIMULASI
  LSTM: loss 0.276429 → 0.274235 (k = 1) → 0.0000055602 (k = 500), ŷ' 0.174235 → 0.697642
  GRU : loss 0.245212 → 0.242723 (k = 1) → 0.0000000036 (k = 500), ŷ' 0.204811 → 0.699940
```

**Pemeriksaan otomatis yang lolos saat seri ini dibuat:**

- Gradien BPTT manual = gradien numerik untuk 14 parameter LSTM (selisih maksimum 9.3e-11) dan 14 parameter GRU (selisih maksimum 3.5e-11).
- Uji geser pada lapisan dense: perubahan loss sebenarnya = gradien × 0.001 (toleransi 2e-6).
- Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter, dan langkah Adam tidak berubah bila gradien dikali 100 (Bagian 2).
- Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah 500 iterasi, untuk LSTM maupun GRU.
- Bias z dan r GRU tetap kembar (bias masukan = bias rekuren), bias kandidat berbeda, baik pada simulasi maupun model terlatih (Bagian 4).
- Rentang harga × √(loss validasi) = RMSE validasi pada tabel tuning (toleransi 0.1%) untuk LSTM dan GRU (Bagian 2).
- Jumlah parameter lapisan rekuren dari rumus (persamaan 22 dan 27) sama dengan tabel tuning (Bagian 1).

---

← Sebelumnya: [Bagian 2. Fungsi Loss dan Adam](tahap_7_8_2_fungsi_loss_dan_adam.md)

→ Berikutnya: [Bagian 4. Bias pada LSTM dan GRU](tahap_7_8_4_bias_lstm_gru.md)
