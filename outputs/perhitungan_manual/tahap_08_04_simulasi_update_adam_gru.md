# Perhitungan Manual - Tahap 8.4: Simulasi Pelatihan GRU, Langkah 4: Update Adam

Subbagian ini melanjutkan [Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md). Gradien setiap bobot dan bias sudah
diketahui; sekarang **Adam** memakai gradien itu untuk menggeser setiap parameter. Setelah
itu forward pass diulang untuk membuktikan loss turun, lalu siklusnya diulang sampai
500 iterasi.

**Daftar isi**

- [1. Rumus Adam](#1-rumus-adam)
- [2. Update Adam pada Model Mini, Iterasi k = 1](#2-update-adam-pada-model-mini-iterasi-k--1)
- [3. Forward Ulang dengan Bobot Baru](#3-forward-ulang-dengan-bobot-baru)
- [4. Iterasi k = 2: Momentum Mulai Bekerja](#4-iterasi-k--2-momentum-mulai-bekerja)
- [5. Siklus Diulang: Perjalanan Loss](#5-siklus-diulang-perjalanan-loss)
- [6. Dari Model Mini ke Model Penelitian](#6-dari-model-mini-ke-model-penelitian)
- [7. Ringkasan Siklus dan Hasil Verifikasi](#7-ringkasan-siklus-dan-hasil-verifikasi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Rumus Adam

Rumus Adam untuk GRU **sama persis** dengan LSTM (persamaan 29-31, η = 0.001, β₁ = 0.9,
β₂ = 0.999, ε = 10⁻⁷). Makna setiap langkah A-D, contoh satu parameter, dan alasan Adam
cocok untuk penelitian ini dijelaskan di [Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md). Ringkasnya:

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\,g_k \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)\,g_k^2$$

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad
\hat{v}_k = \frac{v_k}{1-\beta_2^k} \qquad
\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}$$

## 2. Update Adam pada Model Mini, Iterasi k = 1

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

**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias membuat
m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap parameter bergeser
sebesar 0.001** (= η) berlawanan arah dengan tanda gradiennya, berapa pun besar
gradiennya. Selisih kecil pada digit ke-7 (misalnya pada U_r) berasal dari ε
yang ditambahkan ke penyebut. Parameter dengan gradien **positif** (W_z, U_z, b_z(in), b_z(rec)) **turun** 0.001, sedangkan parameter lain (gradien negatif) naik 0.001.

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding, SGD biasa
(Δ = η × g) akan menggeser b_y sebesar 0.0009904 tetapi
U_r hanya 0.0000003, yaitu
3,515 kali lebih kecil. Adam menyamakan kecepatan
belajar semua parameter.

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

**Makna gerak z.** Adam **menurunkan** bobot dan bias update gate (gradiennya positif),
sehingga zₜ mengecil dan GRU mengambil lebih banyak kandidat baru h̃ₜ. Ini masuk akal:
kandidat baru lebih besar daripada memori lama, sedangkan prediksi masih terlalu rendah.

## 3. Forward Ulang dengan Bobot Baru

```
Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:
  ŷ' = 0.207331     (sebelumnya 0.204811)
  L  = (0.7 - 0.207331)² = 0.242723     (sebelumnya 0.245212)
  Loss turun 0.002489 hanya dengan satu kali update.
```

## 4. Iterasi k = 2: Momentum Mulai Bekerja

Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT menghasilkan
gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol, sehingga langkah
Adam merupakan **campuran** gradien sekarang dan gradien sebelumnya. Berikut perhitungan
Adam untuk dua parameter yang sama:

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

m₂ menggabungkan arah gradien iterasi 1 dan 2. Jika suatu saat gradien berbalik arah
(misalnya ketika bobot melewati titik minimum), m akan mengecil dan langkahnya otomatis
melambat.

Tabel koreksi bias (1 − βᵏ) di [Tahap 7.4, bagian 6](tahap_07_04_simulasi_update_adam_lstm.md#6-koreksi-bias-seiring-iterasi) berlaku sama untuk GRU.

## 5. Siklus Diulang: Perjalanan Loss

Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah update. Bobot
berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom b_y dan
b_z(in) memperlihatkan bias ikut dipelajari seperti bobot.

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

Perhatikan pasangan bias: **b_z(in) = b_z(rec) = -0.190088** dan
**b_r(in) = b_r(rec) = 0.249159** tetap kembar sampai akhir, sedangkan
**b_h(in) = 0.158837** dan **b_h(rec) = 0.168930** berbeda.
[Tahap 8.12 (Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias)](tahap_08_12_bias_gru.md) menjelaskan sebabnya.

## 6. Dari Model Mini ke Model Penelitian

| Aspek | Model mini | Model penelitian |
|---|---|---|
| Masukan per time step | 1 angka | vektor 10 fitur |
| Time step (BPTT mundur sejauh) | 2 | 7 |
| Bobot per gerbang | angka tunggal | matriks W (10 × n_u) dan U (n_u × n_u), vektor b (n_u) |
| Loss per iterasi | 1 sampel | rata-rata 32 sampel (1 batch) |
| Jumlah iterasi | 500 | 26 per epoch × epoch grid (100, 500, 1,000) |

Jumlah parameter GRU penelitian untuk setiap jumlah neuron pada grid (persamaan 27, termasuk dense):

| Neuron n_u | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| Parameter | 671 | 1,941 | 3,811 | 6,281 | 9,351 |

Yang sama: urutan forward → loss → BPTT → Adam, rumus setiap langkah, dan pengaturan
Adam. Setiap parameter (termasuk setiap elemen matriks) punya gradien, m, dan v sendiri.

**Jika memakai batch (N > 1).** Persamaan (28) memakai rata-rata
$\mathcal{L} = \frac{1}{N}\sum_k (y'_k - \hat{y}'_k)^2$. Setiap $\hat{y}'_k$ hanya muncul di
suku ke-$k$, sehingga:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}'_k} = -\frac{2}{N}\,(y'_k - \hat{y}'_k) \qquad
\frac{\partial \mathcal{L}}{\partial W_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}\,h_{T,k} \qquad
\frac{\partial \mathcal{L}}{\partial b_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}$$

Setiap sampel menerima sinyal $\partial\mathcal{L}/\partial h_{T,k} =
\partial\mathcal{L}/\partial\hat{y}'_k \cdot W_y$ yang menjalankan BPTT-nya sendiri, lalu
gradien semua sampel dijumlahkan sebelum Adam dipanggil sekali.

```
Contoh N = 2 (sampel 1 = model mini GRU; sampel 2 = ilustrasi h_T = 0.30, y' = 0.65)
  ŷ'₁ = 0.204811,   ŷ'₂ = 0.8 × 0.30 + 0 = 0.240000
  ∂L/∂ŷ'₁ = -(2/2) × (0.7 - 0.204811) = -0.495189
  ∂L/∂ŷ'₂ = -(2/2) × (0.65 - 0.240000) = -0.410000
  ∂L/∂W_y = (-0.495189) × 0.256014 + (-0.410000) × 0.3 = (-0.126775) + (-0.123000) = -0.249775
  ∂L/∂b_y = (-0.495189) + (-0.410000) = -0.905189
```

**Versi vektor pada lapisan dense.** Pada model penelitian, h_T dan W_y masing-masing
berisi n_u elemen, sehingga ŷ′ = Σⱼ h_T,ⱼ·W_y,ⱼ + b_y. Rumus dense pada BPTT berlaku untuk
setiap elemen j: ∂L/∂W_y,ⱼ = Σₖ ∂L/∂ŷ′ₖ·h_T,ₖ,ⱼ. Hasilnya n_u gradien bobot + 1 gradien
bias, yaitu suku (n_u + 1) pada persamaan 22 dan 27. Sinyal yang masuk ke neuron j adalah
∂L/∂h_T,ⱼ = ∂L/∂ŷ′·W_y,ⱼ.

**Batasan simulasi.** Loss model mini turun hampir ke nol karena hanya ada **1 sampel**,
sehingga model bisa "menghafal" targetnya; pada data penelitian loss berhenti di nilai
yang lebih besar karena model harus menemukan pola umum. Angka akhir simulasi juga
**tidak dapat dipakai untuk membandingkan** LSTM dan GRU, karena bobot awalnya dipilih bulat
agar mudah dihitung, sedangkan Keras memakai bobot acak. Perbandingan yang sah adalah hasil
Tahap 9 sampai 12.

## 7. Ringkasan Siklus dan Hasil Verifikasi

```
SATU SIKLUS PELATIHAN GRU

 1. Inisialisasi : bobot acak, bias 0, Adam m = v = 0
 2. Forward      : x₁ → gerbang → h₁ → x₂ → ... → h_T → ŷ' = W_y h_T + b_y
 3. Loss         : L = (y' - ŷ')²  (rata-rata satu batch)
 4. Dense        : ∂L/∂ŷ' = -2(y' - ŷ') → ∂L/∂W_y, ∂L/∂b_y, ∂L/∂h_T
 5. Sel, t = T   : ∂L/∂h_T → turunan setiap gerbang → δ setiap gerbang
 6. Through time : δ dikirim ke t-1 lewat U dan lewat jalur z × h_(t-1)
 7. Ulangi 5-6 sampai t = 1
 8. Gradien      : ∂L/∂W = Σ δx,  ∂L/∂U = Σ δh_(t-1),  ∂L/∂b = Σ δ
 9. Adam         : m, v → m̂, v̂ → θ baru = θ - η m̂/(√v̂ + ε)
10. Kembali ke langkah 2 dengan batch berikutnya

HASIL SIMULASI GRU: loss 0.245212 → 0.242723 (k = 1) → 0.0000000036 (k = 500); ŷ' 0.204811 → 0.699940
```

**Pemeriksaan otomatis yang lolos saat berkas ini dibuat:**

- Gradien BPTT manual = gradien numerik untuk 14 parameter (selisih maksimum 3.5e-11).
- Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter.
- Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah 500 iterasi.
- Bias masukan dan bias rekuren z serta r tetap kembar selama 500 iterasi.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.3: Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time](tahap_08_03_simulasi_bptt_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.8: Jumlah Parameter Model GRU](tahap_08_08_jumlah_parameter_gru.md)
