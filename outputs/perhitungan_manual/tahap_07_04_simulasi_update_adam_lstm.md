# Perhitungan Manual - Tahap 7.4: Simulasi Pelatihan LSTM, Langkah 4: Update Adam

Subbagian ini melanjutkan [Tahap 7.3 (Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time)](tahap_07_03_simulasi_bptt_lstm.md). Gradien setiap bobot dan bias sudah
diketahui; sekarang **Adam** memakai gradien itu untuk menggeser setiap parameter. Setelah
itu forward pass diulang untuk membuktikan loss turun, lalu siklusnya diulang sampai
500 iterasi.

**Daftar isi**

- [1. Adam: Alur Satu Kali Update](#1-adam-alur-satu-kali-update)
- [2. Contoh Angka: 1 Parameter, 2 Iterasi](#2-contoh-angka-1-parameter-2-iterasi)
- [3. Update Adam pada Model Mini, Iterasi k = 1](#3-update-adam-pada-model-mini-iterasi-k--1)
- [4. Forward Ulang dengan Bobot Baru](#4-forward-ulang-dengan-bobot-baru)
- [5. Iterasi k = 2: Momentum Mulai Bekerja](#5-iterasi-k--2-momentum-mulai-bekerja)
- [6. Koreksi Bias Seiring Iterasi](#6-koreksi-bias-seiring-iterasi)
- [7. Siklus Diulang: Perjalanan Loss](#7-siklus-diulang-perjalanan-loss)
- [8. Dari Model Mini ke Model Penelitian](#8-dari-model-mini-ke-model-penelitian)
- [9. Mengapa Adam Cocok untuk Penelitian Ini](#9-mengapa-adam-cocok-untuk-penelitian-ini)
- [10. Ringkasan Siklus dan Hasil Verifikasi](#10-ringkasan-siklus-dan-hasil-verifikasi)
- [11. Catatan untuk Naskah Skripsi](#11-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Adam: Alur Satu Kali Update

Adam dijalankan **untuk setiap parameter secara terpisah**. Setiap bobot dan bias punya
m dan v miliknya sendiri, sehingga setiap parameter punya "kecepatan belajar" sendiri.
Satu kali update terdiri dari empat langkah:

**Langkah A — momen pertama (persamaan 29, kiri)**

$$m_k = \beta_1\, m_{k-1} + (1-\beta_1)\, g_k \qquad (\beta_1 = 0.9)$$

Isinya adalah **rata-rata bergerak dari arah gradien**, kira-kira merangkum ±10 gradien
terakhir karena 1/(1 − 0.9) = 10. Fungsinya seperti **momentum bola yang
menggelinding**: gradien dari satu batch bisa "berisik", dan dengan dirata-rata, arah
langkah menjadi lebih stabil.

**Langkah B — momen kedua (persamaan 29, kanan)**

$$v_k = \beta_2\, v_{k-1} + (1-\beta_2)\, g_k^2 \qquad (\beta_2 = 0.999)$$

Isinya adalah **rata-rata bergerak dari besarnya gradien (dikuadratkan)**, kira-kira
merangkum ±1,000 gradien terakhir. Fungsinya **mengukur seberapa besar atau bergejolak
gradien parameter itu**; nilainya dipakai sebagai pembagi di langkah D.

**Langkah C — koreksi bias (persamaan 30)**

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad \hat{v}_k = \frac{v_k}{1-\beta_2^k}$$

m dan v dimulai dari **nol**, sehingga di awal pelatihan nilainya "tertarik" ke nol.
Pembagi (1 − βᵏ) mengoreksinya, dan pengaruhnya hilang setelah banyak iterasi
(bagian 6).

**Langkah D — update parameter (persamaan 31)**

$$\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}
\qquad (\eta = 0.001,\ \epsilon = 10^{-7})$$

- Pembilang m̂ₖ menentukan **arah** langkah.
- Pembagi √v̂ₖ menyesuaikan **ukuran langkah**: parameter yang gradiennya besar atau
  bergejolak mendapat langkah lebih kecil, sedangkan yang gradiennya kecil mendapat
  langkah relatif lebih besar.
- Tanda minus berarti bergerak **berlawanan** arah gradien, yaitu menuruni loss.
- η = 0.001 adalah batas kasar ukuran langkah. Rasio m̂/√v̂ biasanya sekitar ±1, jadi
  setiap parameter bergeser paling jauh sekitar ±0.001 per iterasi.
- ε hanya pengaman agar tidak terjadi pembagian dengan nol.

## 2. Contoh Angka: 1 Parameter, 2 Iterasi

Sebelum dipakai pada model mini, perhatikan satu bobot bernilai awal θ₀ = 0.5, dengan gradien iterasi 1 g₁ = 0.2 dan iterasi 2 g₂ = 0.1:

| Tahap | Iterasi 1 (g = 0.2) | Iterasi 2 (g = 0.1) |
|---|---|---|
| m = 0.9 m_lama + 0.1 g | 0.9 × 0 + 0.1 × 0.2 = 0.020000 | 0.9 × 0.020000 + 0.1 × 0.1 = 0.028000 |
| v = 0.999 v_lama + 0.001 g² | 0.001 × 0.2² = 0.00004000 | 0.999 × 0.00004000 + 0.001 × 0.1² = 0.00004996 |
| m̂ = m / (1 − 0.9ᵏ) | 0.020000 / 0.1 = 0.200000 | 0.028000 / 0.19 = 0.147368 |
| v̂ = v / (1 − 0.999ᵏ) | 0.00004000 / 0.001 = 0.040000 | 0.00004996 / 0.001999 = 0.024992 |
| √v̂ | 0.200000 | 0.158090 |
| Langkah η × m̂ / (√v̂ + ε) | 0.001 × 0.200000 / 0.200000 = 0.0010000 | 0.001 × 0.147368 / 0.158090 = 0.0009322 |
| θ baru = θ lama − langkah | 0.5 − 0.0010000 = 0.499000 | 0.499000 − 0.0009322 = 0.498068 |

**Bukti "tidak dipengaruhi penskalaan gradien".** Jika gradiennya **100 kali lebih besar**
(g₁ = 20, g₂ = 10), langkahnya tetap
**0.0010000 dan 0.0009322**, sama persis. Rasio m̂/√v̂
menghapus skala gradien. Sebagai pembanding, SGD biasa (langkah = η × g) akan melangkah
100 kali lebih jauh.

## 3. Update Adam pada Model Mini, Iterasi k = 1

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

**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias membuat
m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap parameter bergeser
sebesar 0.001** (= η) berlawanan arah dengan tanda gradiennya, berapa pun besar
gradiennya. Selisih kecil pada digit ke-7 (misalnya pada U_f) berasal dari ε
yang ditambahkan ke penyebut. Semua gradien negatif, sehingga **semua parameter naik 0.001**.

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding, SGD biasa
(Δ = η × g) akan menggeser b_y sebesar 0.0010515 tetapi
U_f hanya 0.0000014, yaitu
766 kali lebih kecil. Adam menyamakan kecepatan
belajar semua parameter.

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

## 4. Forward Ulang dengan Bobot Baru

```
Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:
  ŷ' = 0.176325     (sebelumnya 0.174235)
  L  = (0.7 - 0.176325)² = 0.274235     (sebelumnya 0.276429)
  Loss turun 0.002193 hanya dengan satu kali update.
```

## 5. Iterasi k = 2: Momentum Mulai Bekerja

Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT menghasilkan
gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol, sehingga langkah
Adam merupakan **campuran** gradien sekarang dan gradien sebelumnya. Berikut perhitungan
Adam untuk dua parameter yang sama:

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
(misalnya ketika bobot melewati titik minimum), m akan mengecil dan langkahnya otomatis
melambat.

## 6. Koreksi Bias Seiring Iterasi

Pembagi koreksi bias makin lama makin mendekati 1, sehingga pengaruhnya hilang. Koreksi
m hanya penting di epoch pertama, sedangkan koreksi v masih berpengaruh sampai puluhan
epoch (angka epoch memakai jumlah iterasi per epoch penelitian):

| Iterasi k | Keterangan | 1 − 0.9ᵏ (pembagi m) | 1 − 0.999ᵏ (pembagi v) |
|---|---|---|---|
| 1 | iterasi pertama | 0.100000 | 0.001000 |
| 2 | iterasi kedua | 0.190000 | 0.001999 |
| 26 | akhir epoch 1 | 0.935389 | 0.025678 |
| 1,000 | ± epoch 38 | 1.000000 | 0.632305 |
| 2,600 | akhir epoch 100 | 1.000000 | 0.925823 |
| 13,000 | akhir epoch 500 | 1.000000 | 0.999998 |
| 26,000 | akhir epoch 1,000 | 1.000000 | 1.000000 |

## 7. Siklus Diulang: Perjalanan Loss

Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah update. Bobot
berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom b_y dan
b_f memperlihatkan bias ikut dipelajari seperti bobot.

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

## 8. Dari Model Mini ke Model Penelitian

| Aspek | Model mini | Model penelitian |
|---|---|---|
| Masukan per time step | 1 angka | vektor 10 fitur |
| Time step (BPTT mundur sejauh) | 2 | 7 |
| Bobot per gerbang | angka tunggal | matriks W (10 × n_u) dan U (n_u × n_u), vektor b (n_u) |
| Loss per iterasi | 1 sampel | rata-rata 32 sampel (1 batch) |
| Jumlah iterasi | 500 | 26 per epoch × epoch grid (100, 500, 1,000) |

Jumlah parameter LSTM penelitian untuk setiap jumlah neuron pada grid (persamaan 22, termasuk dense):

| Neuron n_u | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| Parameter | 851 | 2,501 | 4,951 | 8,201 | 12,251 |

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
Contoh N = 2 (sampel 1 = model mini LSTM; sampel 2 = ilustrasi h_T = 0.30, y' = 0.65)
  ŷ'₁ = 0.174235,   ŷ'₂ = 0.8 × 0.30 + 0 = 0.240000
  ∂L/∂ŷ'₁ = -(2/2) × (0.7 - 0.174235) = -0.525765
  ∂L/∂ŷ'₂ = -(2/2) × (0.65 - 0.240000) = -0.410000
  ∂L/∂W_y = (-0.525765) × 0.217794 + (-0.410000) × 0.3 = (-0.114508) + (-0.123000) = -0.237508
  ∂L/∂b_y = (-0.525765) + (-0.410000) = -0.935765
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

## 9. Mengapa Adam Cocok untuk Penelitian Ini

- Adam menggabungkan dua ide: **momentum** (dari m) dan **langkah adaptif per parameter**
  (dari v, ide RMSProp/AdaGrad; Kingma & Ba, 2015).
- Gradien dari batch data kripto yang fluktuatif cenderung berisik; momentum meredamnya.
- Parameter LSTM sangat beragam (bobot gerbang, bobot kandidat, bias, dense). Langkah
  adaptif membuat semuanya bisa belajar dengan kecepatan wajar tanpa harus mengatur
  learning rate satu per satu: di bagian 3 terlihat gradien terbesar dan terkecil berbeda
  766 kali, tetapi semua parameter tetap bergeser ±0.001 pada iterasi pertama.
- Pengaturan Adam **identik** untuk LSTM dan GRU (η = 0.001, β₁ = 0.9, β₂ = 0.999,
  ε = 10⁻⁷, batch 32, seed sama), sehingga perbedaan hasil hanya berasal dari
  arsitektur.

## 10. Ringkasan Siklus dan Hasil Verifikasi

```
SATU SIKLUS PELATIHAN LSTM

 1. Inisialisasi : bobot acak, bias 0 (forget = 1), Adam m = v = 0
 2. Forward      : x₁ → gerbang → h₁ (dan c₁) → x₂ → ... → h_T → ŷ' = W_y h_T + b_y
 3. Loss         : L = (y' - ŷ')²  (rata-rata satu batch)
 4. Dense        : ∂L/∂ŷ' = -2(y' - ŷ') → ∂L/∂W_y, ∂L/∂b_y, ∂L/∂h_T
 5. Sel, t = T   : ∂L/∂h_T → turunan setiap gerbang → δ setiap gerbang
 6. Through time : δ dikirim ke t-1 lewat U dan lewat jalur cell state
 7. Ulangi 5-6 sampai t = 1
 8. Gradien      : ∂L/∂W = Σ δx,  ∂L/∂U = Σ δh_(t-1),  ∂L/∂b = Σ δ
 9. Adam         : m, v → m̂, v̂ → θ baru = θ - η m̂/(√v̂ + ε)
10. Kembali ke langkah 2 dengan batch berikutnya

HASIL SIMULASI LSTM: loss 0.276429 → 0.274235 (k = 1) → 0.0000055602 (k = 500); ŷ' 0.174235 → 0.697642
```

**Pemeriksaan otomatis yang lolos saat berkas ini dibuat:**

- Gradien BPTT manual = gradien numerik untuk 14 parameter (selisih maksimum 9.3e-11).
- Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter.
- Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah 500 iterasi.

## 11. Catatan untuk Naskah Skripsi

1. **Keterangan N pada persamaan (28).** Saat pelatihan, loss dihitung per *mini-batch*,
   jadi N = 32 (batch terakhir 4), bukan seluruh sampel.
   Saran kalimat:

   > Saat pelatihan, ℒ dihitung pada setiap mini-batch berukuran N = 32,
   > sedangkan loss yang dilaporkan per epoch merupakan rata-rata loss seluruh mini-batch.

2. **Arti iterasi ke-k pada Adam.** Satu iterasi adalah satu kali update per batch, bukan
   per epoch. Saran kalimat:

   > Satu iterasi k bersesuaian dengan satu mini-batch, sehingga dengan
   > 804 sampel latih dan batch size 32 terdapat
   > 26 iterasi per epoch.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.3: Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time](tahap_07_03_simulasi_bptt_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.8: Jumlah Parameter Model LSTM](tahap_07_08_jumlah_parameter_lstm.md)
