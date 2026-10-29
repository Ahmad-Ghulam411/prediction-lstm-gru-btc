# Perhitungan Manual - Tahap 8.8: Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time

Langkah 3 dari siklus pelatihan GRU: menghitung gradien loss batch pertama
([Tahap 8.7 (Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss)](tahap_08_07_pelatihan_gru_forward_pass_dan_loss.md)) terhadap **seluruh 3,811 parameter** model GRU
utama. Polanya sama dengan LSTM ([Tahap 7.8 (Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time)](tahap_07_08_pelatihan_lstm_bptt.md)); yang berbeda rumus di dalam sel.

**Daftar isi**

- [1. Ide Dasarnya](#1-ide-dasarnya)
- [2. Masuk ke Sel pada t = 7 (Sampel 1, Neuron ke-1)](#2-masuk-ke-sel-pada-t--7-sampel-1-neuron-ke-1)
- [3. Mundur ke t = 6: Jalan Langsung dan Jalan Lewat Gerbang](#3-mundur-ke-t--6-jalan-langsung-dan-jalan-lewat-gerbang)
- [4. Seberapa Jauh Sinyal Mengalir Mundur](#4-seberapa-jauh-sinyal-mengalir-mundur)
- [5. Gradien Total Setiap Bobot dan Bias](#5-gradien-total-setiap-bobot-dan-bias)
- [6. Pemeriksaan](#6-pemeriksaan)
- [7. Catatan untuk Naskah Skripsi](#7-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Ide Dasarnya

Gradien menjawab pertanyaan sederhana: **kalau satu parameter dinaikkan sedikit, loss naik
atau turun, dan seberapa cepat?** Tandanya memberi arah (positif berarti parameter harus
diturunkan), besarnya memberi seberapa sensitif loss terhadap parameter itu.

Karena loss bergantung pada parameter lewat rantai yang panjang (bobot → gerbang → memori →
hidden state → ... → prediksi → loss), gradien dihitung dengan **aturan rantai**, dimulai
dari belakang (loss) dan berjalan mundur ke hari pertama. Itulah arti *backpropagation
through time*. Dua turunan dasar yang dipakai berulang kali:

- σ′(a) = σ(a)(1 − σ(a))
- tanh′(a) = 1 − tanh²(a)

### Mulai dari loss

Dari L = (1/32) Σ (y′ₙ − ŷ′ₙ)², turunannya terhadap prediksi sampel ke-n adalah
∂L/∂ŷ′ₙ = 2(ŷ′ₙ − y′ₙ)/32. Untuk tiga sampel pertama:

```
  sampel 1: 2 × (-0.125065 − 0.052770) / 32 = -0.011115
  sampel 2: 2 × (-0.108161 − 0.051446) / 32 = -0.009975
  sampel 3: 2 × (-0.093345 − 0.051446) / 32 = -0.009049
  … (sampel lainnya dengan cara yang sama)
```

### Lapisan Dense

```
∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = -0.011115 + (-0.009975) + … (32 suku) = -0.224177
∂L/∂W_y[1] = Σₙ ∂L/∂ŷ′ₙ × h₇,ₙ[1]
           = (-0.011115)×(-0.200635) + (-0.009975)×(-0.197398) + … = 0.044590
```

Bias b_y menerima gradien paling "polos": jumlah semua ∂L/∂ŷ′. Bobot W_y[j] mendapat gradien
yang juga ditimbang dengan nilai h₇[j]: neuron yang keluarannya besar ikut
bertanggung jawab lebih besar atas galat.

## 2. Masuk ke Sel pada t = 7 (Sampel 1, Neuron ke-1)

```
δh₇ = ∂L/∂ŷ′₁ × W_y[1] = (-0.011115) × 0.277997 = -0.003090

Update gate, dari h = z × h_lama + (1 − z) × h̃:
  ∂L/∂z₇ = δh × (h₆ − h̃₇) = (-0.003090) × (-0.171977 − (-0.224005)) = -0.000161
  δa_z    = ∂L/∂z × z(1 − z) = (-0.000161) × 0.247418 = -3.9775e-05
Kandidat:
  ∂L/∂h̃₇ = δh × (1 − z) = (-0.003090) × 0.550812 = -0.001702
  δa_h    = ∂L/∂h̃ × (1 − h̃²) = (-0.001702) × 0.949822 = -0.001617
Reset gate, dari a_h = x W_h + b_h(in) + r × q, dengan q = h_lama U_h + b_h(rec):
  ∂L/∂r₇ = δa_h × q = (-0.001617) × 0.047668 = -7.7057e-05
  δa_r    = ∂L/∂r × r(1 − r) = (-7.7057e-05) × 0.242111 = -1.8656e-05
  ∂L/∂q   = δa_h × r = (-0.001617) × 0.411182 = -0.000665
```

Perhatikan bahwa kandidat punya dua "jatah kesalahan": δa_h untuk jalur masukan (dipakai
W_h dan b_h(in)) dan ∂L/∂q = δa_h × r untuk jalur rekuren (dipakai U_h dan b_h(rec)). Inilah
sebabnya dua bias kandidat GRU mendapat gradien berbeda ([Tahap 8.12 (Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias)](tahap_08_12_bias_gru.md)).

## 3. Mundur ke t = 6: Jalan Langsung dan Jalan Lewat Gerbang

```
Jalan 1, langsung (z × h_lama pada Gambar 7): cukup dikali update gate
  δh₇ × z₇ = (-0.003090) × 0.449188 = -0.001388
Jalan 2, lewat tiga gerbang × 30 neuron (δa_z U_z + δa_r U_r + ∂L/∂q U_h)
  = -0.000169

δh₆[1] = (-0.001388) + (-0.000169) = -0.001557
```

Untuk sampel dan neuron ini, 89.1% besar sinyal ke h₆ datang lewat jalan
langsung. Jalan ini hanya dikali z, tanpa matriks, jadi perannya mirip jalan tol cell state
pada LSTM.

## 4. Seberapa Jauh Sinyal Mengalir Mundur

| t | rata-rata \|δhₜ\| | dibanding t = 7 |
|---|---|---|
| 7 | 0.001491 | 100.0% |
| 6 | 0.000754 | 50.6% |
| 5 | 0.000404 | 27.1% |
| 4 | 0.000221 | 14.8% |
| 3 | 0.000129 | 8.6% |
| 2 | 7.4917e-05 | 5.0% |
| 1 | 4.5298e-05 | 3.0% |

Di hari pertama sinyalnya tinggal 3.0% dari hari
terakhir. Jadi hari-hari terakhir dalam jendela paling menentukan arah perbaikan bobot,
sedangkan hari-hari awal hanya ikut menyumbang sedikit.

## 5. Gradien Total Setiap Bobot dan Bias

Seperti LSTM, gradien setiap parameter adalah jumlah sumbangan dari 32 sampel × 7 hari:
∂L/∂W_g = Σ xᵀδa_g dan ∂L/∂b_g(in) = Σ δa_g untuk g ∈ {z, r, h}; untuk jalur rekuren
∂L/∂U_g = Σ hₜ₋₁ᵀ δ_g(rec) dan ∂L/∂b_g(rec) = Σ δ_g(rec), dengan δ_z(rec) = δa_z,
δ_r(rec) = δa_r, dan δ_h(rec) = ∂L/∂q.

| Parameter | Bentuk | Elemen | Contoh: elemen pertama | Rata-rata \|g\| | \|g\| terbesar |
|---|---|---|---|---|---|
| W_z | (10, 30) | 300 | -1.0759e-05 | 0.000125 | 0.001143 |
| W_r | (10, 30) | 300 | -3.2173e-05 | 0.000286 | 0.002656 |
| W_h | (10, 30) | 300 | -0.002912 | 0.013429 | 0.055316 |
| U_z | (30, 30) | 900 | 2.6822e-05 | 4.7781e-05 | 0.000408 |
| U_r | (30, 30) | 900 | 0.000129 | 0.000152 | 0.001666 |
| U_h | (30, 30) | 900 | 0.004758 | 0.003425 | 0.019044 |
| b_z(in) | (30,) | 30 | -0.000220 | 0.000409 | 0.002023 |
| b_r(in) | (30,) | 30 | -0.000669 | 0.001011 | 0.004863 |
| b_h(in) | (30,) | 30 | -0.060545 | 0.047388 | 0.101297 |
| b_z(rec) | (30,) | 30 | -0.000220 | 0.000409 | 0.002023 |
| b_r(rec) | (30,) | 30 | -0.000669 | 0.001011 | 0.004863 |
| b_h(rec) | (30,) | 30 | -0.024833 | 0.023137 | 0.055955 |
| W_y | (30, 1) | 30 | 0.044590 | 0.035901 | 0.079333 |
| b_y | (1,) | 1 | -0.224177 | 0.224177 | 0.224177 |

Lihat baris bias: gradien b_z(in) sama persis dengan b_z(rec), begitu pula b_r(in) dengan
b_r(rec) (selisihnya tepat 0), sedangkan b_h(in) dan b_h(rec) berbeda.

## 6. Pemeriksaan

Seluruh 3,811 gradien dihitung dengan cara di atas lalu dibandingkan dengan dua
cara lain:

1. **TensorFlow `GradientTape`** (diferensiasi otomatis yang dipakai `model.fit`): selisih
   terbesar di antara semua parameter = **1.7e-08** (gradien terbesar
   0.224177).
2. **Turunan numerik**: naikkan dan turunkan satu parameter sebesar 10⁻⁶, hitung ulang loss,
   lalu (L₊ − L₋)/(2 × 10⁻⁶). Cara ini tidak memakai rumus turunan sama sekali.

| Parameter | BPTT manual | Turunan numerik |
|---|---|---|
| b_y[1] | -0.224177 | -0.224177 |
| W_y[1, 1] | 0.044590 | 0.044590 |
| W_z[1, 1] | -1.0759e-05 | -1.0759e-05 |
| U_z[1, 1] | 2.6822e-05 | 2.6822e-05 |

Ketiga cara memberi angka yang sama, jadi rumus BPTT di atas memang rumus yang dipakai Keras.

## 7. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11):

> Pada GRU, sinyal kesalahan untuk hidden state hari sebelumnya mengalir melalui jalur
> langsung zₜ ⊙ hₜ₋₁ yang hanya dikalikan update gate, serta melalui ketiga lapisan
> berbobot. Jalur langsung ini berperan serupa dengan jalur cell state pada LSTM.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.7: Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_08_07_pelatihan_gru_forward_pass_dan_loss.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.9: Pelatihan GRU Model Utama, Langkah 4: Update Adam](tahap_08_09_pelatihan_gru_update_adam.md)
