# Perhitungan Manual - Tahap 7.8: Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time

Langkah 3 dari siklus pelatihan: menghitung gradien loss batch pertama
([Tahap 7.7 (Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss)](tahap_07_07_pelatihan_lstm_forward_pass_dan_loss.md)) terhadap **seluruh 8,201 parameter** model LSTM
utama. Rinciannya kita ikuti untuk sampel 1 dan neuron ke-1, sisanya dikerjakan komputer
dengan rumus yang sama.

**Daftar isi**

- [1. Ide Dasarnya](#1-ide-dasarnya)
- [2. Masuk ke Sel pada t = 7 (Sampel 1, Neuron ke-1)](#2-masuk-ke-sel-pada-t--7-sampel-1-neuron-ke-1)
- [3. Mundur ke t = 6: Dua Jalan Sinyal Kesalahan](#3-mundur-ke-t--6-dua-jalan-sinyal-kesalahan)
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
  sampel 1: 2 × (0.041640 − 0.052770) / 32 = -0.000696
  sampel 2: 2 × (0.030432 − 0.051446) / 32 = -0.001313
  sampel 3: 2 × (0.026984 − 0.051446) / 32 = -0.001529
  … (sampel lainnya dengan cara yang sama)
```

### Lapisan Dense

```
∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = -0.000696 + (-0.001313) + … (32 suku) = -0.007474
∂L/∂W_y[1] = Σₙ ∂L/∂ŷ′ₙ × h₇,ₙ[1]
           = (-0.000696)×0.017936 + (-0.001313)×0.006218 + … = 9.7972e-05
```

Bias b_y menerima gradien paling "polos": jumlah semua ∂L/∂ŷ′. Bobot W_y[j] mendapat gradien
yang juga ditimbang dengan nilai h₇[j]: neuron yang keluarannya besar ikut
bertanggung jawab lebih besar atas galat.

## 2. Masuk ke Sel pada t = 7 (Sampel 1, Neuron ke-1)

Sinyal kesalahan yang masuk ke hidden state neuron ke-1 pada hari terakhir adalah
δh₇ = ∂L/∂ŷ′ × W_y[1]. Dari sana sinyal dibagi ke gerbang-gerbang (nilai fₜ, iₜ, c̃ₜ, oₜ,
cₜ diambil dari forward pass):

```
δh₇  = ∂L/∂ŷ′₁ × W_y[1] = (-0.000696) × 0.241729 = -0.000168

Output gate (Gambar 6), dari h = o × tanh(c):
  ∂L/∂o₇ = δh × tanh(c₇) = (-0.000168) × 0.033147 = -5.5737e-06
  δa_o    = ∂L/∂o × o(1 − o) = (-5.5737e-06) × 0.248312 = -1.3840e-06

Cell state, dari h = o × tanh(c) (belum ada sinyal dari hari sesudahnya):
  δc₇ = δh × o × (1 − tanh²c) = (-0.000168) × 0.541088 × 0.998901 = -9.0884e-05

Forget gate (Gambar 3), dari c = f × c_lama + i × c̃:
  ∂L/∂f₇ = δc × c₆ = (-9.0884e-05) × 0.051370 = -4.6687e-06
  δa_f    = ∂L/∂f × f(1 − f) = (-4.6687e-06) × 0.189648 = -8.8540e-07
Input gate (Gambar 4):
  ∂L/∂i₇ = δc × c̃ = (-9.0884e-05) × (-0.009690) = 8.8064e-07
  δa_i    = ∂L/∂i × i(1 − i) = 2.1931e-07
Kandidat:
  ∂L/∂c̃₇ = δc × i = (-9.0884e-05) × 0.531013 = -4.8260e-05
  δa_c    = ∂L/∂c̃ × (1 − c̃²) = (-4.8260e-05) × 0.999906 = -4.8256e-05
```

Angka δa inilah "jatah kesalahan" setiap gerbang. Dari δa, gradien bobotnya tinggal dikali
masukannya. Misalnya, sumbangan sampel 1 hari ke-7 untuk bobot **Close Price → forget
gate neuron ke-1**:

∂L/∂W_f[1, 1] (sumbangan ini) = δa_f × x₇,₁ = (-8.8540e-07) × 0.048427 = -4.2877e-08

dan untuk biasnya cukup δa_f itu sendiri = -8.8540e-07, karena ∂a/∂b = 1.

## 3. Mundur ke t = 6: Dua Jalan Sinyal Kesalahan

Ke hari ke-6, sinyal kesalahan untuk cell state neuron ke-1 datang lewat **dua jalan**:

```
Jalan 1, lewat cell state ("jalan tol", Gambar 2): cukup dikali forget gate
  δc₇ × f₇ = (-9.0884e-05) × 0.745666 = -6.7769e-05
Jalan 2, lewat hidden state: δh₆ dikumpulkan dari 4 gerbang × 40 neuron
  δh₆[1] = Σ (δa_f U_f + δa_i U_i + δa_c U_c + δa_o U_o)[1, :] = 1.0700e-05
  lalu masuk ke cell state: δh₆ × o₆ × (1 − tanh²c₆) = 5.6585e-06

δc₆ = (-6.7769e-05) + 5.6585e-06 = -6.2110e-05
```

Untuk sampel dan neuron ini, 92.3% besar sinyal ke c₆ datang dari jalan tol.
Jalan tol hanya melewati perkalian dengan f, tanpa matriks dan tanpa turunan aktivasi, jadi
sinyalnya tidak cepat mengecil. Inilah alasan LSTM bisa "mengingat" lebih lama daripada RNN
biasa. Proses yang sama diulang mundur sampai t = 1.

## 4. Seberapa Jauh Sinyal Mengalir Mundur

Rata-rata besar sinyal kesalahan (seluruh 32 sampel × 40 neuron) di setiap hari:

| t | rata-rata \|δhₜ\| | rata-rata \|δcₜ\| | \|δcₜ\| dibanding t = 7 |
|---|---|---|---|
| 7 | 0.000133 | 6.3528e-05 | 100.0% |
| 6 | 1.3704e-05 | 4.6893e-05 | 73.8% |
| 5 | 1.0141e-05 | 3.4814e-05 | 54.8% |
| 4 | 7.1704e-06 | 2.6045e-05 | 41.0% |
| 3 | 5.2635e-06 | 1.9936e-05 | 31.4% |
| 2 | 4.0345e-06 | 1.5419e-05 | 24.3% |
| 1 | 3.0907e-06 | 1.2047e-05 | 19.0% |

Semakin jauh ke belakang, sinyalnya mengecil: di hari
pertama tinggal 19.0% dari sinyal di hari terakhir. Artinya
hari-hari terakhir dalam jendela paling berpengaruh pada perbaikan bobot, tetapi hari pertama
pun masih ikut menyumbang.

## 5. Gradien Total Setiap Bobot dan Bias

Gradien satu parameter adalah **jumlah sumbangan dari semua sampel dan semua hari**:

$$\frac{\partial L}{\partial \mathbf{W}_g} = \sum_{n=1}^{32}\sum_{t=1}^{7} \mathbf{x}_{t,n}^{\top}\,\delta \mathbf{a}_{g,t,n}
\qquad
\frac{\partial L}{\partial \mathbf{U}_g} = \sum_{n}\sum_{t} \mathbf{h}_{t-1,n}^{\top}\,\delta \mathbf{a}_{g,t,n}
\qquad
\frac{\partial L}{\partial \mathbf{b}_g} = \sum_{n}\sum_{t} \delta \mathbf{a}_{g,t,n}$$

untuk g ∈ {f, i, c, o}. Pada t = 1 bagian U tidak mendapat sumbangan karena h₀ = 0.
Ringkasan seluruh gradien iterasi pertama:

| Parameter | Bentuk | Elemen | Contoh: elemen pertama | Rata-rata \|g\| | \|g\| terbesar |
|---|---|---|---|---|---|
| W_i | (10, 40) | 400 | 5.7619e-07 | 1.6856e-05 | 0.000111 |
| W_f | (10, 40) | 400 | -2.3416e-07 | 1.5238e-05 | 8.9734e-05 |
| W_c | (10, 40) | 400 | -8.4244e-05 | 0.000340 | 0.001328 |
| W_o | (10, 40) | 400 | 6.1053e-07 | 1.9356e-05 | 0.000128 |
| U_i | (40, 40) | 1,600 | -7.3661e-08 | 3.0524e-06 | 3.0439e-05 |
| U_f | (40, 40) | 1,600 | -2.0085e-07 | 2.9330e-06 | 2.9968e-05 |
| U_c | (40, 40) | 1,600 | -7.3982e-06 | 6.0142e-05 | 0.000394 |
| U_o | (40, 40) | 1,600 | -1.2932e-07 | 4.0296e-06 | 4.4876e-05 |
| b_i | (40,) | 40 | 1.1947e-05 | 5.5341e-05 | 0.000197 |
| b_f | (40,) | 40 | -5.0019e-06 | 4.8950e-05 | 0.000158 |
| b_c | (40,) | 40 | -0.001543 | 0.001136 | 0.002350 |
| b_o | (40,) | 40 | 1.1836e-05 | 5.7935e-05 | 0.000220 |
| W_y | (40, 1) | 40 | 9.7972e-05 | 0.000577 | 0.001786 |
| b_y | (1,) | 1 | -0.007474 | 0.007474 | 0.007474 |

Lapisan Dense (W_y, b_y) menerima gradien terbesar karena paling dekat dengan loss. Di dalam
sel, rata-rata gradien bobot masukan terbesar ada di **kandidat**.

## 6. Pemeriksaan

Seluruh 8,201 gradien dihitung dengan cara di atas lalu dibandingkan dengan dua
cara lain:

1. **TensorFlow `GradientTape`** (diferensiasi otomatis yang dipakai `model.fit`): selisih
   terbesar di antara semua parameter = **4.7e-09** (gradien terbesar
   0.007474).
2. **Turunan numerik**: naikkan dan turunkan satu parameter sebesar 10⁻⁶, hitung ulang loss,
   lalu (L₊ − L₋)/(2 × 10⁻⁶). Cara ini tidak memakai rumus turunan sama sekali.

| Parameter | BPTT manual | Turunan numerik |
|---|---|---|
| b_y[1] | -0.007474 | -0.007474 |
| W_y[1, 1] | 9.7972e-05 | 9.7972e-05 |
| W_f[1, 1] | -2.3416e-07 | -2.3416e-07 |
| U_f[1, 1] | -2.0085e-07 | -2.0085e-07 |

Ketiga cara memberi angka yang sama, jadi rumus BPTT di atas memang rumus yang dipakai Keras.

## 7. Catatan untuk Naskah Skripsi

Saran kalimat (subbab 1.5.11):

> Gradien setiap bobot dan bias dihitung dengan backpropagation through time, yaitu aturan
> rantai yang dijalankan mundur dari loss ke time step terakhir hingga time step pertama.
> Pada LSTM, sinyal kesalahan untuk cell state mengalir melalui dua jalur: jalur langsung
> cₜ₋₁ → cₜ yang hanya dikalikan forget gate, dan jalur melalui hidden state. Jalur langsung
> inilah yang membuat gradien tidak cepat menghilang.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.7: Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_07_07_pelatihan_lstm_forward_pass_dan_loss.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.9: Pelatihan LSTM Model Utama, Langkah 4: Update Adam](tahap_07_09_pelatihan_lstm_update_adam.md)
