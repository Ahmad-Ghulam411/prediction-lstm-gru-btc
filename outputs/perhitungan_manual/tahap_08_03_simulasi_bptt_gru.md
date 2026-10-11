# Perhitungan Manual - Tahap 8.3: Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time

Subbagian ini melanjutkan [Tahap 8.2 (Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss)](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md). Loss model mini GRU sudah
diketahui, L = 0.245212. Sekarang kesalahan itu ditelusuri **mundur** untuk
mencari gradien setiap bobot dan bias: lapisan dense dulu, lalu ke dalam sel pada t = 2,
lalu mundur ke t = 1 (*through time*).

**Daftar isi**

- [1. Rumus Turunan Dasar](#1-rumus-turunan-dasar)
- [2. Backward di Lapisan Dense](#2-backward-di-lapisan-dense)
- [3. Backward di Dalam Sel, t = 2](#3-backward-di-dalam-sel-t--2)
- [4. Mengirim Kesalahan ke t = 1 (Inti Through Time)](#4-mengirim-kesalahan-ke-t--1-inti-through-time)
- [5. Backward di Dalam Sel, t = 1](#5-backward-di-dalam-sel-t--1)
- [6. Gradien Total Setiap Bobot dan Bias](#6-gradien-total-setiap-bobot-dan-bias)
- [7. Pemeriksaan Gradien dengan Turunan Numerik](#7-pemeriksaan-gradien-dengan-turunan-numerik)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Rumus Turunan Dasar

Seluruh backward pass hanya mengulang beberapa rumus turunan berikut. Polanya selalu
sama: **kalikan sinyal kesalahan dengan turunan lokal, lalu teruskan ke kiri (ke langkah
sebelumnya).**

| Rumus | Bentuk | Contoh | Dipakai di |
|---|---|---|---|
| Aturan pangkat | d(u²)/du = 2u | d(x²)/dx = 2x | ∂L/∂ŷ′ |
| Fungsi linear | d(a·u + b)/du = a | d(3x + 5)/dx = 3 | pra-aktivasi dan dense: ∂a/∂W = x, ∂a/∂U = hₜ₋₁, ∂a/∂b = 1 |
| Aturan perkalian | ∂(u·v)/∂u = v | ∂(3u)/∂u = 3 | h = z·hₜ₋₁ + (1−z)·h̃, r·q pada kandidat |
| Aturan rantai | ∂L/∂a = ∂L/∂b · ∂b/∂a | - | semua langkah |
| Aturan penjumlahan | variabel yang dipakai di beberapa tempat: gradiennya dijumlah | - | h₁ dipakai di 4 tempat; bobot dipakai di setiap t |
| Turunan sigmoid | σ′(a) = σ(a)·(1 − σ(a)) | σ(1.25) = 0.777300 → σ′ = 0.173105 | gerbang z, r |
| Turunan tanh | tanh′(a) = 1 − tanh²(a) | tanh(0.35) = 0.336376 → 0.886851 | h̃ |

Asal turunan sigmoid dan tanh diuraikan di [Tahap 7.3, bagian 1](tahap_07_03_simulasi_bptt_lstm.md#1-rumus-turunan-dasar).

## 2. Backward di Lapisan Dense

Rumusnya identik dengan [Tahap 7.3, bagian 2](tahap_07_03_simulasi_bptt_lstm.md#2-backward-di-lapisan-dense); hanya nilai h₂ dan ŷ′ yang berbeda.

```
  ∂L/∂ŷ'  = -2(y' - ŷ') = -2 × (0.7 - 0.204811) = -0.990377
  ∂L/∂W_y = ∂L/∂ŷ' × h₂ = (-0.990377) × 0.256014 = -0.253551
  ∂L/∂b_y = ∂L/∂ŷ' × 1  = -0.990377
  ∂L/∂h₂  = ∂L/∂ŷ' × W_y = (-0.990377) × 0.8 = -0.792302   (masuk ke sel GRU)
```

## 3. Backward di Dalam Sel, t = 2

Rumus forward yang diturunkan: `h₂ = z₂ × h₁ + (1 − z₂) × h̃₂` (menuju update gate dan
kandidat), lalu `h̃₂ = tanh(W_h x₂ + b_h(in) + r₂ × q₂)` (menuju reset gate dan bagian
rekuren q₂).

```
Masukan dari bagian 2: ∂L/∂h₂ = -0.792302

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
(h̃₂ = 0.411706) lebih besar daripada memori lama (h₁ = 0.147273), sedangkan
prediksi masih terlalu rendah. Adam nanti akan **menurunkan** z, artinya GRU belajar
untuk lebih mempercayai informasi baru.

## 4. Mengirim Kesalahan ke t = 1 (Inti Through Time)

h₁ dipakai di empat tempat pada t = 2: langsung di `z₂ × h₁`, di update gate (lewat
U_z), di reset gate (lewat U_r), dan di bagian rekuren q₂ (lewat U_h). Keempat
kontribusinya dijumlahkan.

```
  ∂L/∂h₁ = ∂L/∂h₂ × z₂ + U_z × δz₂ + U_r × δr₂ + U_h × ∂L/∂q₂
         = (-0.792302) × 0.588776 + 0.4 × 0.050726 + 0.3 × (-0.001913) + 0.2 × (-0.162269)
         = (-0.466489) + 0.020291 + (-0.000574) + (-0.032454)
         = -0.479226
  Porsi jalur langsung z₂ × h₁ = -0.466489 / -0.479226 = 97.3%
```

Sekitar **97.3%** sinyal mengalir lewat jalur langsung `zₜ × hₜ₋₁`. Jalur ini
adalah "jalan tol gradien" pada GRU, padanan cell state pada LSTM.

## 5. Backward di Dalam Sel, t = 1

```
Masukan dari bagian 4: ∂L/∂h₁ = -0.479226

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

## 6. Gradien Total Setiap Bobot dan Bias

Kontribusi kedua time step dijumlahkan. Untuk kandidat, bobot $U_h$ dan bias
$b_h^{(rec)}$ berada di dalam $q_t$, sehingga gradiennya memakai
$\partial L/\partial q_t$; bobot $W_h$ dan bias $b_h^{(in)}$ memakai $\delta\tilde{h}_t$.

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

Lapisan dense (dari bagian 2):
  ∂L/∂W_y = -0.253551
  ∂L/∂b_y = -0.990377
```

Gradien bias masukan dan bias rekuren pada z dan r **selalu sama**, sedangkan pada
kandidat **berbeda** karena bias rekuren ikut dikalikan rₜ. Penjelasan lengkap tentang dua
bias ini ada di [Tahap 8.12 (Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias)](tahap_08_12_bias_gru.md).

## 7. Pemeriksaan Gradien dengan Turunan Numerik

Untuk memastikan seluruh rantai turunan di atas benar, setiap gradien dibandingkan
dengan turunan numerik (beda pusat):
$\dfrac{L(\theta+10^{-6}) - L(\theta-10^{-6})}{2\cdot10^{-6}}$.

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

Selisih terbesar hanya 3.5e-11, jadi seluruh gradien manual terbukti benar.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.2: Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.4: Simulasi Pelatihan GRU, Langkah 4: Update Adam](tahap_08_04_simulasi_update_adam_gru.md)
