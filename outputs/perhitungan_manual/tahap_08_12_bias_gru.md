# Perhitungan Manual - Tahap 8.12: Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias

Subbagian ini mengumpulkan pembahasan tentang **bias pada GRU**: mengapa bias perlu ada,
dan bagaimana bias dihitung langkah demi langkah, serta mengapa GRU memiliki dua jenis bias.
Angka contohnya diambil dari simulasi model mini ([Tahap 8.2 (Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss)](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md) sampai
[Tahap 8.4 (Simulasi Pelatihan GRU, Langkah 4: Update Adam)](tahap_08_04_simulasi_update_adam_gru.md)) dan dari model GRU terlatih penelitian.

**Daftar isi**

- [1. Intinya](#1-intinya)
- [2. Mengapa Bias Diperlukan](#2-mengapa-bias-diperlukan)
- [3. Cara Menghitung Bias Langkah demi Langkah](#3-cara-menghitung-bias-langkah-demi-langkah)
- [4. Dua Bias pada GRU: Asal dan Buktinya](#4-dua-bias-pada-gru-asal-dan-buktinya)
- [5. Catatan untuk Naskah Skripsi](#5-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Intinya

Bias adalah **titik awal (intercept)** setiap gerbang dan neuron, sama seperti intercept
*a* pada regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa bias, ketika
masukannya nol, setiap gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu setengah
terbuka; tanh(0) = 0). Dengan bias, setiap gerbang dapat menentukan **posisi bawaannya
sendiri**. Bias tidak dihitung dengan satu rumus langsung; bias **dipelajari** lewat siklus
yang sama dengan bobot (bagian 3).

## 2. Mengapa Bias Diperlukan

**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Ini sering
terjadi di penelitian: h₀ = 0 di awal setiap jendela, dan normalisasi min-max membuat
fitur bernilai dekat 0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman kurva, bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:

| x | Dengan bias: σ(5x − 2.5) | Tanpa bias: σ(5x) |
|---|---|---|
| 0 | 0.076 | 0.500 |
| 0.25 | 0.223 | 0.777 |
| 0.5 | 0.500 | 0.924 |
| 0.75 | 0.777 | 0.977 |
| 1 | 0.924 | 0.993 |

Tanpa bias, gerbang **tidak pernah turun di bawah 0.5** untuk data ternormalisasi yang
positif, sehingga model tidak bisa menyatakan aturan "tutup gerbang saat nilai fitur
rendah, buka saat tinggi".

**(c) Bias menentukan perilaku bawaan setiap gerbang.** Pada GRU, bias update gate
menentukan campuran bawaan memori lama vs baru, dan bias reset gate menentukan seberapa
banyak masa lalu dipakai untuk kandidat. Nilai bawaan pada **model GRU terlatih** (neuron
ke-1, saat kontribusi xW + hU = 0; nilai bias yang sama tampil di Tahap 8.11):

| Gerbang | Bias masukan + bias rekuren | Nilai bawaan | Arti |
|---|---|---|---|
| update z | 0.211800 + 0.211800 = 0.423600 | σ = 0.604 | cenderung mempertahankan 60% memori lama |
| reset r | -0.249689 + (-0.249689) = -0.499377 | σ = 0.378 | memakai sekitar 38% masa lalu untuk kandidat |
| kandidat h̃ | b_h(in) = 0.000974, b_h(rec) = 0.007737 | - | b_h(rec) ikut dikalikan rₜ (bagian 4) |

Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari sesuai
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di rentang (−1, 1),
jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada
model GRU terlatih, b_y = 0.042936, setara 0.042936 × 98,196.75
≈ **4,216.21 USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih (25,162.70 USD) setiap kali h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai
nol bila masukannya nol (lihat ∂L/∂U di
[Tahap 8.3, bagian 6](tahap_08_03_simulasi_bptt_gru.md#6-gradien-total-setiap-bobot-dan-bias): pada t = 1 tidak ada
kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.

**(f) Biayanya kecil.** Model GRU terlatih memiliki 181 bias (termasuk b_y) dari 3,811 parameter, atau 4.7%.

## 3. Cara Menghitung Bias Langkah demi Langkah

Bias tidak dihitung dengan satu rumus langsung; bias **dipelajari** dengan cara yang
persis sama seperti bobot. Berikut lima langkahnya pada model mini, dengan rujukan ke
subbagian yang sudah dihitung:

```
Langkah 1 — Nilai awal (Tahap 8.2)
  Semua bias gerbang (masukan dan rekuren) = 0. Bias dense b_y = 0.

Langkah 2 — Dipakai di forward pass (Tahap 8.2)
  Gerbang : a = W × x + U × h + b
  Dense   : ŷ' = W_y × h_T + b_y

Langkah 3 — Hitung gradiennya (Tahap 8.3, bagian 6)
  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga
  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ
  Contoh b_y : ∂L/∂b_y = ∂L/∂ŷ' × 1 = -0.990377
  Contoh b_h(in)  : δh̃₁ + δh̃₂ = (-0.186076) + (-0.270587) = -0.456663
  Contoh b_h(rec) : ∂L/∂q₁ + ∂L/∂q₂ = (-0.106890) + (-0.162269) = -0.269159

Langkah 4 — Update dengan Adam (Tahap 8.4)
  b_y: 0 → 0.001000 (k = 1) → 0.002000 (k = 2)

Langkah 5 — Ulangi siklus (Tahap 8.4, perjalanan loss)
  Setelah 500 iterasi b_y = 0.159666.
```

Ringkasan rumus gradien seluruh bias pada model mini (iterasi k = 1):

| Bias | Rumus gradien | Nilai |
|---|---|---|
| Dense b_y | ∂L/∂ŷ′ | -0.990377 |
| b_z(in) dan b_z(rec) | keduanya δz₁ + δz₂ | 0.090403 dan 0.090403 |
| b_r(in) dan b_r(rec) | keduanya δr₁ + δr₂ | -0.001913 dan -0.001913 |
| b_h(in) | δh̃₁ + δh̃₂ | -0.456663 |
| b_h(rec) | δh̃₁·r₁ + δh̃₂·r₂ | -0.269159 |

## 4. Dua Bias pada GRU: Asal dan Buktinya

**Letaknya pada Gambar 7** ([Tahap 8.1 (Alur Sel GRU (Gambar 7) dengan Model Mini)](tahap_08_01_alur_sel_gru.md)). Bias tidak digambar; bias berada di
dalam setiap kotak kuning (σ, σ, tanh). Setiap kotak menerima dua garis masuk, yaitu xₜ dari
bawah dan hₜ₋₁ dari garis vertikal kiri. Dengan `reset_after=True`, Keras menghitung kedua
garis itu terpisah, masing-masing dengan biasnya sendiri: garis xₜ membawa **bias masukan**
(xₜW + b(in)) dan garis hₜ₋₁ membawa **bias rekuren** (hₜ₋₁U + b(rec)).

```
Kotak σ untuk z (dan r, sama persis): kedua cabang langsung dijumlahkan
  h_(t-1) ──► h_(t-1) × U_z + b_z(rec) ──┐
                                         (+) ──► σ ──► z_t
  x_t ─────► x_t × W_z + b_z(in) ────────┘

Kotak tanh untuk kandidat: cabang h_(t-1) melewati lingkaran × milik r_t dulu
  h_(t-1) ──► h_(t-1) × U_h + b_h(rec) ──► (× r_t) ──┐
                                                     (+) ──► tanh ──► h̃_t
  x_t ─────► x_t × W_h + b_h(in) ─────────────────────┘
```

**Pada z dan r, dua bias sebenarnya berlebih.** b(in) + b(rec) langsung dijumlahkan, jadi
gradien keduanya selalu sama. Karena nilai awalnya juga sama (0), keduanya akan selalu
kembar:

- model mini: setelah 500 iterasi, b_z(in) = b_z(rec) = -0.190088 dan
  b_r(in) = b_r(rec) = 0.249159;
- model GRU terlatih (neuron ke-1): b_z(in) = 0.211800, b_z(rec) = 0.211800; b_r(in) = -0.249689, b_r(rec) = -0.249689 (kembar).

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikalikan rₜ (jika rₜ mendekati 0,
bias ini ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu aktif. Gradiennya
berbeda (Σ δh̃·r vs Σ δh̃), sehingga nilainya juga berbeda:

- model mini: gradien -0.456663 vs -0.269159; setelah 500 iterasi
  b_h(in) = 0.158837 dan b_h(rec) = 0.168930;
- model terlatih (neuron ke-1): b_h(in) = 0.000974 dan b_h(rec) = 0.007737.

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak memuat bias sama sekali
(penulisnya menyebut *"to make the equations uncluttered, we omit biases"*). Bentuk dua
bias berasal dari implementasi GRU pada pustaka **NVIDIA cuDNN**, yang memisahkan bagian
masukan dan bagian rekuren setiap gerbang. Keras memakainya sebagai bawaan
(`reset_after=True`, *"cuDNN compatible"*). LSTM di Keras cukup memakai satu bias karena
pada LSTM semua bias hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU 30 neuron memiliki
2 × 3 × 30 = 180 bias gerbang (bias Keras berbentuk
(2, 90)). Dengan satu bias jumlahnya hanya 90, sehingga total
parameter menjadi 3,721, bukan 3,811.

## 5. Catatan untuk Naskah Skripsi

**Penjelasan dua bias GRU** (setelah Gambar 7 atau persamaan 26):

> Pada Gambar 7, bias tidak digambarkan secara eksplisit karena termuat di dalam setiap
> lapisan (kotak σ dan tanh). Pada implementasi Keras dengan pengaturan reset_after=True,
> setiap lapisan memisahkan kontribusi masukan xₜW + b⁽ⁱⁿ⁾ dan kontribusi rekuren
> hₜ₋₁U + b⁽ʳᵉᶜ⁾, masing-masing dengan biasnya sendiri. Pada update gate dan reset gate
> kedua bias hanya dijumlahkan, sedangkan pada kandidat hidden state bias rekuren
> b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga keduanya tidak dapat digabung
> menjadi satu. Pemisahan ini mengikuti implementasi GRU pada pustaka cuDNN yang
> digunakan Keras (Chollet et al., 2015), sedangkan Cho et al. (2014) sendiri tidak
> menuliskan suku bias pada persamaannya.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.11: Forward Pass GRU (Jendela Pertama Data Uji)](tahap_08_11_forward_pass_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 9.2: Perbandingan Struktur LSTM dan GRU](tahap_09_02_perbandingan_struktur_lstm_gru.md)
