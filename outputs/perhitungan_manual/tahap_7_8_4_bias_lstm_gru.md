# Perhitungan Manual - Tahap 7 & 8 (Bagian 4 dari 4): Bias pada LSTM dan GRU

> Berkas ini dibuat otomatis oleh `tools/simulasi_pelatihan_manual.py`. Semua
> angka dihitung ulang oleh skrip itu dan diperiksa dengan `assert`. Untuk membuat
> ulang seluruh seri: `python tools/simulasi_pelatihan_manual.py`.

**Seri perhitungan manual pelatihan LSTM dan GRU** (baca berurutan):

1. [Alur Sel LSTM dan GRU (Gambar 1-7)](tahap_7_8_1_alur_sel_lstm_gru.md)
2. [Fungsi Loss dan Adam](tahap_7_8_2_fungsi_loss_dan_adam.md)
3. [Simulasi Satu Siklus Pelatihan LSTM dan GRU](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md)
4. **Bias pada LSTM dan GRU** ← sedang dibaca

Bagian ini mengumpulkan semua pembahasan tentang **bias**: mengapa bias perlu ada di
LSTM dan GRU, bagaimana bias dihitung langkah demi langkah, dan mengapa GRU memiliki
dua jenis bias. Angka contohnya diambil dari simulasi di [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md) dan dari model
terlatih penelitian.

**Daftar isi**

- [1. Intinya](#1-intinya)
- [2. Mengapa Bias Diperlukan](#2-mengapa-bias-diperlukan)
- [3. Cara Menghitung Bias Langkah demi Langkah](#3-cara-menghitung-bias-langkah-demi-langkah)
- [4. Dua Bias pada GRU: Asal dan Buktinya](#4-dua-bias-pada-gru-asal-dan-buktinya)
- [5. Catatan untuk Naskah Skripsi](#5-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Seperti berkas perhitungan manual lainnya, angka memakai
titik sebagai pemisah desimal dan koma sebagai pemisah ribuan. Angka ditampilkan
6 desimal, tetapi skrip menghitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Intinya

Bias adalah **titik awal (intercept)** setiap gerbang dan neuron, sama seperti
intercept *a* pada regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa
bias, ketika masukannya nol, setiap gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu
setengah terbuka; tanh(0) = 0). Dengan bias, setiap gerbang dapat menentukan **posisi
bawaannya sendiri**.

Bias tidak dihitung dengan satu rumus langsung. Bias **dipelajari** lewat siklus yang
sama dengan bobot: forward → loss → BPTT → Adam (subbagian 3).

## 2. Mengapa Bias Diperlukan

**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Ini sering
terjadi di penelitian: h₀ = 0 di awal setiap jendela 7 hari, dan normalisasi min-max
membuat fitur bernilai dekat 0 ketika nilainya mendekati minimum data latih.

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

Tanpa bias, gerbang **tidak pernah turun di bawah 0.5** untuk data ternormalisasi
yang positif, sehingga model tidak bisa menyatakan aturan "tutup gerbang saat nilai
fitur rendah, buka saat tinggi".

**(c) Bias menentukan perilaku bawaan setiap gerbang.** Contoh terpenting: forget gate
LSTM. Jika gerbang hanya ditentukan oleh biasnya, sisa memori setelah 7 hari adalah f⁷:

| Bias forget | f = σ(b) | Memori tersisa setelah 7 hari (f⁷) |
|---|---|---|
| 0 (tanpa bias) | 0.500 | 0.8% |
| 0.681141 (neuron ke-1 LSTM terlatih) | 0.664 | 5.7% |
| 1 (inisialisasi Keras) | 0.731 | 11.2% |
| 2 | 0.881 | 41.1% |

Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*).

Nilai bawaan gerbang pada **model terlatih penelitian** (neuron ke-1, saat kontribusi
xW + hU = 0), dibaca dari `tahap_7_forward_pass_lstm.md` dan
`tahap_8_forward_pass_gru.md`:

| Model | Gerbang | Bias | Nilai bawaan | Arti |
|---|---|---|---|---|
| LSTM | input i | -0.342494 | σ = 0.415 | cenderung hemat menerima informasi baru |
| LSTM | forget f | 0.681141 | σ = 0.664 | cenderung mempertahankan memori |
| LSTM | kandidat c̃ | -0.011402 | tanh = -0.011 | isi bawaan hampir netral |
| LSTM | output o | -0.373906 | σ = 0.408 | cenderung menahan sebagian keluaran |
| GRU | update z | 0.211800 + 0.211800 | σ = 0.604 | cenderung mempertahankan 60% memori lama |
| GRU | reset r | -0.249689 + (-0.249689) | σ = 0.378 | memakai sekitar 38% masa lalu untuk kandidat |

Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari sesuai
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di rentang
(−1, 1), jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y cukup menangani
variasinya. Pada model terlatih, b_y LSTM = 0.081419 (setara
0.081419 × 98,196.75 ≈ **7,995.08 USD**) dan
b_y GRU = 0.042936 (≈ **4,216.18 USD**). Tanpa b_y,
prediksi dipaksa jatuh ke harga terendah data latih (25,162.70 USD) setiap kali
h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁
bernilai nol bila masukannya nol (lihat ∂L/∂U di [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md), subbagian 3.8: pada t = 1
tidak ada kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.

**(f) Biayanya kecil.** Pada model penelitian, bias hanya
161 dari 8,201 parameter LSTM
(2.0%) dan
181 dari 3,811 parameter GRU
(4.7%).

## 3. Cara Menghitung Bias Langkah demi Langkah

Bias diperlakukan **persis seperti bobot**. Berikut lima langkahnya, dengan rujukan ke
perhitungan di [Bagian 3](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md):

```
Langkah 1 — Nilai awal (Bagian 3: 3.1 dan 4.1)
  Bias gerbang = 0, kecuali bias forget gate LSTM = 1. Bias dense b_y = 0.

Langkah 2 — Dipakai di forward pass (Bagian 3: 3.2 dan 4.2)
  Gerbang : a = W × x + U × h + b
  Dense   : ŷ' = W_y × h_T + b_y

Langkah 3 — Hitung gradiennya (Bagian 3: 3.8 dan 4.8)
  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga
  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ
  Contoh b_y : ∂L/∂b_y = ∂L/∂ŷ' × 1 = -1.051530
  Contoh b_c : ∂L/∂b_c = δc̃₁ + δc̃₂ = (-0.192701) + (-0.203589) = -0.396290

Langkah 4 — Update dengan Adam (Bagian 3: 3.10 dan 3.12)
  b_y: 0 → 0.001000 (k = 1) → 0.002000 (k = 2)

Langkah 5 — Ulangi siklus (Bagian 3: 3.13)
  Setelah 500 iterasi b_y = 0.205581. Pada model penelitian, b_y LSTM = 0.081419
  adalah hasil akhir proses yang sama setelah 13,000 iterasi.
```

Ringkasan rumus gradien seluruh bias pada simulasi (iterasi k = 1):

| Bias | Rumus gradien | Nilai pada simulasi |
|---|---|---|
| Dense b_y (LSTM) | ∂L/∂ŷ′ | -1.051530 |
| LSTM b_f | δf₁ + δf₂ | -0.013077 |
| LSTM b_i | δi₁ + δi₂ | -0.072199 |
| LSTM b_c | δc̃₁ + δc̃₂ | -0.396290 |
| LSTM b_o | δo₁ + δo₂ | -0.082908 |
| Dense b_y (GRU) | ∂L/∂ŷ′ | -0.990377 |
| GRU b_z(in) dan b_z(rec) | keduanya δz₁ + δz₂ | 0.090403 dan 0.090403 |
| GRU b_r(in) dan b_r(rec) | keduanya δr₁ + δr₂ | -0.001913 dan -0.001913 |
| GRU b_h(in) | δh̃₁ + δh̃₂ | -0.456663 |
| GRU b_h(rec) | δh̃₁·r₁ + δh̃₂·r₂ | -0.269159 |

## 4. Dua Bias pada GRU: Asal dan Buktinya

**Letaknya pada Gambar 7** ([Bagian 1, subbagian 9](tahap_7_8_1_alur_sel_lstm_gru.md#9-gambar-7-struktur-sel-gru)). Bias tidak digambar; bias berada
di dalam setiap kotak kuning (σ, σ, tanh). Setiap kotak menerima dua garis masuk,
yaitu xₜ dari bawah dan hₜ₋₁ dari garis vertikal kiri. Dengan `reset_after=True`, Keras
menghitung kedua garis itu terpisah, masing-masing dengan biasnya sendiri: garis xₜ
membawa **bias masukan** (xₜW + b(in)) dan garis hₜ₋₁ membawa **bias rekuren**
(hₜ₋₁U + b(rec)).

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

**Pada z dan r, dua bias sebenarnya berlebih.** b(in) + b(rec) langsung dijumlahkan,
jadi gradien keduanya selalu sama. Karena nilai awalnya juga sama (0), keduanya akan
selalu kembar:

- simulasi: setelah 500 iterasi, b_z(in) = b_z(rec) = -0.190088 dan
  b_r(in) = b_r(rec) = 0.249159;
- model GRU terlatih penelitian (neuron ke-1): b_z(in) = 0.211800,
  b_z(rec) = 0.211800; b_r(in) = -0.249689, b_r(rec) = -0.249689.

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikalikan rₜ (jika rₜ
mendekati 0, bias ini ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu
aktif. Gradiennya berbeda (Σ δh̃·r vs Σ δh̃), sehingga nilainya juga berbeda:

- simulasi: gradien -0.456663 vs -0.269159; setelah 500 iterasi
  b_h(in) = 0.158837 dan b_h(rec) = 0.168930;
- model terlatih (neuron ke-1): b_h(in) = 0.000974 dan b_h(rec) = 0.007737.

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak memuat bias sama sekali
(penulisnya menyebut *"to make the equations uncluttered, we omit biases"*). Bentuk
dua bias berasal dari implementasi GRU pada pustaka **NVIDIA cuDNN**, yang memisahkan
bagian masukan dan bagian rekuren setiap gerbang. Keras memakainya sebagai bawaan
(`reset_after=True`, *"cuDNN compatible"*). LSTM di Keras cukup memakai satu bias
karena pada LSTM semua bias hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU 30 neuron memiliki
2 × 3 × 30 = 180 bias gerbang (Tabel 12: bias
berbentuk (2, 90)). Dengan satu bias jumlahnya hanya
90, sehingga total parameter menjadi
3,721, bukan 3,811.

## 5. Catatan untuk Naskah Skripsi

1. **Kalimat ringkas tentang fungsi bias** (misalnya setelah persamaan 9 di subbab 1.5.7):

   > Bias berfungsi menggeser fungsi aktivasi sehingga setiap gerbang dan neuron
   > dapat menentukan kondisi bawaannya sendiri dan tidak dipaksa bernilai tetap
   > ketika masukannya bernilai nol, misalnya σ(0) = 0,5 atau tanh(0) = 0. Pada LSTM
   > dan GRU, bias menentukan ambang dan kondisi bawaan setiap gerbang, sedangkan pada
   > lapisan dense bias menentukan tingkat dasar nilai prediksi.

2. **Penjelasan dua bias GRU** (setelah Gambar 7 atau persamaan 26):

   > Pada Gambar 7, bias tidak digambarkan secara eksplisit karena termuat di dalam
   > setiap lapisan (kotak σ dan tanh). Pada implementasi Keras dengan pengaturan
   > reset_after=True, setiap lapisan memisahkan kontribusi masukan xₜW + b⁽ⁱⁿ⁾ dan
   > kontribusi rekuren hₜ₋₁U + b⁽ʳᵉᶜ⁾, masing-masing dengan biasnya sendiri. Pada
   > update gate dan reset gate kedua bias hanya dijumlahkan, sedangkan pada kandidat
   > hidden state bias rekuren b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga
   > keduanya tidak dapat digabung menjadi satu. Pemisahan ini mengikuti implementasi
   > GRU pada pustaka cuDNN yang digunakan Keras (Chollet et al., 2015), sedangkan
   > Cho et al. (2014) sendiri tidak menuliskan suku bias pada persamaannya.

---

← Sebelumnya: [Bagian 3. Simulasi Satu Siklus Pelatihan LSTM dan GRU](tahap_7_8_3_simulasi_pelatihan_lstm_gru.md)

Kembali ke awal seri: [Bagian 1. Alur Sel LSTM dan GRU (Gambar 1-7)](tahap_7_8_1_alur_sel_lstm_gru.md)
