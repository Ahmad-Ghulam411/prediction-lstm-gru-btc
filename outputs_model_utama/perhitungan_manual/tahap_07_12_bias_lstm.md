# Perhitungan Manual - Tahap 7.12: Bias pada LSTM: Peran dan Cara Menghitungnya

Subbagian ini merangkum pembahasan **bias pada LSTM**: kenapa bias perlu ada dan bagaimana
nilainya diperoleh. Semua contoh
angka diambil dari model LSTM utama (40 neuron): bobot awal dan satu siklus
pelatihannya (Tahap 7.6 sampai Tahap 7.9) serta bobot akhirnya.

**Daftar isi**

- [1. Intinya](#1-intinya)
- [2. Mengapa Bias Diperlukan](#2-mengapa-bias-diperlukan)
- [3. Cara Menghitung Bias Langkah demi Langkah](#3-cara-menghitung-bias-langkah-demi-langkah)
- [4. Catatan untuk Naskah Skripsi](#4-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Intinya

Bias itu **titik awal (intercept)** setiap gerbang dan neuron, mirip intercept *a* pada
regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa bias, setiap kali
masukannya nol, gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu setengah terbuka;
tanh(0) = 0). Dengan bias, setiap gerbang bisa punya **posisi bawaannya sendiri**. Nilainya
tidak dihitung dengan satu rumus, tetapi **dipelajari** lewat siklus yang sama dengan bobot
(bagian 3).

## 2. Mengapa Bias Diperlukan

**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Kondisi ini sering
terjadi: h₀ = 0 di awal setiap jendela, dan normalisasi min-max membuat fitur bernilai dekat
0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman, sedangkan bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:

| x | Dengan bias: σ(5x − 2.5) | Tanpa bias: σ(5x) |
|---|---|---|
| 0 | 0.076 | 0.500 |
| 0.25 | 0.223 | 0.777 |
| 0.5 | 0.500 | 0.924 |
| 0.75 | 0.777 | 0.977 |
| 1 | 0.924 | 0.993 |

Tanpa bias, gerbang **tidak pernah di bawah 0.5** untuk data ternormalisasi yang positif,
sehingga model tidak bisa membuat aturan "tutup gerbang saat fitur rendah, buka saat tinggi".

**(c) Bias menentukan sikap bawaan setiap gerbang.** Contoh paling penting adalah forget
gate. Di Tahap 7.6 terlihat bahwa pada bobot awal, forget gate neuron ke-1
sudah bernilai 0.729 di hari pertama karena biasnya 1. Kalau gerbang hanya ditentukan
biasnya, sisa memori setelah 7 hari adalah f^7:

| Bias forget | f = σ(b) | Memori tersisa setelah 7 hari |
|---|---|---|
| 0 (tanpa bias) | 0.500 | 0.8% |
| 1 (bobot awal Keras) | 0.731 | 11.2% |
| 0.681141 (neuron ke-1 model terlatih) | 0.664 | 5.7% |
| 2 | 0.881 | 41.1% |

Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*). Sikap bawaan setiap gerbang pada **model
LSTM terlatih** (neuron ke-1, saat xW + hU = 0; nilai bias yang sama tampil di Tahap 7.11):

| Gerbang | Bias | Nilai bawaan | Artinya |
|---|---|---|---|
| input i | -0.342494 | σ = 0.415 | cenderung hemat menerima informasi baru |
| forget f | 0.681141 | σ = 0.664 | cenderung menyimpan memori |
| kandidat c̃ | -0.011402 | tanh = -0.011 | isi bawaan hampir netral |
| output o | -0.373906 | σ = 0.408 | cenderung menahan sebagian keluaran |

Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari mengikuti
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menentukan tingkat dasar prediksi.** h_T selalu di rentang (−1, 1), jadi b_y
yang mengatur "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada model LSTM
terlatih, b_y = 0.081419, setara 0.081419 × 98,196.75 ≈
**7,995.13 USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih (25,162.70 USD) setiap kali h_T = 0.

**(e) Bias selalu mendapat sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai nol
kalau masukannya nol (di Tahap 7.8, U tidak mendapat sumbangan dari t = 1 karena
h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa-apa.

**(f) Biayanya kecil.** Model LSTM terlatih memiliki 161 bias (termasuk b_y) dari 8,201 parameter, atau 2.0%.

## 3. Cara Menghitung Bias Langkah demi Langkah

Bias tidak punya rumus langsung; bias **dipelajari** lewat siklus yang sama dengan bobot.
Berikut lima langkahnya pada model LSTM utama, memakai angka yang sudah dihitung di
subbagian sebelumnya. Contohnya **b_y** dan **b_f[1]** (bias neuron ke-1).

```
Langkah 1 — Nilai awal (Tahap 7.6)
  b_f = 1 untuk semua neuron (unit forget bias), b_i = b_c = b_o = 0, dan b_y = 0.

Langkah 2 — Dipakai di forward pass (Tahap 7.6)
  Gerbang : a = x W + h U + b
  Dense   : ŷ′ = h_T W_y + b_y

Langkah 3 — Gradien (Tahap 7.8)
  Karena ∂a/∂b = 1, gradien bias = jumlah δ untuk semua sampel dan semua hari:
  ∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = -0.007474
  ∂L/∂b_f[1] = Σₜ Σₙ δa_f  (neuron ke-1), per hari:
    t=1: 0, t=2: -5.1496e-08, t=3: -5.7490e-07, t=4: -1.3810e-06, t=5: -2.0738e-06, t=6: -1.9508e-06, t=7: 1.0300e-06
    jumlah = -5.0019e-06

Langkah 4 — Update Adam (Tahap 7.9)
  b_y : 0.000000 → 0.001000 (k = 1) → 0.000295 (k = 2)
  b_f[1]: 1.000000 → 1.000613 (k = 1) → 1.001371 (k = 2)

Langkah 5 — Ulangi 13,000 kali (26 batch × 500 epoch)
  b_y akhir = 0.081419,  b_f[1] akhir = 0.681141
```

## 4. Catatan untuk Naskah Skripsi

**Kalimat ringkas tentang fungsi bias** (misalnya setelah persamaan 9 di subbab 1.5.7):

> Bias berfungsi menggeser fungsi aktivasi sehingga setiap gerbang dan neuron dapat
> menentukan kondisi bawaannya sendiri dan tidak dipaksa bernilai tetap ketika masukannya
> bernilai nol, misalnya σ(0) = 0,5 atau tanh(0) = 0. Pada LSTM dan GRU, bias menentukan
> ambang dan kondisi bawaan setiap gerbang, sedangkan pada lapisan dense bias menentukan
> tingkat dasar nilai prediksi.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.11: Forward Pass LSTM (Jendela Pertama Data Uji)](tahap_07_11_forward_pass_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.4: Jumlah Parameter Model GRU](tahap_08_04_jumlah_parameter_gru.md)
