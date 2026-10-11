# Perhitungan Manual - Tahap 7.12: Bias pada LSTM: Peran dan Cara Menghitungnya

Subbagian ini mengumpulkan pembahasan tentang **bias pada LSTM**: mengapa bias perlu ada,
dan bagaimana bias dihitung langkah demi langkah.
Angka contohnya diambil dari simulasi model mini ([Tahap 7.2 (Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss)](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md) sampai
[Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md)) dan dari model LSTM terlatih penelitian.

**Daftar isi**

- [1. Intinya](#1-intinya)
- [2. Mengapa Bias Diperlukan](#2-mengapa-bias-diperlukan)
- [3. Cara Menghitung Bias Langkah demi Langkah](#3-cara-menghitung-bias-langkah-demi-langkah)
- [4. Catatan untuk Naskah Skripsi](#4-catatan-untuk-naskah-skripsi)

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

**(c) Bias menentukan perilaku bawaan setiap gerbang.** Contoh terpenting: forget gate.
Jika gerbang hanya ditentukan oleh biasnya, sisa memori setelah 7 hari adalah
f^7:

| Bias forget | f = σ(b) | Memori tersisa setelah 7 hari |
|---|---|---|
| 0 (tanpa bias) | 0.500 | 0.8% |
| 0.681141 (neuron ke-1 model terlatih) | 0.664 | 5.7% |
| 1 (inisialisasi Keras) | 0.731 | 11.2% |
| 2 | 0.881 | 41.1% |

Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*).

Nilai bawaan setiap gerbang pada **model LSTM terlatih** (neuron ke-1, saat kontribusi
xW + hU = 0; nilai bias yang sama tampil di Tahap 7.11):

| Gerbang | Bias | Nilai bawaan | Arti |
|---|---|---|---|
| input i | -0.342494 | σ = 0.415 | cenderung hemat menerima informasi baru |
| forget f | 0.681141 | σ = 0.664 | cenderung mempertahankan memori |
| kandidat c̃ | -0.011402 | tanh = -0.011 | isi bawaan hampir netral |
| output o | -0.373906 | σ = 0.408 | cenderung menahan sebagian keluaran |

Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari sesuai
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di rentang (−1, 1),
jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada
model LSTM terlatih, b_y = 0.081419, setara 0.081419 × 98,196.75
≈ **7,995.13 USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih (25,162.70 USD) setiap kali h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai
nol bila masukannya nol (lihat ∂L/∂U di
[Tahap 7.3, bagian 6](tahap_07_03_simulasi_bptt_lstm.md#6-gradien-total-setiap-bobot-dan-bias): pada t = 1 tidak ada
kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.

**(f) Biayanya kecil.** Model LSTM terlatih memiliki 161 bias (termasuk b_y) dari 8,201 parameter, atau 2.0%.

## 3. Cara Menghitung Bias Langkah demi Langkah

Bias tidak dihitung dengan satu rumus langsung; bias **dipelajari** dengan cara yang
persis sama seperti bobot. Berikut lima langkahnya pada model mini, dengan rujukan ke
subbagian yang sudah dihitung:

```
Langkah 1 — Nilai awal (Tahap 7.2)
  Bias gerbang = 0, kecuali bias forget gate = 1. Bias dense b_y = 0.

Langkah 2 — Dipakai di forward pass (Tahap 7.2)
  Gerbang : a = W × x + U × h + b
  Dense   : ŷ' = W_y × h_T + b_y

Langkah 3 — Hitung gradiennya (Tahap 7.3, bagian 6)
  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga
  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ
  Contoh b_y : ∂L/∂b_y = ∂L/∂ŷ' × 1 = -1.051530
  Contoh b_c : ∂L/∂b_c = δc̃₁ + δc̃₂ = (-0.192701) + (-0.203589) = -0.396290

Langkah 4 — Update dengan Adam (Tahap 7.4)
  b_y: 0 → 0.001000 (k = 1) → 0.002000 (k = 2)

Langkah 5 — Ulangi siklus (Tahap 7.4, perjalanan loss)
  Setelah 500 iterasi b_y = 0.205581.
```

Ringkasan rumus gradien seluruh bias pada model mini (iterasi k = 1):

| Bias | Rumus gradien | Nilai |
|---|---|---|
| Dense b_y | ∂L/∂ŷ′ | -1.051530 |
| b_f | δf₁ + δf₂ | -0.013077 |
| b_i | δi₁ + δi₂ | -0.072199 |
| b_c | δc̃₁ + δc̃₂ | -0.396290 |
| b_o | δo₁ + δo₂ | -0.082908 |

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

→ Berikutnya: [Tahap 8.1: Alur Sel GRU (Gambar 7) dengan Model Mini](tahap_08_01_alur_sel_gru.md)
