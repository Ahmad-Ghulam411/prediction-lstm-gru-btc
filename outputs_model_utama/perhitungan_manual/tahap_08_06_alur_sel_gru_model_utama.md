# Perhitungan Manual - Tahap 8.6: Alur Sel GRU pada Model Utama (Gambar 7)

Bagian ini menjelaskan alur **Gambar 7** (struktur sel GRU, subbab 1.5.9) memakai model GRU
yang benar-benar dipakai penelitian. Polanya sama dengan LSTM di Tahap 7.6, tetapi selnya
lebih ramping: dua gerbang dan tanpa cell state terpisah.

**Semua angka di sini berasal dari model GRU utama penelitian**, yaitu model terbaik hasil
tuning: **30 neuron**, masukan **7 hari × 10 fitur**, batch
**32**, dan dilatih 500 epoch. Tidak ada model mini atau angka
karangan. Karena angkanya banyak (3,811 parameter), triknya begini: kita ikuti
**satu neuron (neuron ke-1)** dan **satu sampel** secara rinci sampai bisa dicek dengan
kalkulator, sedangkan neuron dan sampel lainnya dihitung komputer dengan rumus yang sama
lalu dicocokkan dengan Keras.

**Daftar isi**

- [1. Model dan Data yang Dipakai](#1-model-dan-data-yang-dipakai)
- [2. Cara Membaca Gambar 7](#2-cara-membaca-gambar-7)
- [3. Time Step t = 1, Neuron ke-1](#3-time-step-t--1-neuron-ke-1)
- [4. Time Step t = 2: Memori Mulai Bekerja](#4-time-step-t--2-memori-mulai-bekerja)
- [5. Tujuh Time Step Neuron ke-1](#5-tujuh-time-step-neuron-ke-1)
- [6. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)](#6-dari-seluruh-neuron-ke-prediksi-lapisan-dense)
- [7. Ringkasan Satu Langkah GRU](#7-ringkasan-satu-langkah-gru)
- [8. Catatan untuk Naskah Skripsi](#8-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).

## 1. Model dan Data yang Dipakai

**Arsitektur:** Input (7 × 10) → GRU (30 neuron) → Dense (1), total
**3,811 parameter** (rinciannya di [Tahap 8.4 (Jumlah Parameter Model GRU)](tahap_08_04_jumlah_parameter_gru.md)).

Sama seperti LSTM, yang dipakai adalah **bobot awal** model terbaik, dibuat ulang dengan
`bangun_model("GRU", 30, seed=42)` sehingga sama persis dengan bobot awal saat model ini
dilatih di Tahap 8.1:

| Bobot | Bentuk | Cara Keras mengisinya | Nilai pada model ini |
|---|---|---|---|
| kernel W (masukan) | (10, 90) = 10 fitur × 3 gerbang × 30 neuron | acak Glorot uniform, rentang ±√(6/(fan_in+fan_out)) = ±0.2449 | -0.2424 s.d. 0.2444 |
| recurrent kernel U | (30, 90) | acak ortogonal (baris-barisnya saling tegak lurus) | -0.4262 s.d. 0.3163; cek ortogonal: selisih 3.7e-07 |
| bias b | (2, 90) | semua nol (baris 1 = bias masukan, baris 2 = bias rekuren) | semuanya 0 |
| Dense W_y | (30, 1) | Glorot uniform, ±0.4399 | -0.4046 s.d. 0.4234 |
| Dense b_y | (1,) | nol | 0 |

Berbeda dengan LSTM, GRU tidak punya trik bias awal: semua biasnya mulai dari nol.

**Data:** jendela pertama data latih (`X_latih[0]`), sama dengan yang dipakai LSTM:

| Hari | Tanggal | X1 | X2 | X3 | X4 | X5 | X6 | X7 | X8 | X9 | X10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t = 1 | 2023-07-01 | 0.0541 | 0.0778 | 0.0000 | 0.0408 | 0.5139 | 0.4820 | 0.4001 | 0.2466 | 0.1195 | 0.1251 |
| t = 2 | 2023-07-02 | 0.0553 | 0.1639 | 0.0000 | 0.1535 | 0.3832 | 0.6123 | 0.2669 | 0.0935 | 0.0986 | 0.2833 |
| t = 3 | 2023-07-03 | 0.0556 | 0.0985 | 0.0000 | 0.0527 | 0.6993 | 0.5825 | 0.4506 | 0.2360 | 0.1252 | 0.1170 |
| t = 4 | 2023-07-04 | 0.0610 | 0.1059 | 0.0000 | 0.0675 | 0.6256 | 0.5379 | 0.4357 | 0.3023 | 0.1988 | 0.1418 |
| t = 5 | 2023-07-05 | 0.0572 | 0.1247 | 0.0000 | 0.1060 | 0.5074 | 0.5117 | 0.5434 | 0.4290 | 0.3667 | 0.1533 |
| t = 6 | 2023-07-06 | 0.0544 | 0.0888 | 0.0000 | 0.0586 | 0.5287 | 0.4781 | 0.5320 | 0.3412 | 0.2195 | 0.0741 |
| t = 7 | 2023-07-07 | 0.0484 | 0.1666 | 0.0000 | 0.1653 | 0.4277 | 0.5982 | 0.6295 | 0.4274 | 0.4581 | 0.1722 |

Targetnya adalah Close Price ternormalisasi tanggal **2023-07-08**:
y′ = **0.052770**, setara 0.052770 × 98,196.75 + 25,162.70 =
**30,344.53 USD**. Keadaan awal memori mengikuti Keras: h₀ = 0
untuk semua neuron.

## 2. Cara Membaca Gambar 7

| Simbol pada Gambar 7 | Arti |
|---|---|
| Kotak kuning σ (dua buah) | lapisan berbobot yang menghasilkan update gate zₜ dan reset gate rₜ |
| Kotak kuning tanh | lapisan berbobot yang menghasilkan kandidat h̃ₜ |
| Lingkaran × | perkalian elemen demi elemen (⊙), tanpa bobot |
| Lingkaran 1− | menghitung 1 − zₜ |
| Lingkaran + | penjumlahan, menghasilkan hₜ |
| Garis atas | hidden state hₜ, satu-satunya memori GRU |

Seperti di LSTM, σ berperan sebagai keran dan tanh sebagai isi informasi. Rumus dan
penjabaran keduanya (persamaan 11-13) ada di
[Tahap 7.6, bagian 3](tahap_07_06_alur_sel_lstm_model_utama.md#3-sigmoid-dan-tanh-penjabaran-persamaan-11-13).

Rumus yang dihitung mengikuti Keras (`reset_after=True`, Tahap 8.A), dengan dua bias per
gerbang:

- zₜ = σ(xₜW_z + b_z(in) + hₜ₋₁U_z + b_z(rec))  (persamaan 23)
- rₜ = σ(xₜW_r + b_r(in) + hₜ₋₁U_r + b_r(rec))  (persamaan 24)
- h̃ₜ = tanh(xₜW_h + b_h(in) + rₜ ⊙ (hₜ₋₁U_h + b_h(rec)))  (persamaan 25)
- hₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ h̃ₜ  (persamaan 26)

## 3. Time Step t = 1, Neuron ke-1

Di hari pertama h₀ = 0, jadi semua bagian memori nol dan reset gate belum berpengaruh
apa-apa. Contoh lengkap **update gate** neuron ke-1 untuk hari 2023-07-01:

| Kode | Fitur | x₁ | W_z[k, 1] | x₁ × W_z |
|---|---|---|---|---|
| X1 | Close Price (USD) | 0.054063 | 0.175476 | 0.009487 |
| X2 | Miners Revenue (USD) | 0.077789 | 0.022169 | 0.001725 |
| X3 | Difficulty | 0.000000 | (-0.048002) | 0.000000 |
| X4 | Hash Rate | 0.040839 | (-0.191756) | (-0.007831) |
| X5 | Median Confirmation Time | 0.513902 | (-0.050970) | (-0.026194) |
| X6 | Average Block Size | 0.481952 | (-0.070206) | (-0.033836) |
| X7 | Total Unique Addresses | 0.400071 | (-0.238462) | (-0.095402) |
| X8 | Transaction per Block | 0.246577 | (-0.093799) | (-0.023129) |
| X9 | Confirmed Transaction | 0.119541 | 0.215843 | 0.025802 |
| X10 | Cost % per Transaction | 0.125072 | (-0.072847) | (-0.009111) |
|  | **jumlah** |  |  | **-0.158489** |

Reset gate dan kandidat dihitung dengan cara yang sama:

| Bagian | Σ x₁·W | b(in) | Σ h₀·U + b(rec) | Hasil |
|---|---|---|---|---|
| update z₁ | -0.158489 | 0 | 0 | σ(-0.158489) = **0.460461** |
| reset r₁ | -0.185674 | 0 | 0 | σ(-0.185674) = **0.453715** |
| kandidat h̃₁ | -0.147593 | 0 | r₁ × 0 = 0 | tanh(-0.147593) = **-0.146531** |

```
Campuran memori lama dan baru (persamaan 26):
  h₁ = z₁ × h₀ + (1 − z₁) × h̃₁ = 0.460461 × 0 + 0.539539 × (-0.146531) = -0.079059
```

Perhatikan bedanya dengan LSTM: di GRU, **z dan (1 − z) saling terikat**. Kalau z = 0.460,
berarti 46.0% memori lama dipertahankan dan otomatis hanya 54.0% kandidat
baru yang masuk. Di LSTM, forget gate dan input gate bisa bergerak sendiri-sendiri.

## 4. Time Step t = 2: Memori Mulai Bekerja

Mulai hari kedua, neuron ke-1 menerima h₁ dari **seluruh 30 neuron**:
h₁ = [-0.079059, 0.051047, -0.004121, 0.024832, …]. Contoh bagian memori untuk update gate:

```
  h₁·U_z[:,1] = (-0.079059)×0.002427 + 0.051047×(-0.021410) + (-0.004121)×(-0.075875) + … (27 suku lagi) = -0.007695
```

| Bagian | Σ x₂·W + b(in) | Σ h₁·U + b(rec) (30 suku) | Hasil |
|---|---|---|---|
| update z₂ | -0.150381 | -0.007695 | σ(-0.158076) = **0.460563** |
| reset r₂ | -0.210527 | -0.086339 | σ(-0.296867) = **0.426324** |
| kandidat h̃₂ | -0.075561 | r₂ × 0.019582 = 0.426324 × 0.019582 = 0.008348 | tanh(-0.067213) = **-0.067112** |

```
Reset gate: dari isi memori h₁U_h + b_h(rec) = 0.019582, hanya 42.6% yang dipakai menyusun kandidat.
h₂ = z₂ × h₁ + (1 − z₂) × h̃₂ = 0.460563 × (-0.079059) + 0.539437 × (-0.067112)
   = (-0.036412) + (-0.036203) = -0.072614
```

## 5. Tujuh Time Step Neuron ke-1

| t | Tanggal | zₜ | rₜ | h̃ₜ | hₜ |
|---|---|---|---|---|---|
| 1 | 2023-07-01 | 0.460461 | 0.453715 | -0.146531 | -0.079059 |
| 2 | 2023-07-02 | 0.460563 | 0.426324 | -0.067112 | -0.072614 |
| 3 | 2023-07-03 | 0.453422 | 0.417400 | -0.146023 | -0.112738 |
| 4 | 2023-07-04 | 0.457243 | 0.414176 | -0.150738 | -0.133363 |
| 5 | 2023-07-05 | 0.456041 | 0.418666 | -0.198930 | -0.169029 |
| 6 | 2023-07-06 | 0.452327 | 0.411649 | -0.174412 | -0.171977 |
| 7 | 2023-07-07 | 0.449188 | 0.411182 | -0.224005 | -0.200635 |

Update gate neuron ke-1 berada di kisaran 0.449 sampai 0.461, jadi
setiap hari sekitar 45% memori lama dibawa ke hari berikutnya. Porsi isi memori hari
pertama yang masih tersisa di hari ke-7 kira-kira 0.9%. Nilai-nilai ini
akan berubah setelah bobot dilatih.

## 6. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)

```
  ŷ′ = Σ h₇[j] × W_y[j] + b_y   (j = 1 … 30)
     h₇[1] × W_y[1] = (-0.200635) × 0.277997 = (-0.055776)
     h₇[2] × W_y[2] = 0.196763 × 0.120049 = 0.023621
     h₇[3] × W_y[3] = (-0.078034) × 0.001027 = (-0.000080)
     h₇[4] × W_y[4] = 0.018196 × 0.161227 = 0.002934
     … (26 suku lagi)
  Jumlah 30 suku = -0.125065
  ŷ′ = -0.125065 + 0.000000 = -0.125065

  Dalam USD: 12,881.70 USD   (target 30,344.53 USD)
  Selisih dengan model.predict() Keras: 1.69e-09
```

Hasilnya cocok dengan Keras. Prediksinya meleset 17,462.83 USD dari
target; wajar karena bobotnya masih acak. Bobot inilah yang akan diperbaiki lewat siklus pelatihan di [Tahap 8.7 (Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss)](tahap_08_07_pelatihan_gru_forward_pass_dan_loss.md) sampai
[Tahap 8.9 (Pelatihan GRU Model Utama, Langkah 4: Update Adam)](tahap_08_09_pelatihan_gru_update_adam.md).

## 7. Ringkasan Satu Langkah GRU

| Urutan | Bagian Gambar 7 | Rumus | Fungsinya |
|---|---|---|---|
| 1. Update gate | σ kiri | zₜ (23) | berapa persen memori lama dipertahankan |
| 2. Reset gate | σ tengah | rₜ (24) | berapa banyak memori lama dipakai menyusun kandidat |
| 3. Kandidat | tanh | h̃ₜ (25) | usulan isi memori yang baru |
| 4. Campuran | 1−, ×, + | hₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ h̃ₜ (26) | memori baru sekaligus keluaran hari ini |
| 5. Prediksi | - | ŷ′ = h₇W_y + b_y | dipakai sekali, setelah hari terakhir |

## 8. Catatan untuk Naskah Skripsi

1. **Satu memori saja.** Pada GRU 30 neuron ini, memori setiap hari hanyalah vektor hₜ
   berisi 30 angka; tidak ada cₜ seperti pada LSTM.
2. **Gambar 7 dan `reset_after=True`.** Gambar 7 menggambar rₜ mengalikan hₜ₋₁ sebelum
   masuk tanh, sedangkan Keras mengalikan rₜ setelah hₜ₋₁ dikali U_h. Saran kalimat:

   > Gambar 7 merupakan ilustrasi konseptual; pada implementasi Keras yang digunakan dalam
   > penelitian ini (reset_after=True), reset gate diterapkan setelah perkalian hₜ₋₁ dengan
   > matriks bobot rekuren, sebagaimana persamaan (25).

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 8.4: Jumlah Parameter Model GRU](tahap_08_04_jumlah_parameter_gru.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.7: Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss](tahap_08_07_pelatihan_gru_forward_pass_dan_loss.md)
