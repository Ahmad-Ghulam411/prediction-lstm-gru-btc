# Perhitungan Manual - Tahap 8.1: Alur Sel GRU (Gambar 7) dengan Model Mini

Subbagian ini menjelaskan alur **Gambar 7** (struktur sel GRU) pada subbab 1.5.9
langkah demi langkah: apa yang mengalir di setiap garis, apa yang dikerjakan setiap
kotak dan lingkaran, dan apa fungsinya. Setiap langkah diberi contoh angka dari model
mini GRU pada **time step t = 2**.

**Daftar isi**

- [1. Model Mini yang Dipakai](#1-model-mini-yang-dipakai)
- [2. Cara Membaca Gambar 7](#2-cara-membaca-gambar-7)
- [3. Alur Gambar 7 Langkah demi Langkah](#3-alur-gambar-7-langkah-demi-langkah)
- [4. Fungsi Update Gate dan Reset Gate](#4-fungsi-update-gate-dan-reset-gate)
- [5. Ringkasan Satu Langkah GRU](#5-ringkasan-satu-langkah-gru)
- [6. Catatan untuk Naskah Skripsi](#6-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Model Mini yang Dipakai

Supaya setiap langkah punya angka nyata yang bisa diikuti dengan tangan, seluruh
subbagian perhitungan manual pelatihan GRU memakai **model mini** yang sama:
**1 neuron, 1 fitur, 2 time step, dan 1 sampel**. Rumus dan urutan langkahnya
identik dengan model penelitian (10 fitur, 7 time step,
batch 32); yang berbeda hanya jumlah angkanya.

**Data** (harga penutupan ternormalisasi, angka ilustrasi):

- hari 1: x₁ = 0.5
- hari 2: x₂ = 0.6
- target hari 3: y′ = 0.7

**Keadaan awal:** h₀ = 0, sama seperti bawaan Keras.

**Bobot awal** dibuat bulat agar mudah dihitung. Keras sebenarnya mengisi bobot secara
acak (Glorot uniform untuk W, ortogonal untuk U), tetapi aturan biasnya diikuti:
semua bias = 0.

| Gerbang | W (bobot masukan) | U (bobot rekuren) | b(in) (bias masukan) | b(rec) (bias rekuren) |
|---|---|---|---|---|
| update (z) | 0.5 | 0.4 | 0 | 0 |
| reset (r) | 0.6 | 0.3 | 0 | 0 |
| kandidat (h̃) | 0.7 | 0.2 | 0 | 0 |
| dense | W_y = 0.8 | - | b_y = 0 | - |

Setiap gerbang GRU punya **dua bias**, yaitu bias masukan b(in) dan bias rekuren
b(rec), karena Keras memakai `reset_after=True` (dijelaskan di [Tahap 8.12 (Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias)](tahap_08_12_bias_gru.md)).
Hasil time step t = 1 yang dipakai di contoh berikut adalah h₁ = 0.147273.
Perhitungan lengkap kedua time step ada di [Tahap 8.2 (Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss)](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md).

## 2. Cara Membaca Gambar 7

| Simbol pada Gambar 7 | Arti |
|---|---|
| Kotak kuning σ (kiri dan tengah) | lapisan dengan bobot dan bias yang menghasilkan update gate zₜ dan reset gate rₜ |
| Kotak kuning tanh | lapisan dengan bobot dan bias yang menghasilkan kandidat h̃ₜ |
| Lingkaran × | perkalian elemen demi elemen (⊙), tanpa bobot |
| Lingkaran 1− | menghitung 1 − zₜ |
| Lingkaran + | penjumlahan, menghasilkan hₜ |
| Garis atas | hidden state hₜ, satu-satunya memori GRU |

Seperti pada LSTM, σ (0 sampai 1) berperan sebagai **keran** dan tanh (-1 sampai 1)
sebagai **isi informasi**; penjelasan lengkapnya di [Tahap 7.1, bagian 2](tahap_07_01_alur_sel_lstm.md#2-cara-membaca-gambar).
GRU **hanya punya satu garis memori**, yaitu hₜ. Tidak ada cₜ terpisah dan tidak ada
output gate.

## 3. Alur Gambar 7 Langkah demi Langkah

**Langkah 0 — masukan.**

- hₜ₋₁ masuk dari **kiri atas**, lalu bercabang: tetap di garis atas, turun lewat garis
  vertikal kiri ke jalur bawah, dan bercabang ke tengah menuju lingkaran × milik rₜ.
- xₜ masuk dari **bawah** ke jalur horizontal bawah.
- Jalur bawah membawa hₜ₋₁ dan xₜ ke ketiga kotak kuning: σ (z), σ (r), dan tanh.
- Contoh: h₁ = 0.147273 (hasil t = 1) dan x₂ = 0.6.

**Langkah 1 — update gate (kotak σ kiri) menghasilkan zₜ** (persamaan 23).

- z₂ = σ(W_z × x₂ + U_z × h₁ + b_z(in) + b_z(rec))
  = σ(0.5 × 0.6 + 0.4 × 0.147273 + 0 + 0) = σ(0.358909) = **0.588776**.
- Panah zₜ naik dan **bercabang dua**: ke **lingkaran × kiri atas**, mengalikan hₜ₋₁
  menjadi z₂ × h₁ = 0.588776 × 0.147273 = **0.086711**; dan ke
  kanan menuju **lingkaran "1−"**, menghasilkan 1 − z₂ = **0.411224**.

**Langkah 2 — reset gate (kotak σ tengah) menghasilkan rₜ** (persamaan 24).

- r₂ = σ(W_r × x₂ + U_r × h₁ + b_r(in) + b_r(rec))
  = σ(0.6 × 0.6 + 0.3 × 0.147273 + 0 + 0) = σ(0.404182) = **0.599692**.
- Panah rₜ naik ke **lingkaran × tengah**, tempat ia mengalikan memori lama yang datang
  dari kiri.

**Langkah 3 — kandidat (kotak tanh) menghasilkan h̃ₜ** (persamaan 25).

- Cabang memori lama melewati lingkaran × milik rₜ sebelum masuk tanh. Pada Keras
  (`reset_after=True`), yang dikalikan rₜ adalah **bagian rekuren**
  qₜ = hₜ₋₁U_h + b_h(rec): q₂ = 0.2 × 0.147273 + 0 = 0.029455,
  sehingga r₂ × q₂ = 0.599692 × 0.029455 = 0.017664.
- xₜ masuk dari bawah sebagai **bagian masukan**:
  x₂W_h + b_h(in) = 0.7 × 0.6 + 0 = 0.420000.
- h̃₂ = tanh(0.420000 + 0.017664) = tanh(0.437664) = **0.411706**.

**Langkah 4 — pencampuran** (persamaan 26).

- h̃ₜ naik ke **lingkaran × kanan** dan dikalikan (1 − zₜ):
  0.411224 × 0.411706 = 0.169303.
- Hasilnya naik ke **lingkaran +** dan dijumlahkan dengan zₜ × hₜ₋₁ dari kiri:
  h₂ = 0.086711 + 0.169303 = **0.256014**.

**Langkah 5 — keluaran.** h₂ keluar ke **kanan** (ke langkah berikutnya) dan ke
**atas** (keluaran waktu t). Karena t = 2 adalah langkah terakhir, h₂ masuk ke dense:
ŷ′ = 0.8 × 0.256014 + 0 = **0.204811**.

## 4. Fungsi Update Gate dan Reset Gate

**Update gate zₜ bekerja seperti penggeser (slider) pencampur.** zₜ mendekati 1 berarti
hₜ hampir sama dengan hₜ₋₁ (memori lama dipertahankan); zₜ mendekati 0 berarti hₜ hampir
sama dengan h̃ₜ (diganti informasi baru). Porsi lama dan porsi baru **selalu berjumlah 1**.
Pada contoh: 58.9% memori lama + 41.1% kandidat baru.

**Reset gate rₜ mengatur seberapa banyak masa lalu dipakai untuk menyusun usulan
baru.** rₜ mendekati 0 berarti kandidat disusun hampir hanya dari xₜ ("mulai dari nol");
rₜ mendekati 1 berarti masa lalu ikut diperhitungkan penuh. Pada t = 1, q₁ = 0 karena
h₀ = 0, jadi reset gate belum berpengaruh.

**Jalur langsung zₜ ⊙ hₜ₋₁** (lingkaran × kiri atas ke lingkaran +) adalah "jalan tol
gradien" pada GRU, padanan cell state pada LSTM. Di [Tahap 8.3 (Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time)](tahap_08_03_simulasi_bptt_gru.md) terlihat
**97.3%** sinyal kesalahan dari t = 2 ke t = 1 mengalir lewat
jalur ini.

## 5. Ringkasan Satu Langkah GRU

| Urutan | Bagian Gambar 7 | Perhitungan | Hasil (t = 2) |
|---|---|---|---|
| 1. Update gate | kotak σ kiri | z₂ = σ(a) | 0.588776 |
| 2. Reset gate | kotak σ tengah | r₂ = σ(a) | 0.599692 |
| 3. Kandidat | lingkaran × tengah, kotak tanh | h̃₂ = tanh(x₂W_h + b_h(in) + r₂q₂) | r₂q₂ = 0.017664; h̃₂ = 0.411706 |
| 4. Pencampuran | lingkaran × kiri, 1−, × kanan, + | h₂ = z₂h₁ + (1 − z₂)h̃₂ | 0.086711 + 0.169303 = 0.256014 |
| 5. Dense | - | ŷ′ = W_y h₂ + b_y | 0.204811 |

Prediksi ŷ′ = 0.204811 masih jauh dari target 0.7. Langkah perbaikannya
dihitung mulai dari [Tahap 8.2 (Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss)](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md).

## 6. Catatan untuk Naskah Skripsi

1. **Klaim tentang rumus asli Cho et al. (2014) perlu diperbaiki** (subbab 1.5.9,
   halaman 17 draf). Persamaan (7) pada makalah aslinya berbunyi
   hⱼ⟨t⟩ = zⱼ hⱼ⟨t−1⟩ + (1 − zⱼ) h̃ⱼ⟨t⟩, yaitu **sama** dengan persamaan (26) (konvensi
   Keras), sehingga peran zₜ **tidak tertukar**. Bentuk dengan peran zₜ tertukar,
   hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, berasal dari Chung et al. (2014). Perbedaan dengan
   Cho et al. (2014) hanya pada posisi reset gate. Saran kalimat:

   > Rumus GRU pada makalah asli Cho et al. (2014) sedikit berbeda, yaitu gerbang
   > reset diterapkan sebelum perkalian dengan matriks bobot rekuren,
   > h̃ₜ = tanh(xₜW_h + (rₜ ⊙ hₜ₋₁)U_h). Adapun bentuk
   > hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, dengan peran zₜ tertukar, digunakan oleh
   > Chung et al. (2014).

2. **Gambar 7 dan pengaturan `reset_after=True`.** Gambar 7 menggambar rₜ mengalikan
   hₜ₋₁ sebelum kotak tanh (bentuk Cho et al.), sedangkan persamaan (25) mengalikan rₜ
   dengan (hₜ₋₁U_h + b_h(rec)) setelah perkalian matriks. Fungsinya sama, jadi gambar
   tidak perlu diubah. Cukup tambahkan kalimat berikut, atau beri label garis dari
   lingkaran × rₜ ke kotak tanh dengan rₜ ⊙ (hₜ₋₁U_h + b_h(rec)) dan garis xₜ ke kotak
   tanh dengan xₜW_h + b_h(in):

   > Gambar 7 merupakan ilustrasi konseptual; urutan perhitungan yang digunakan
   > mengikuti persamaan (25).

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.12: Bias pada LSTM: Peran dan Cara Menghitungnya](tahap_07_12_bias_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 8.2: Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss](tahap_08_02_simulasi_forward_pass_dan_loss_gru.md)
