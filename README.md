# Perbandingan Metode LSTM dan GRU dalam Memprediksi Harga Bitcoin Berdasarkan Informasi Blockchain

Repositori ini berisi seluruh kode, data, dan keluaran untuk skripsi S1 Ilmu
Aktuaria dengan judul di atas. Penelitian membandingkan dua arsitektur
*recurrent neural network* — **LSTM** (*Long Short-Term Memory*) dan **GRU**
(*Gated Recurrent Unit*) — dalam memprediksi harga penutupan harian Bitcoin
berdasarkan sepuluh variabel informasi *blockchain*.

**Pertanyaan penelitian.** Apakah GRU yang strukturnya lebih sederhana mampu
menyamai atau mengungguli akurasi LSTM, dan bagaimana perbandingan efisiensinya
(jumlah parameter dan waktu latih)?

---

## 1. Ringkasan Penelitian

| Aspek | Keterangan |
|---|---|
| Aset | Bitcoin (BTC), harga penutupan harian dalam USD |
| Periode | 1 Juli 2023 s.d. 31 Juli 2026 (1.127 hari) |
| Sumber data | Blockchain.com Charts API (`https://api.blockchain.info/charts/...`) |
| Variabel | 1 target (Close Price) + 10 fitur (Close Price berperan ganda sebagai X1) |
| Panjang *window* | 7 hari (`LOOKBACK = 7`) |
| Pembagian data | 80% latih (10% terakhirnya menjadi validasi) dan 20% uji, kronologis |
| Normalisasi | Min-Max, `fit` **hanya** pada data latih |
| Grid *tuning* | Neuron {10, 20, 30, 40, 50} × Epoch {100, 500, 1000} = 15 model per arsitektur |
| Pemilihan model | RMSE **data validasi** (bukan data uji) |
| Metrik evaluasi | RMSE, MAE, MAPE, dan akurasi arah |
| Uji signifikansi | Diebold-Mariano (*loss* kuadrat, h = 1, koreksi Harvey-Leybourne-Newbold) |
| Uji kestabilan | 5 *seed* berbeda: 42, 7, 21, 100, 2024 |

---

## 2. Variabel Penelitian

| Kode | Variabel | Satuan | Nama *chart* Blockchain.com |
|---|---|---|---|
| Y / X1 | Close Price | USD | `market-price` |
| X2 | Miners Revenue | USD | `miners-revenue` |
| X3 | Difficulty | — | `difficulty` |
| X4 | Hash Rate | TH/s | `hash-rate` |
| X5 | Median Confirmation Time | menit | `median-confirmation-time` |
| X6 | Average Block Size | MB | `avg-block-size` |
| X7 | Total Unique Addresses | alamat | `n-unique-addresses` |
| X8 | Transaction per Block | transaksi/blok | `n-transactions-per-block` |
| X9 | Confirmed Transaction | transaksi | `n-transactions` |
| X10 | Cost % per Transaction | % | `cost-per-transaction-percent` |

**Catatan ketersediaan data.** Kesepuluh *chart* telah diverifikasi tersedia
pada frekuensi **harian** untuk seluruh periode penelitian, sehingga **tidak ada
variabel yang diganti**. Dua hal yang perlu diketahui:

- **Difficulty (X3)** bernilai konstan selama beberapa hari berturut-turut karena
  jaringan Bitcoin menyesuaikan tingkat kesulitan setiap 2.016 blok (± dua
  minggu). Nilainya berbentuk **fungsi tangga**, bukan data hilang.
- Beberapa *chart* kehilangan 3–5 tanggal akibat *gap* pelaporan API (total 27
  sel kosong atau 0,24% dari seluruh sel). Tanggal tersebut ditangani dengan
  ***forward fill*** — memakai nilai hari sebelumnya — karena variabel
  *blockchain* berubah lambat dan cara ini tidak memakai informasi masa depan
  sehingga bebas *look-ahead bias*.

---

## 3. Struktur Folder

```
prediction-lstm-gru-btc/
├── data/
│   └── dataset_bitcoin.csv             # dataset gabungan (tanggal + 10 variabel)
├── notebooks/
│   └── btc_lstm_vs_gru.ipynb           # NOTEBOOK UTAMA, sudah dieksekusi
├── src/                                # fungsi pendukung
│   ├── data.py                         # pengambilan data Blockchain.com API
│   ├── metrik.py                       # RMSE, MAE, MAPE, akurasi arah, Diebold-Mariano
│   └── utils.py                        # header tahap, format angka, penyimpanan tabel/gambar
├── outputs/
│   ├── tabel/                          # seluruh tabel (.csv dan .xlsx)
│   ├── gambar/                         # seluruh gambar (.png, 300 dpi)
│   └── perhitungan_manual/             # perhitungan manual per tahap (.md)
├── tools/
│   ├── bangun_notebook.py              # penyusun notebook (opsional)
│   └── uji_cepat.py                    # uji cepat pipeline dengan grid kecil
├── requirements.txt
└── README.md
```

---

## 4. Cara Menjalankan

### 4.1 Persiapan lingkungan

```bash
# (disarankan) buat lingkungan virtual terlebih dahulu
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 4.2 Menjalankan notebook

**Cara A — lewat Jupyter (disarankan untuk membaca tahap demi tahap):**

```bash
jupyter lab notebooks/btc_lstm_vs_gru.ipynb
```

Jalankan sel dari atas ke bawah. Setiap tahap mencetak header, tabel, gambar,
perhitungan manual, dan *Ringkasan Tahap*.

**Cara B — eksekusi ulang seluruh notebook dari awal:**

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    notebooks/btc_lstm_vs_gru.ipynb
```

> **Perkiraan waktu.** Sekitar **75–95 menit** pada CPU 4 *core*. Rinciannya:
> *tuning* LSTM ± 21 menit, *tuning* GRU ± 24 menit, uji kestabilan ± 28 menit,
> sisanya pengolahan data dan pembuatan gambar. Bila memakai GPU, waktunya jauh
> lebih singkat.

### 4.3 Uji cepat (bila hanya ingin memastikan kode berjalan)

```bash
python tools/uji_cepat.py          # menyusun skrip uji dengan grid kecil
cd notebooks && python ../tools/_skrip_uji_cepat.py
```

Perintah ini menjalankan **seluruh alur** notebook dengan grid kecil (2 neuron ×
2 epoch) sehingga selesai dalam beberapa menit. Berguna untuk memeriksa
pemasangan pustaka tanpa menunggu pelatihan penuh.

### 4.4 Mengubah pengaturan penelitian

Seluruh pengaturan berada pada satu blok `CONFIG` di awal notebook:

```python
CONFIG = {
    "ASET": "BTC",
    "TANGGAL_MULAI": "2023-07-01",
    "TANGGAL_AKHIR": "2026-07-31",
    "RASIO_LATIH": 0.8,
    "RASIO_VALIDASI_DARI_LATIH": 0.1,
    "LOOKBACK": 7,
    "NEURON_GRID": [10, 20, 30, 40, 50],
    "EPOCH_GRID": [100, 500, 1000],
    "BATCH_SIZE": 32,
    "JUMLAH_LAYER_REKUREN": 1,
    "OPTIMIZER": "adam",
    "LEARNING_RATE": 0.001,
    "SEED": 42,
    "SEED_ULANGAN": [42, 7, 21, 100, 2024],
}
```

### 4.5 Mengunduh ulang data

Notebook memakai `data/dataset_bitcoin.csv` bila berkas itu ada. Untuk mengunduh
ulang dari Blockchain.com API, hapus berkas tersebut lalu jalankan kembali
notebook:

```bash
rm data/dataset_bitcoin.csv
```

---

## 5. Tahapan di Dalam Notebook

| Tahap | Judul | Perhitungan manual yang diverifikasi |
|---|---|---|
| 0 | Persiapan lingkungan | — |
| 1 | Pengumpulan data | — |
| 2 | Eksplorasi data | rata-rata dan standar deviasi Close Price |
| 3 | Pembersihan data | — |
| 4 | Pembagian data (sebelum normalisasi) | — |
| 5 | Normalisasi Min-Max | $x' = (x - x_{min})/(x_{max} - x_{min})$ |
| 6 | Pembentukan *sliding window* | susunan jendela pertama |
| 7 | Model LSTM | jumlah parameter **dan** *forward pass* lengkap |
| 8 | Model GRU | jumlah parameter **dan** *forward pass* lengkap |
| 9 | Perbandingan arsitektur dan kestabilan | reproduksibilitas *seed* |
| 10 | Denormalisasi | $x = x'(x_{max} - x_{min}) + x_{min}$ |
| 11 | Evaluasi model | RMSE, MAE, MAPE, akurasi arah |
| 12 | Uji Diebold-Mariano | $d_t$, $\bar{d}$, $\gamma_0$, DM, DM\*, *p-value* |
| 13 | Visualisasi | — |
| 14 | Kesimpulan otomatis | — |

Setiap perhitungan manual dilakukan **dua kali**: (a) langkah demi langkah
dengan angka asli, lalu (b) dibandingkan dengan hasil *library* dan dicek dengan
`assert` pada toleransi $10^{-5}$. Versi teksnya tersimpan di
`outputs/perhitungan_manual/tahap_X.md` dan siap disalin ke Bab III.

---

## 6. Jaminan Metodologis

- **Tidak ada kebocoran data** (*data leakage*). Pembagian data dilakukan
  **sebelum** normalisasi, `MinMaxScaler` di-*fit* **hanya** pada data latih, dan
  *hyperparameter* dipilih dari **data validasi** — bukan data uji.
- **Perbandingan adil.** LSTM dan GRU memakai fungsi `bangun_model()` dan
  `latih_model()` yang sama persis, dengan data, *window*, grid, *batch size*,
  *optimizer*, *learning rate*, dan *seed* identik. Satu-satunya perbedaan adalah
  jenis lapisan rekuren.
- **Data runtun waktu tidak diacak.** Seluruh proses memakai `shuffle=False`.
- **Reproduksibel.** *Seed* `random`, `numpy`, dan `tensorflow` dikunci, dan mode
  deterministik TensorFlow (`enable_op_determinism`) diaktifkan. Notebook
  memverifikasi sendiri bahwa pelatihan ulang dengan *seed* 42 menghasilkan
  prediksi yang sama persis.
- **Hanya dua model** yang dibangun dan dibandingkan: LSTM dan GRU.

---

## 7. Catatan Pembacaan Angka

Seluruh angka dicetak dengan **pemisah ribuan tanda koma** dan **pemisah desimal
tanda titik** (konvensi keluaran komputer). Contoh: `61,234.56` berarti enam
puluh satu ribu dua ratus tiga puluh empat koma lima enam dolar AS. Nilai USD
memakai 2 desimal, nilai ternormalisasi memakai 4–6 desimal.

---

## 8. Keterbatasan Penelitian

1. Model hanya memakai informasi **7 hari terakhir**, sehingga lonjakan harga
   yang mendadak baru "tersusul" pada hari berikutnya.
2. Penskala hanya mengenal rentang harga **data latih**. Pada periode uji ketika
   harga memecahkan rekor, nilai ternormalisasi keluar dari $[0,1]$ dan model
   cenderung **meremehkan** (*underestimate*) harga sebenarnya — keterbatasan
   wajar dari ekstrapolasi.
3. Hanya satu lapisan rekuren yang dipakai (`JUMLAH_LAYER_REKUREN = 1`) agar
   seluruh perhitungan dapat diverifikasi secara manual.

---

## 9. Rujukan Utama

- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
  Schwenk, H., & Bengio, Y. (2014). *Learning Phrase Representations using RNN
  Encoder–Decoder for Statistical Machine Translation*. EMNLP.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy*.
  Journal of Business & Economic Statistics, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). *Testing the equality of
  prediction mean squared errors*. International Journal of Forecasting, 13(2),
  281–291.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural
  Computation, 9(8), 1735–1780.
- Lewis, C. D. (1982). *Industrial and Business Forecasting Methods*. Butterworths.
