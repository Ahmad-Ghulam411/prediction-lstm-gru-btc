# Perhitungan Manual - Tahap 11: Metrik Evaluasi (RMSE, MAE, MAPE, Akurasi Arah)

## Rumus

$$RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\left(y_t - \hat{y}_t\right)^2}
\qquad
MAE = \frac{1}{n}\sum_{t=1}^{n}\left|y_t - \hat{y}_t\right|$$

$$MAPE = \frac{100\%}{n}\sum_{t=1}^{n}\left|\frac{y_t - \hat{y}_t}{y_t}\right|
\qquad
DA = \frac{100\%}{n}\sum_{t=1}^{n}\mathbb{1}\!\left[\operatorname{sign}(y_t - y_{t-1})
= \operatorname{sign}(\hat{y}_t - y_{t-1})\right]$$

dengan $n = 226$ hari data uji.

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Jumlah hari uji  n = 226

==================================================================
MODEL LSTM
==================================================================
  Hari Tanggal           Aktual y   Prediksi y_hat   e = y - y_hat               e^2   |e/y| x 100%
  1    2025-12-18       86,132.82        88,235.09       -2,102.27      4,419,556.89        2.4407%
  2    2025-12-19       85,458.35        86,359.86         -901.51        812,712.37        1.0549%
  3    2025-12-20       88,098.82        85,962.64        2,136.17      4,563,242.32        2.4247%
  ...  ...                    ...              ...             ...               ...            ...
  226  2026-07-31       64,720.99        62,368.96        2,352.03      5,532,057.75        3.6341%

  CARA MERATA-RATAKAN (tiga suku pertama ditampilkan, sisanya dijumlahkan):

  RMSE  (urutan wajib: kuadratkan -> rata-ratakan -> akarkan)
    Langkah 1 (kuadratkan): e^2 = 4,419,556.89, 812,712.37, 4,563,242.32, ...
    Langkah 2 (jumlahkan) : SUM e^2 = 945,609,870.08
    Langkah 3 (rata-rata) : SUM e^2 / n = 945,609,870.08 / 226 = 4,184,114.47
    Langkah 4 (akarkan)   : RMSE = akar(4,184,114.47) = 2,045.51 USD

  MAE
    |e| = 2,102.27, 901.51, 2,136.17, ...
    SUM |e| = 344,816.41
    MAE = 344,816.41 / 226 = 1,525.74 USD

  MAPE
    |e/y| x 100% = 2.4407%, 1.0549%, 2.4247%, ...
    SUM = 475.9864%
    MAPE = 475.9864% / 226 = 2.1061%

  AKURASI ARAH
    Hari         y(t-1)          y(t)      y_hat(t)   arah aktual   arah prediksi   cocok?
    1         87,835.39     86,132.82     88,235.09         TURUN            NAIK    TIDAK
    2         86,132.82     85,458.35     86,359.86         TURUN            NAIK    TIDAK
    3         85,458.35     88,098.82     85,962.64          NAIK            NAIK       YA
    Jumlah arah benar = 111 dari 226 hari
    DA = 111 / 226 x 100% = 49.1150%

  (b) PEMBANDINGAN DENGAN FUNGSI metrik.py — MODEL LSTM
    Metrik                        Manual      Fungsi library       Selisih
    RMSE (USD)              2,045.510809        2,045.510809      0.00e+00
    MAE (USD)               1,525.736327        1,525.736327      0.00e+00
    MAPE (%)                    2.106134            2.106134      0.00e+00
    Akurasi Arah (%)           49.115044           49.115044      0.00e+00
    OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)

==================================================================
MODEL GRU
==================================================================
  Hari Tanggal           Aktual y   Prediksi y_hat   e = y - y_hat               e^2   |e/y| x 100%
  1    2025-12-18       86,132.82        88,762.92       -2,630.10      6,917,405.87        3.0535%
  2    2025-12-19       85,458.35        87,246.46       -1,788.11      3,197,335.71        2.0924%
  3    2025-12-20       88,098.82        87,038.05        1,060.77      1,125,222.95        1.2041%
  ...  ...                    ...              ...             ...               ...            ...
  226  2026-07-31       64,720.99        62,211.63        2,509.36      6,296,877.19        3.8772%

  CARA MERATA-RATAKAN (tiga suku pertama ditampilkan, sisanya dijumlahkan):

  RMSE  (urutan wajib: kuadratkan -> rata-ratakan -> akarkan)
    Langkah 1 (kuadratkan): e^2 = 6,917,405.87, 3,197,335.71, 1,125,222.95, ...
    Langkah 2 (jumlahkan) : SUM e^2 = 912,748,765.13
    Langkah 3 (rata-rata) : SUM e^2 / n = 912,748,765.13 / 226 = 4,038,711.35
    Langkah 4 (akarkan)   : RMSE = akar(4,038,711.35) = 2,009.65 USD

  MAE
    |e| = 2,630.10, 1,788.11, 1,060.77, ...
    SUM |e| = 331,821.47
    MAE = 331,821.47 / 226 = 1,468.24 USD

  MAPE
    |e/y| x 100% = 3.0535%, 2.0924%, 1.2041%, ...
    SUM = 462.7540%
    MAPE = 462.7540% / 226 = 2.0476%

  AKURASI ARAH
    Hari         y(t-1)          y(t)      y_hat(t)   arah aktual   arah prediksi   cocok?
    1         87,835.39     86,132.82     88,762.92         TURUN            NAIK    TIDAK
    2         86,132.82     85,458.35     87,246.46         TURUN            NAIK    TIDAK
    3         85,458.35     88,098.82     87,038.05          NAIK            NAIK       YA
    Jumlah arah benar = 116 dari 226 hari
    DA = 116 / 226 x 100% = 51.3274%

  (b) PEMBANDINGAN DENGAN FUNGSI metrik.py — MODEL GRU
    Metrik                        Manual      Fungsi library       Selisih
    RMSE (USD)              2,009.654535        2,009.654535      0.00e+00
    MAE (USD)               1,468.236576        1,468.236576      0.00e+00
    MAPE (%)                    2.047584            2.047584      0.00e+00
    Akurasi Arah (%)           51.327434           51.327434      0.00e+00
    OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)
```

## Kesimpulan

| Metrik | LSTM | GRU |
|---|---|---|
| RMSE (USD) | 2,045.51 | 2,009.65 |
| MAE (USD) | 1,525.74 | 1,468.24 |
| MAPE (%) | 2.1061 | 2.0476 |
| Kategori MAPE | Sangat baik (<10%) | Sangat baik (<10%) |
| Akurasi arah (%) | 49.1150 | 51.3274 |

Seluruh metrik hasil hitung tangan cocok dengan fungsi pada `src/metrik.py`
sampai toleransi $10^{-5}$.
