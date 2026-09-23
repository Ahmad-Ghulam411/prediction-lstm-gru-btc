# Perhitungan Manual - Tahap 10: Denormalisasi Prediksi ke Skala USD

## Rumus

$$x = x' \times \left(x_{max} - x_{min}\right) + x_{min}$$

dengan $x_{min} = 25,162.70$ USD dan
$x_{max} = 123,359.45$ USD, keduanya berasal dari data latih.

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Tanggal prediksi : 2025-12-18 (hari pertama data uji)

Rumus:  x = x' * (x_max - x_min) + x_min

  x_min = 25,162.70 USD   (harga terendah DATA LATIH)
  x_max = 123,359.45 USD   (harga tertinggi DATA LATIH)
  x_max - x_min = 98,196.75 USD

  MODEL LSTM
    x' (prediksi ternormalisasi) = 0.642306
    x = 0.642306 * 98,196.75 + 25,162.70
    x = 63,072.39 + 25,162.70
    x = 88,235.09 USD   <- hasil manual

  MODEL GRU
    x' (prediksi ternormalisasi) = 0.647681
    x = 0.647681 * 98,196.75 + 25,162.70
    x = 63,600.22 + 25,162.70
    x = 88,762.92 USD   <- hasil manual

(b) PEMBANDINGAN DENGAN HASIL LIBRARY (scaler.inverse_transform)

  Model         Manual (USD)   inverse_transform (USD)       Selisih
  LSTM             88,235.09                 88,235.09      0.00e+00
  GRU              88,762.92                 88,762.92      0.00e+00

  Harga AKTUAL 2025-12-18 = 86,132.82 USD
    Galat LSTM  = 86,132.82 - 88,235.09 = -2,102.27 USD (overestimate)
    Galat GRU   = 86,132.82 - 88,762.92 = -2,630.10 USD (overestimate)

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)
```

## Kesimpulan

Denormalisasi manual menghasilkan angka yang identik dengan
`penskala.inverse_transform()` sampai toleransi $10^{-5}$, sehingga seluruh
prediksi pada Tahap 11 dapat dinyatakan dalam USD dengan yakin.
