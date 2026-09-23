# Perhitungan Manual - Tahap 5: Normalisasi Min-Max Close Price

## Rumus

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

dengan $x_{min}$ dan $x_{max}$ **hanya** dihitung dari data latih
(hari 2023-07-01 s.d. 2025-09-18).

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Variabel : close_price (Close Price, USD)
Tanggal  : 2023-07-01 (hari pertama data latih)

Rumus:  x' = (x - x_min) / (x_max - x_min)

  x      = 30,471.50 USD
  x_min  = 25,162.70 USD   (harga terendah DATA LATIH)
  x_max  = 123,359.45 USD   (harga tertinggi DATA LATIH)

  Pembilang = x - x_min = 30,471.50 - 25,162.70 = 5,308.80
  Penyebut  = x_max - x_min = 123,359.45 - 25,162.70 = 98,196.75

  x' = 5,308.80 / 98,196.75
  x' = 0.054063

(b) PEMBANDINGAN DENGAN HASIL LIBRARY (sklearn MinMaxScaler)

  Hasil manual            : 0.054063
  Hasil MinMaxScaler      : 0.054063
  Selisih absolut         : 1.39e-17

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)

(c) UJI BALIK — denormalisasi mengembalikan nilai asli
  x = x' * (x_max - x_min) + x_min
  x = 0.054063 * 98,196.75 + 25,162.70
  x = 30,471.50 USD   (nilai asli: 30,471.50 USD)
  OK Denormalisasi mengembalikan nilai asli dengan tepat.
```

## Kesimpulan

Harga penutupan 2023-07-01 sebesar 30,471.50 USD setara dengan
nilai ternormalisasi **0.054063**, sama dengan keluaran
`MinMaxScaler` sampai toleransi $10^{-5}$. Proses balik (denormalisasi)
mengembalikan angka 30,471.50 USD, yaitu nilai aslinya.
