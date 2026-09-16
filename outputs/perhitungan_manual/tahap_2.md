# Perhitungan Manual - Tahap 2: Rata-rata dan Standar Deviasi Close Price

## Rumus

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\qquad
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(x_i - \bar{x}\right)^2}$$

dengan $x_i$ = harga penutupan Bitcoin hari ke-$i$ (USD) dan $n$ = 1,127.

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Jumlah data          n = 1,127

RATA-RATA:  x_bar = (1/n) * (x_1 + x_2 + ... + x_n)

  x_bar = (1/1127) * ( 30,471.50  +  30,591.53  +  30,618.25  +  31,150.92  +  30,774.87  + ... + 64,720.99 )
  Jumlah seluruh x     = 82,516,521.90
  x_bar = 82,516,521.90 / 1127
  x_bar = 73,217.854392 USD

STANDAR DEVIASI (sampel, pembagi n-1):
  s = akar( SUM (x_i - x_bar)^2 / (n - 1) )

  Lima suku pertama (x_i - x_bar)^2 :
    i=1: (30,471.50 - 73,217.85)^2 = (-42,746.35)^2 = 1,827,250,813.82
    i=2: (30,591.53 - 73,217.85)^2 = (-42,626.32)^2 = 1,817,003,531.19
    i=3: (30,618.25 - 73,217.85)^2 = (-42,599.60)^2 = 1,814,726,294.37
    i=4: (31,150.92 - 73,217.85)^2 = (-42,066.93)^2 = 1,769,626,969.16
    i=5: (30,774.87 - 73,217.85)^2 = (-42,442.98)^2 = 1,801,406,924.12
  SUM (x_i - x_bar)^2  = 779,439,786,641.80
  Varians  s^2 = 779,439,786,641.80 / (1127 - 1) = 692,220,059.18
  Std dev  s   = akar(692,220,059.18) = 26,310.075241 USD

(b) PEMBANDINGAN DENGAN HASIL LIBRARY (pandas)

                                        Manual                pandas         Selisih
  Rata-rata (USD)                73,217.854392         73,217.854392        0.00e+00
  Standar deviasi (USD)          26,310.075241         26,310.075241        0.00e+00

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)
```

## Kesimpulan

Rata-rata Close Price hasil hitung tangan sebesar
**73,217.85 USD** dan standar deviasinya
**26,310.08 USD**, identik dengan keluaran `pandas`
(`.mean()` dan `.std()`) sampai toleransi $10^{-5}$.
