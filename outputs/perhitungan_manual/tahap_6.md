# Perhitungan Manual - Tahap 6: Pembentukan Sliding Window

## Rumus

$$\mathbf{X}^{(i)} = \left[\mathbf{x}_{t-6}, \mathbf{x}_{t-5}, \ldots,
\mathbf{x}_{t}\right] \in \mathbb{R}^{7 \times 10}
\qquad
y^{(i)} = \text{Close}_{t+1}$$

Bentuk akhir data: $\mathbf{X} \in \mathbb{R}^{n \times 7 \times 10}$ dan
$\mathbf{y} \in \mathbb{R}^{n \times 1}$.

## Perhitungan

```
(a) SUSUNAN JENDELA PERTAMA (X_latih[0]) — NILAI TERNORMALISASI

Baris = hari t-6 s.d. t (2023-07-01 s.d. 2023-07-07)
Kolom = 10 variabel X1..X10
Target y = Close Price hari t+1 = 2023-07-08

Hari    Tanggal   Y / X1       X2       X3       X4       X5       X6       X7       X8       X9      X10
 t-6 2023-07-01 0.054063 0.077789 0.000000 0.040839 0.513902 0.481952 0.400071 0.246577 0.119541 0.125072
 t-5 2023-07-02 0.055285 0.163910 0.000000 0.153461 0.383225 0.612254 0.266889 0.093495 0.098573 0.283276
 t-4 2023-07-03 0.055557 0.098530 0.000000 0.052694 0.699259 0.582513 0.450626 0.236019 0.125150 0.117043
 t-3 2023-07-04 0.060982 0.105879 0.000000 0.067513 0.625579 0.537856 0.435673 0.302290 0.198834 0.141810
 t-2 2023-07-05 0.057152 0.124744 0.000000 0.106041 0.507414 0.511676 0.543396 0.428973 0.366709 0.153320
 t-1 2023-07-06 0.054379 0.088756 0.000000 0.058622 0.528730 0.478138 0.531970 0.341238 0.219478 0.074135
   t 2023-07-07 0.048427 0.166597 0.000000 0.165316 0.427711 0.598191 0.629480 0.427381 0.458060 0.172204

Target y_latih[0] (ternormalisasi) = 0.052770
Target dalam USD                   = 30,344.53 USD (tanggal 2023-07-08)

(b) PEMBANDINGAN DENGAN DATA ASLI (verifikasi penyusunan window)

Close Price (X1) pada jendela di atas harus sama dengan kolom close_price
ternormalisasi pada 7 baris pertama dataset:

  2023-07-01  jendela = 0.054063   dataset = 0.054063   selisih = 1.02e-09
  2023-07-02  jendela = 0.055285   dataset = 0.055285   selisih = 6.48e-10
  2023-07-03  jendela = 0.055557   dataset = 0.055557   selisih = 1.03e-09
  2023-07-04  jendela = 0.060982   dataset = 0.060982   selisih = 5.09e-10
  2023-07-05  jendela = 0.057152   dataset = 0.057152   selisih = 1.48e-09
  2023-07-06  jendela = 0.054379   dataset = 0.054379   selisih = 2.91e-10
  2023-07-07  jendela = 0.048427   dataset = 0.048427   selisih = 5.58e-10

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)
```

## Kesimpulan

Jendela pertama berisi data 2023-07-01 s.d. 2023-07-07
(7 hari x 10 variabel) dengan target harga penutupan 2023-07-08. Susunan
jendela terbukti konsisten dengan dataset ternormalisasi.
