# Perhitungan Manual - Tahap 12: Uji Diebold-Mariano LSTM vs GRU

## Hipotesis

$$H_0: E\left[d_t\right] = 0 \qquad H_1: E\left[d_t\right] \neq 0$$

dengan $d_t = e_{LSTM,t}^2 - e_{GRU,t}^2$, horizon $h = 1$, dan koreksi
Harvey-Leybourne-Newbold:

$$DM^{*} = \frac{\bar{d}}{\sqrt{\widehat{V}(\bar{d})}}
\times \sqrt{\frac{T + 1 - 2h + \frac{h(h-1)}{T}}{T}}$$

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

HIPOTESIS
  H0 : E[d_t] = 0   (akurasi LSTM dan GRU sama)
  H1 : E[d_t] != 0  (akurasi LSTM dan GRU berbeda)
  Fungsi loss : kuadrat (p = 2);  horizon h = 1;  alpha = 5%

LANGKAH 1 — Hitung d_t = e_LSTM,t^2 - e_GRU,t^2

  Hari Tanggal             e_LSTM         e_GRU          e_LSTM^2           e_GRU^2               d_t
  1    2025-12-18       -2,102.27     -2,630.10      4,419,556.89      6,917,405.87     -2,497,848.98
  2    2025-12-19         -901.51     -1,788.11        812,712.37      3,197,335.71     -2,384,623.34
  3    2025-12-20        2,136.17      1,060.77      4,563,242.32      1,125,222.95      3,438,019.37
  4    2025-12-21          653.36       -311.99        426,875.00         97,339.77        329,535.23
  5    2025-12-22          706.17       -230.16        498,681.76         52,974.86        445,706.91
  6    2025-12-23         -577.99       -539.62        334,069.97        291,191.43         42,878.54
  ...  ...                    ...           ...               ...               ...               ...
  226  2026-07-31        2,352.03      2,509.36      5,532,057.75      6,296,877.19       -764,819.45

  Jumlah pengamatan T = 226
  Banyak hari dengan d_t < 0 (LSTM lebih akurat) : 100 hari
  Banyak hari dengan d_t > 0 (GRU lebih akurat)  : 126 hari

LANGKAH 2 — Hitung rata-rata d_bar
  SUM d_t = 32,861,104.94
  d_bar = 32,861,104.94 / 226 = 145,403.12

LANGKAH 3 — Hitung autokovarians gamma_0
  gamma_0 = (1/T) * SUM (d_t - d_bar)^2
  Tiga suku pertama (d_t - d_bar)^2 :
    t=1: (-2,497,848.98 - 145,403.12)^2 = 6,986,781,671,641.36
    t=2: (-2,384,623.34 - 145,403.12)^2 = 6,401,033,878,131.78
    t=3: (3,438,019.37 - 145,403.12)^2 = 10,841,321,781,627.73
  SUM (d_t - d_bar)^2 = 2,324,328,756,613,639.50
  gamma_0 = 2,324,328,756,613,639.50 / 226 = 10,284,640,515,989.55

  Karena h = 1, penjumlahan autokovarians k = 1..h-1 KOSONG,
  sehingga tidak ada suku gamma_k tambahan.

LANGKAH 4 — Hitung varians d_bar
  V(d_bar) = gamma_0 / T = 10,284,640,515,989.55 / 226 = 45,507,258,920.31
  akar V(d_bar) = 213,324.30

LANGKAH 5 — Hitung statistik DM
  DM = d_bar / akar(V(d_bar)) = 145,403.12 / 213,324.30
  DM = 0.681606

LANGKAH 6 — Koreksi Harvey-Leybourne-Newbold (sampel kecil)
  faktor = akar( (T + 1 - 2h + h(h-1)/T) / T )
         = akar( (226 + 1 - 2(1) + 0) / 226 )
         = akar( 225 / 226 ) = 0.99778516
  DM* = DM x faktor = 0.681606 x 0.99778516 = 0.680096

LANGKAH 7 — Hitung p-value dua sisi (distribusi t, df = T - 1)
  df = 226 - 1 = 225
  p = 2 x [ 1 - F_t(0.680096) ] = 0.49714264
  Nilai kritis t(0.975, 225) = +/- 1.970563

(b) PEMBANDINGAN DENGAN FUNGSI metrik.diebold_mariano()

  Besaran                             Manual                Fungsi       Selisih
  d_bar                     145,403.11922538      145,403.11922538      0.00e+00
  gamma_0                     1.02846405e+13        1.02846405e+13      0.00e+00
  V(d_bar)                    4.55072589e+10        4.55072589e+10      0.00e+00
  DM                              0.68160597            0.68160597      0.00e+00
  Faktor HLN                      0.99778516            0.99778516      0.00e+00
  DM*                             0.68009632            0.68009632      0.00e+00
  p-value                         0.49714264            0.49714264      0.00e+00

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)
```

## Kesimpulan

| Komponen | Hasil |
|---|---|
| Statistik $DM^{*}$ | 0.680096 |
| Derajat bebas | 225 |
| Nilai kritis ($\alpha = 5\%$) | $\pm$ 1.9706 |
| $p$-value | 0.49714264 |
| Keputusan | **Gagal tolak $H_0$** |

Perbedaan akurasi LSTM dan GRU **tidak signifikan** pada taraf 5%, sehingga kedua model dapat dianggap memiliki akurasi yang setara. Dengan demikian GRU yang strukturnya lebih sederhana mampu menyamai LSTM.
