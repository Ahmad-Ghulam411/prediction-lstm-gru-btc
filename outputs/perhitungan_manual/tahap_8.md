# Perhitungan Manual - Tahap 8: Jumlah Parameter Model GRU

## Rumus

$$\text{Param}_{\text{GRU}} = 3 \times \left(n_{unit} \times
\left(n_{unit} + n_{fitur}\right) + 2 \times n_{unit}\right)
\qquad
\text{Param}_{\text{Dense}} = n_{unit} \times 1 + 1$$

dengan $n_{unit} = 30$ dan $n_{fitur} = 10$. Suku
$2 \times n_{unit}$ muncul karena Keras memakai `reset_after=True` sehingga
bias berukuran $(2, 3n_{unit})$.

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Diketahui: n_unit = 30 neuron, n_fitur = 10 variabel

LAPISAN GRU (Keras, reset_after=True)
  Rumus: Param_GRU = 3 x ( n_unit x (n_unit + n_fitur) + 2 x n_unit )

  Angka 3 = tiga himpunan bobot: z (update), r (reset), h (kandidat).
  Angka 2 pada suku bias = dua himpunan bias (masukan dan rekuren)
  yang muncul karena Keras memakai reset_after=True.

  Per satu gerbang:
    bobot masukan  W : n_unit x n_fitur = 30 x 10 = 300
    bobot rekuren  U : n_unit x n_unit  = 30 x 30 = 900
    bias           b : 2 x n_unit       = 2 x 30 = 60
    jumlah per gerbang                  = 1,260

  Param_GRU = 3 x 1,260 = 3,780

  Pemeriksaan dengan bentuk rumus penuh:
    3 x ( 30 x (30 + 10) + 2 x 30 )
    = 3 x ( 30 x 40 + 60 )
    = 3 x ( 1,200 + 60 )
    = 3 x 1,260
    = 3,780

LAPISAN DENSE
  Param_Dense = n_unit x 1 + 1 = 30 x 1 + 1 = 31

TOTAL PARAMETER = 3,780 + 31 = 3,811

(b) PEMBANDINGAN DENGAN HASIL LIBRARY (Keras)

  Lapisan                 Manual         Keras   Selisih
  GRU                      3,780         3,780         0
  Dense                       31            31         0
  TOTAL                    3,811         3,811         0

  OK Hasil manual SAMA dengan hasil model
```

## Kesimpulan

Model GRU terbaik (30 neuron) memiliki
**3,780 parameter** pada lapisan rekuren dan
31 parameter pada lapisan Dense, totalnya
**3,811 parameter** — sama dengan keluaran
`model.summary()`.
