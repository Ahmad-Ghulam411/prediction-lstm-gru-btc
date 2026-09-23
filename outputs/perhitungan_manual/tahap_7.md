# Perhitungan Manual - Tahap 7: Jumlah Parameter Model LSTM

## Rumus

$$\text{Param}_{\text{LSTM}} = 4 \times \left(n_{unit} \times
\left(n_{unit} + n_{fitur}\right) + n_{unit}\right)
\qquad
\text{Param}_{\text{Dense}} = n_{unit} \times 1 + 1$$

dengan $n_{unit} = 40$ dan $n_{fitur} = 10$.

## Perhitungan

```
(a) PERHITUNGAN MANUAL LANGKAH DEMI LANGKAH

Diketahui: n_unit = 40 neuron, n_fitur = 10 variabel

LAPISAN LSTM
  Rumus: Param_LSTM = 4 x ( n_unit x (n_unit + n_fitur) + n_unit )

  Angka 4 = empat himpunan bobot gerbang: i (input), f (forget),
            c (kandidat cell), dan o (output).

  Per satu gerbang:
    bobot masukan  W : n_unit x n_fitur = 40 x 10 = 400
    bobot rekuren  U : n_unit x n_unit  = 40 x 40 = 1,600
    bias           b : n_unit           = 40
    jumlah per gerbang                  = 2,040

  Param_LSTM = 4 x 2,040 = 8,160

  Pemeriksaan dengan bentuk rumus penuh:
    4 x ( 40 x (40 + 10) + 40 )
    = 4 x ( 40 x 50 + 40 )
    = 4 x ( 2,000 + 40 )
    = 4 x 2,040
    = 8,160

LAPISAN DENSE
  Param_Dense = n_unit x 1 + 1 = 40 x 1 + 1 = 41

TOTAL PARAMETER = 8,160 + 41 = 8,201

(b) PEMBANDINGAN DENGAN HASIL LIBRARY (Keras)

  Lapisan                 Manual         Keras   Selisih
  LSTM                     8,160         8,160         0
  Dense                       41            41         0
  TOTAL                    8,201         8,201         0

  OK Hasil manual SAMA dengan hasil model
```

## Kesimpulan

Model LSTM terbaik (40 neuron) memiliki
**8,160 parameter** pada lapisan rekuren dan
41 parameter pada lapisan Dense, sehingga totalnya
**8,201 parameter** — sama dengan keluaran
`model.summary()` dari Keras.
