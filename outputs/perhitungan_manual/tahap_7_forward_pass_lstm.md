# Perhitungan Manual - Tahap 7: Forward Pass LSTM (Jendela Pertama Data Uji)

## Rumus yang diverifikasi

$$\mathbf{f}_t = \sigma(\mathbf{x}_t\mathbf{W}_f + \mathbf{h}_{t-1}\mathbf{U}_f + \mathbf{b}_f)
\qquad
\mathbf{i}_t = \sigma(\mathbf{x}_t\mathbf{W}_i + \mathbf{h}_{t-1}\mathbf{U}_i + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{x}_t\mathbf{W}_c + \mathbf{h}_{t-1}\mathbf{U}_c + \mathbf{b}_c)
\qquad
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

$$\mathbf{o}_t = \sigma(\mathbf{x}_t\mathbf{W}_o + \mathbf{h}_{t-1}\mathbf{U}_o + \mathbf{b}_o)
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\qquad
\hat{y} = \mathbf{h}_T\mathbf{W}_y + b_y$$

Urutan penyimpanan bobot pada Keras: **i, f, c, o**.
Keadaan awal $\mathbf{h}_0 = \mathbf{c}_0 = \mathbf{0}$.

## Perhitungan

```
(a) PEMECAHAN BOBOT KERAS SESUAI URUTAN i, f, c, o

Matriks gabungan kernel W          : (10, 160)
Matriks gabungan recurrent U       : (40, 160)
Vektor gabungan bias b             : (160,)
Jumlah unit (neuron) n_unit        : 40

Setelah dipecah menjadi empat bagian:
  Gerbang                       W (fitur x unit)     U (unit x unit)    b (unit)
  i (input gate)                        (10, 40)            (40, 40)       (40,)
  f (forget gate)                       (10, 40)            (40, 40)       (40,)
  c (kandidat cell)                     (10, 40)            (40, 40)       (40,)
  o (output gate)                       (10, 40)            (40, 40)       (40,)

Contoh nilai bobot untuk NEURON KE-1 (kolom pertama tiap matriks):

  Gerbang i (input)
    W[:,1] (10 bobot masukan) = [0.413355, -0.016684, 0.125940, -0.200919, -0.059908, -0.082207, -0.518425, -0.265667, -0.408494, -0.317435]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.209807, 0.034556, -0.026079, 0.043449, 0.147345, -0.091875, ...]  (panjang 40)
    b[1]   (bias)             = -0.342494
  Gerbang f (forget)
    W[:,1] (10 bobot masukan) = [-0.066825, 0.108273, 0.047586, -0.092290, -0.222868, -0.322515, -0.626940, -0.091566, -0.190472, -0.258931]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.070839, -0.146409, 0.080002, 0.074698, 0.175408, -0.094144, ...]  (panjang 40)
    b[1]   (bias)             = 0.681141
  Gerbang c (kandidat)
    W[:,1] (10 bobot masukan) = [0.158535, 0.309700, -0.162692, -0.077519, 0.197350, 0.000810, 0.068815, -0.092497, 0.074651, 0.007119]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.170110, -0.031152, -0.162305, 0.061940, -0.099702, 0.019462, ...]  (panjang 40)
    b[1]   (bias)             = -0.011402
  Gerbang o (output)
    W[:,1] (10 bobot masukan) = [0.321498, -0.126170, 0.379633, -0.219034, -0.206085, -0.152426, -0.604805, -0.106231, -0.281759, -1.053337]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.379613, 0.036639, -0.027632, 0.086849, 0.161525, -0.072132, ...]  (panjang 40)
    b[1]   (bias)             = -0.373906

Catatan: bias gerbang f (forget) bernilai sekitar 1 pada awal pelatihan
karena Keras memakai unit_forget_bias=True, yaitu strategi agar memori
jangka panjang tidak langsung terlupakan di awal pelatihan.

(b) FORWARD PASS MANUAL DENGAN NUMPY — JENDELA PERTAMA DATA UJI

Masukan  : X_uji[0] berbentuk (7, 10) (2025-12-11 s.d. 2025-12-17)
Target    : Close Price 2025-12-18 = 86,132.82 USD
Keadaan awal: h_0 = vektor nol (40 elemen), c_0 = vektor nol (40 elemen)

==================================================================
RINCIAN TIMESTEP t = 1  (tanggal 2025-12-11)
==================================================================
  x_1 (10 fitur ternormalisasi)     = [0.680495, 0.297207, 1.112427, 1.035626, 0.507183, 0.393311, 0.235063, 0.373105, 0.336586, 0.053572]  (panjang 10)
  h_0 (hidden state sebelumnya)     = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...]  (panjang 40)
  c_0 (cell state sebelumnya)       = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...]  (panjang 40)

  --- RINCIAN NEURON KE-1 ---
  Gerbang i (input gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.229851 +0.000000 -0.342494
                   = -0.572345
    setelah sigmoid = sigmoid(-0.572345) = 0.360696
  Gerbang f (forget gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.555336 +0.000000 +0.681141
                   = +0.125804
    setelah sigmoid = sigmoid(+0.125804) = 0.531410
  Gerbang c~ (kandidat cell):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = +0.046247 +0.000000 -0.011402
                   = +0.034844
    setelah tanh    = tanh(+0.034844) = 0.034830
  Gerbang o (output gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.120786 +0.000000 -0.373906
                   = -0.494692
    setelah sigmoid = sigmoid(-0.494692) = 0.378789

  c_1[1] = f[1] * c_0[1] + i[1] * c~[1]
           = 0.531410 * +0.000000 + 0.360696 * +0.034830
           = +0.000000 + +0.012563
           = +0.012563

  h_1[1] = o[1] * tanh(c_1[1])
           = 0.378789 * tanh(+0.012563)
           = 0.378789 * +0.012563
           = +0.004759

  --- VEKTOR LENGKAP TIMESTEP INI ---
    i_1  = [0.360696, 0.710295, 0.333379, 0.397552, 0.396200, 0.205154, ...]  (panjang 40)
    f_1  = [0.531410, 0.741350, 0.649090, 0.537142, 0.618303, 0.432504, ...]  (panjang 40)
    c~_1 = [0.034830, 0.218212, -0.399007, 0.282658, 0.385605, 0.062226, ...]  (panjang 40)
    o_1  = [0.378789, 0.622312, 0.496259, 0.324859, 0.474663, 0.226654, ...]  (panjang 40)
    c_1  = [0.012563, 0.154995, -0.133021, 0.112371, 0.152777, 0.012766, ...]  (panjang 40)
    h_1  = [0.004759, 0.095690, -0.065626, 0.036352, 0.071958, 0.002893, ...]  (panjang 40)

==================================================================
RINCIAN TIMESTEP t = 2  (tanggal 2025-12-12)
==================================================================
  x_2 (10 fitur ternormalisasi)     = [0.685718, 0.292879, 1.110696, 0.904121, 0.447173, 0.433900, 0.272197, 0.353912, 0.255213, 0.052495]  (panjang 10)
  h_1 (hidden state sebelumnya)     = [0.004759, 0.095690, -0.065626, 0.036352, 0.071958, 0.002893, ...]  (panjang 40)
  c_1 (cell state sebelumnya)       = [0.012563, 0.154995, -0.133021, 0.112371, 0.152777, 0.012766, ...]  (panjang 40)

  --- RINCIAN NEURON KE-1 ---
  Gerbang i (input gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.181728 +0.200970 -0.342494
                   = -0.323251
    setelah sigmoid = sigmoid(-0.323251) = 0.419884
  Gerbang f (forget gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.549561 +0.035781 +0.681141
                   = +0.167360
    setelah sigmoid = sigmoid(+0.167360) = 0.541743
  Gerbang c~ (kandidat cell):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = +0.042649 -0.094830 -0.011402
                   = -0.063584
    setelah tanh    = tanh(-0.063584) = -0.063499
  Gerbang o (output gate):
    pra-aktivasi z = x_t.W[:,1] + h_(t-1).U[:,1] + b[1]
                   = -0.080591 +0.217060 -0.373906
                   = -0.237437
    setelah sigmoid = sigmoid(-0.237437) = 0.440918

  c_2[1] = f[1] * c_1[1] + i[1] * c~[1]
           = 0.541743 * +0.012563 + 0.419884 * -0.063499
           = +0.006806 + -0.026662
           = -0.019856

  h_2[1] = o[1] * tanh(c_2[1])
           = 0.440918 * tanh(-0.019856)
           = 0.440918 * -0.019853
           = -0.008754

  --- VEKTOR LENGKAP TIMESTEP INI ---
    i_2  = [0.419884, 0.692157, 0.353386, 0.358863, 0.338851, 0.163681, ...]  (panjang 40)
    f_2  = [0.541743, 0.716315, 0.632003, 0.506990, 0.569449, 0.385296, ...]  (panjang 40)
    c~_2 = [-0.063499, 0.045088, -0.076225, 0.210145, 0.347743, 0.145364, ...]  (panjang 40)
    o_2  = [0.440918, 0.573761, 0.495503, 0.327482, 0.447522, 0.198779, ...]  (panjang 40)
    c_2  = [-0.019856, 0.142233, -0.111006, 0.132384, 0.204832, 0.028712, ...]  (panjang 40)
    h_2  = [-0.008754, 0.081062, -0.054779, 0.043102, 0.090406, 0.005706, ...]  (panjang 40)

(c) RINGKASAN SELURUH TIMESTEP (neuron ke-1)

 t    Tanggal     i[1]     f[1]     c~[1]     o[1]      c[1]      h[1]
 1 2025-12-11 0.360696 0.531410  0.034830 0.378789  0.012563  0.004759
 2 2025-12-12 0.419884 0.541743 -0.063499 0.440918 -0.019856 -0.008754
 3 2025-12-13 0.417889 0.555581 -0.103225 0.437682 -0.054168 -0.023685
 4 2025-12-14 0.442823 0.577537 -0.116985 0.458542 -0.083088 -0.038012
 5 2025-12-15 0.431293 0.561556 -0.113300 0.477979 -0.095524 -0.045520
 6 2025-12-16 0.425508 0.568267 -0.141336 0.477070 -0.114423 -0.054351
 7 2025-12-17 0.426827 0.561746 -0.141349 0.478256 -0.124608 -0.059288

(d) LAPISAN DENSE — y_hat = h_T . W_y + b_y

  h_T (hidden state timestep terakhir, T = 7) = [-0.059288, 0.033720, -0.036911, 0.038392, 0.102880, 0.008137, ...]  (panjang 40)
  W_y (bobot Dense) = [0.164646, 0.297487, -0.051887, 0.160470, 0.265961, 0.041397, ...]  (panjang 40)
  b_y (bias Dense)  = +0.081419

  Perkalian suku demi suku (enam suku pertama):
    h_T[1] * W_y[1] = -0.059288 * +0.164646 = -0.009762
    h_T[2] * W_y[2] = +0.033720 * +0.297487 = +0.010031
    h_T[3] * W_y[3] = -0.036911 * -0.051887 = +0.001915
    h_T[4] * W_y[4] = +0.038392 * +0.160470 = +0.006161
    h_T[5] * W_y[5] = +0.102880 * +0.265961 = +0.027362
    h_T[6] * W_y[6] = +0.008137 * +0.041397 = +0.000337
    ... (34 suku lainnya)
  Jumlah seluruh suku = +0.560887
  y_hat = +0.560887 + +0.081419 = 0.64230636   (skala ternormalisasi)

(e) PEMBANDINGAN DENGAN model.predict() DARI KERAS

  Hasil manual NumPy   : 0.6423063572
  Hasil model.predict(): 0.6423063278
  Selisih absolut      : 2.940e-08

  Dalam skala USD:
    Prediksi manual      : 88,235.10 USD
    Prediksi Keras       : 88,235.09 USD
    Harga aktual 2025-12-18 : 86,132.82 USD

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)

Artinya rumus LSTM pada Bab III benar-benar rumus yang dijalankan Keras:
gerbang lupa, gerbang masukan, kandidat cell, pembaruan cell state,
gerbang keluaran, hidden state, dan lapisan Dense — seluruhnya terbukti.
```

## Kesimpulan

Prediksi hasil hitung tangan dengan NumPy sebesar
`0.6423063572` (ternormalisasi) atau **88,235.10 USD**,
sedangkan `model.predict()` Keras menghasilkan `0.6423063278`. Selisih
keduanya hanya 2.94e-08, jauh di bawah toleransi $10^{-5}$, sehingga rumus
LSTM yang ditulis pada Bab III terbukti identik dengan implementasi Keras.
