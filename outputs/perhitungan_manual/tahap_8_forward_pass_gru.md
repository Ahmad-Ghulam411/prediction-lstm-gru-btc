# Perhitungan Manual - Tahap 8_forward_pass_gru: Forward Pass GRU (Jendela Pertama Data Uji)

## Rumus yang diverifikasi (konvensi Keras, `reset_after=True`)

$$\mathbf{z}_t = \sigma\!\left(\mathbf{x}_t\mathbf{W}_z + \mathbf{h}_{t-1}\mathbf{U}_z
+ \mathbf{b}_z^{(in)} + \mathbf{b}_z^{(rec)}\right)$$

$$\mathbf{r}_t = \sigma\!\left(\mathbf{x}_t\mathbf{W}_r + \mathbf{h}_{t-1}\mathbf{U}_r
+ \mathbf{b}_r^{(in)} + \mathbf{b}_r^{(rec)}\right)$$

$$\tilde{\mathbf{h}}_t = \tanh\!\left(\mathbf{x}_t\mathbf{W}_h + \mathbf{b}_h^{(in)}
+ \mathbf{r}_t \odot \left(\mathbf{h}_{t-1}\mathbf{U}_h + \mathbf{b}_h^{(rec)}\right)\right)$$

$$\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1}
+ \left(1-\mathbf{z}_t\right) \odot \tilde{\mathbf{h}}_t
\qquad
\hat{y} = \mathbf{h}_T\mathbf{W}_y + b_y$$

Urutan penyimpanan bobot pada Keras: **z, r, h**. Keadaan awal
$\mathbf{h}_0 = \mathbf{0}$ (GRU tidak memiliki *cell state*).

## Perhitungan

```
(a) PEMECAHAN BOBOT KERAS SESUAI URUTAN z, r, h

Matriks gabungan kernel W      : (10, 90)
Matriks gabungan recurrent U   : (30, 90)
Matriks gabungan bias b        : (2, 90)  (2 baris: in dan rec)
Jumlah unit (neuron) n_unit    : 30

  Gerbang                     W (fitur x unit)     U (unit x unit)      b_in     b_rec
  z (update gate)                     (10, 30)            (30, 30)     (30,)     (30,)
  r (reset gate)                      (10, 30)            (30, 30)     (30,)     (30,)
  h (kandidat hidden)                 (10, 30)            (30, 30)     (30,)     (30,)

Contoh nilai bobot untuk NEURON KE-1 (kolom pertama tiap matriks):

  Gerbang z (update)
    W[:,1] (10 bobot masukan) = [-0.004894, 0.005574, -0.035411, -0.024312, 0.053985, -0.125109, -0.004999, -0.387106, 0.032508, -0.225453]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.082263, -0.149116, 0.878828, 0.098953, -0.085613, 0.049559, ...]  (panjang 30)
    b_in[1]                   = +0.211800
    b_rec[1]                  = +0.211800
  Gerbang r (reset)
    W[:,1] (10 bobot masukan) = [-0.466489, -0.363806, -0.198020, -0.155058, -0.359717, -0.476919, -0.418683, -0.101683, -0.253078, -0.349738]  (panjang 10)
    U[:,1] (bobot rekuren)    = [0.175974, -0.100850, -0.045173, 0.047396, 0.446955, -0.087433, ...]  (panjang 30)
    b_in[1]                   = -0.249689
    b_rec[1]                  = -0.249689
  Gerbang h (kandidat)
    W[:,1] (10 bobot masukan) = [0.398500, -0.016380, 0.119456, 0.269774, -0.034389, -0.098092, -0.104376, -0.147950, -0.151735, 0.017280]  (panjang 10)
    U[:,1] (bobot rekuren)    = [-0.033276, -0.034510, 0.099536, -0.080319, -0.098821, -0.027472, ...]  (panjang 30)
    b_in[1]                   = +0.000974
    b_rec[1]                  = +0.007737

(b) FORWARD PASS MANUAL DENGAN NUMPY — JENDELA PERTAMA DATA UJI

Masukan  : X_uji[0] berbentuk (7, 10) (2025-12-11 s.d. 2025-12-17)
Target    : Close Price 2025-12-18 = 86,132.82 USD
Keadaan awal: h_0 = vektor nol (30 elemen); GRU tidak memiliki cell state

==================================================================
RINCIAN TIMESTEP t = 1  (tanggal 2025-12-11)
==================================================================
  x_1 (10 fitur ternormalisasi)     = [0.680495, 0.297207, 1.112427, 1.035626, 0.507183, 0.393311, 0.235063, 0.373105, 0.336586, 0.053572]  (panjang 10)
  h_0 (hidden state sebelumnya)     = [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, ...]  (panjang 30)

  --- RINCIAN NEURON KE-1 ---
  Gerbang z (update gate):
    pra-aktivasi = x_t.W[:,1] + h_(t-1).U[:,1] + b_in[1] + b_rec[1]
                 = -0.234813 +0.000000 +0.211800 +0.211800
                 = +0.188787
    setelah sigmoid = 0.547057
  Gerbang r (reset gate):
    pra-aktivasi = x_t.W[:,1] + h_(t-1).U[:,1] + b_in[1] + b_rec[1]
                 = -1.416727 +0.000000 -0.249689 -0.249689
                 = -1.916105
    setelah sigmoid = 0.128297

  Kandidat hidden state (reset_after=True):
    jalur rekuren = h_(t-1).U_h[:,1] + b_h_rec[1]
                  = +0.000000 +0.007737 = +0.007737
    pra-aktivasi  = x_t.W_h[:,1] + b_h_in[1] + r[1] * jalur_rekuren
                  = +0.492676 +0.000974 + 0.128297 * +0.007737
                  = +0.494642
    h~[1] = tanh(+0.494642) = +0.457893

  h_1[1] = z[1] * h_0[1] + (1 - z[1]) * h~[1]
           = 0.547057 * +0.000000 + 0.452943 * +0.457893
           = +0.000000 + +0.207399
           = +0.207399

  --- VEKTOR LENGKAP TIMESTEP INI ---
    z_1  = [0.547057, 0.074918, 0.722493, 0.636247, 0.847008, 0.180844, ...]  (panjang 30)
    r_1  = [0.128297, 0.331986, 0.347244, 0.201945, 0.546221, 0.143642, ...]  (panjang 30)
    h~_1 = [0.457893, 0.099916, -0.137106, 0.246517, 0.113892, 0.386166, ...]  (panjang 30)
    h_1  = [0.207399, 0.092430, -0.038048, 0.089672, 0.017425, 0.316330, ...]  (panjang 30)

==================================================================
RINCIAN TIMESTEP t = 2  (tanggal 2025-12-12)
==================================================================
  x_2 (10 fitur ternormalisasi)     = [0.685718, 0.292879, 1.110696, 0.904121, 0.447173, 0.433900, 0.272197, 0.353912, 0.255213, 0.052495]  (panjang 10)
  h_1 (hidden state sebelumnya)     = [0.207399, 0.092430, -0.038048, 0.089672, 0.017425, 0.316330, ...]  (panjang 30)

  --- RINCIAN NEURON KE-1 ---
  Gerbang z (update gate):
    pra-aktivasi = x_t.W[:,1] + h_(t-1).U[:,1] + b_in[1] + b_rec[1]
                 = -0.235080 -0.148659 +0.211800 +0.211800
                 = +0.039861
    setelah sigmoid = 0.509964
  Gerbang r (reset gate):
    pra-aktivasi = x_t.W[:,1] + h_(t-1).U[:,1] + b_in[1] + b_rec[1]
                 = -1.387251 +0.002460 -0.249689 -0.249689
                 = -1.884169
    setelah sigmoid = 0.131911

  Kandidat hidden state (reset_after=True):
    jalur rekuren = h_(t-1).U_h[:,1] + b_h_rec[1]
                  = -0.156035 +0.007737 = -0.148298
    pra-aktivasi  = x_t.W_h[:,1] + b_h_in[1] + r[1] * jalur_rekuren
                  = +0.468519 +0.000974 + 0.131911 * -0.148298
                  = +0.449931
    h~[1] = tanh(+0.449931) = +0.421842

  h_2[1] = z[1] * h_1[1] + (1 - z[1]) * h~[1]
           = 0.509964 * +0.207399 + 0.490036 * +0.421842
           = +0.105766 + +0.206718
           = +0.312484

  --- VEKTOR LENGKAP TIMESTEP INI ---
    z_2  = [0.509964, 0.035180, 0.577554, 0.661301, 0.841039, 0.144350, ...]  (panjang 30)
    r_2  = [0.131911, 0.351159, 0.432788, 0.176515, 0.628059, 0.138082, ...]  (panjang 30)
    h~_2 = [0.421842, 0.035886, -0.083526, 0.248412, 0.054113, 0.352978, ...]  (panjang 30)
    h_2  = [0.312484, 0.037875, -0.057260, 0.143437, 0.023257, 0.347688, ...]  (panjang 30)

(c) RINGKASAN SELURUH TIMESTEP (neuron ke-1)

 t    Tanggal     z[1]     r[1]    h~[1]     h[1]
 1 2025-12-11 0.547057 0.128297 0.457893 0.207399
 2 2025-12-12 0.509964 0.131911 0.421842 0.312484
 3 2025-12-13 0.498756 0.133666 0.383651 0.348156
 4 2025-12-14 0.485916 0.150694 0.385784 0.367500
 5 2025-12-15 0.485834 0.145139 0.372381 0.370009
 6 2025-12-16 0.507048 0.144660 0.399035 0.384318
 7 2025-12-17 0.506857 0.147030 0.376576 0.380500

(d) LAPISAN DENSE — y_hat = h_T . W_y + b_y

  h_T (hidden state timestep terakhir, T = 7) = [0.380500, -0.016342, -0.094670, 0.198929, 0.035380, 0.332692, ...]  (panjang 30)
  W_y (bobot Dense) = [0.256136, 0.098238, -0.066118, 0.121973, 0.076804, 0.173387, ...]  (panjang 30)
  b_y (bias Dense)  = +0.042936

  Perkalian suku demi suku (enam suku pertama):
    h_T[1] * W_y[1] = +0.380500 * +0.256136 = +0.097460
    h_T[2] * W_y[2] = -0.016342 * +0.098238 = -0.001605
    h_T[3] * W_y[3] = -0.094670 * -0.066118 = +0.006259
    h_T[4] * W_y[4] = +0.198929 * +0.121973 = +0.024264
    h_T[5] * W_y[5] = +0.035380 * +0.076804 = +0.002717
    h_T[6] * W_y[6] = +0.332692 * +0.173387 = +0.057685
    ... (24 suku lainnya)
  Jumlah seluruh suku = +0.604745
  y_hat = +0.604745 + +0.042936 = 0.64768150   (skala ternormalisasi)

(e) PEMBANDINGAN DENGAN model.predict() DARI KERAS

  Hasil manual NumPy   : 0.6476814983
  Hasil model.predict(): 0.6476814747
  Selisih absolut      : 2.366e-08

  Dalam skala USD:
    Prediksi manual      : 88,762.92 USD
    Prediksi Keras       : 88,762.92 USD
    Harga aktual 2025-12-18 : 86,132.82 USD

  OK Hasil manual SAMA dengan hasil model (toleransi 1e-5)

(f) BUKTI NUMERIK — RUMUS BUKU TEKS (CHO DKK., 2014) MEMBERI HASIL BERBEDA

Rumus Cho dkk. (2014):
  h~_t = tanh( x_t.W_h + (r_t * h_(t-1)).U_h )      <- reset SEBELUM perkalian
  h_t  = (1 - z_t) * h_(t-1) + z_t * h~_t            <- peran z tertukar

  y_hat dengan rumus Keras (reset_after=True) : 0.6476814983
  y_hat dengan rumus buku teks Cho dkk.       : 0.5364631926
  Selisih                                     : 1.112183e-01
  Dalam USD: Keras = 88,762.92 USD, Cho dkk. = 77,841.64 USD

Kesimpulan: kedua rumus sama-sama sah sebagai GRU, tetapi menghasilkan
angka yang berbeda untuk himpunan bobot yang sama. Karena bobot di sini
dilatih oleh Keras, perhitungan manual WAJIB memakai konvensi Keras.
```

## Kesimpulan

Prediksi hasil hitung tangan dengan NumPy sebesar `0.6476814983`
(ternormalisasi) atau **88,762.92 USD**, sedangkan
`model.predict()` menghasilkan `0.6476814747` dengan selisih hanya
2.37e-08 — jauh di bawah toleransi $10^{-5}$.

Sebaliknya, memakai rumus buku teks Cho dkk. (2014) pada himpunan bobot yang
sama menghasilkan `0.5364631926`, yaitu berbeda
1.11e-01. Hal ini membuktikan pentingnya memakai
konvensi Keras pada perhitungan manual skripsi.
