# Perhitungan Manual - Tahap 7.3: Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time

Subbagian ini melanjutkan [Tahap 7.2 (Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss)](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md). Loss model mini sudah
diketahui, L = 0.276429. Sekarang kesalahan itu ditelusuri **mundur** untuk
mencari gradien setiap bobot dan bias: lapisan dense dulu, lalu ke dalam sel pada t = 2,
lalu mundur ke t = 1 (*through time*).

**Daftar isi**

- [1. Rumus Turunan Dasar](#1-rumus-turunan-dasar)
- [2. Backward di Lapisan Dense](#2-backward-di-lapisan-dense)
- [3. Backward di Dalam Sel, t = 2](#3-backward-di-dalam-sel-t--2)
- [4. Mengirim Kesalahan ke t = 1 (Inti Through Time)](#4-mengirim-kesalahan-ke-t--1-inti-through-time)
- [5. Backward di Dalam Sel, t = 1](#5-backward-di-dalam-sel-t--1)
- [6. Gradien Total Setiap Bobot dan Bias](#6-gradien-total-setiap-bobot-dan-bias)
- [7. Pemeriksaan Gradien dengan Turunan Numerik](#7-pemeriksaan-gradien-dengan-turunan-numerik)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Rumus Turunan Dasar

Seluruh backward pass hanya mengulang beberapa rumus turunan berikut. Polanya selalu
sama: **kalikan sinyal kesalahan dengan turunan lokal, lalu teruskan ke kiri (ke langkah
sebelumnya).**

| Rumus | Bentuk | Contoh | Dipakai di |
|---|---|---|---|
| Aturan pangkat | d(u²)/du = 2u | d(x²)/dx = 2x | ∂L/∂ŷ′ |
| Fungsi linear | d(a·u + b)/du = a | d(3x + 5)/dx = 3 | pra-aktivasi dan dense: ∂a/∂W = x, ∂a/∂U = hₜ₋₁, ∂a/∂b = 1 |
| Aturan perkalian | ∂(u·v)/∂u = v | ∂(3u)/∂u = 3 | h = o·tanh(c), c = f·cₜ₋₁ + i·c̃ |
| Aturan rantai | ∂L/∂a = ∂L/∂b · ∂b/∂a | - | semua langkah |
| Aturan penjumlahan | variabel yang dipakai di beberapa tempat: gradiennya dijumlah | - | h₁ dipakai 4 gerbang; bobot dipakai di setiap t; c₁ punya 2 jalur |
| Turunan sigmoid | σ′(a) = σ(a)·(1 − σ(a)) | σ(1.25) = 0.777300 → σ′ = 0.173105 | gerbang f, i, o |
| Turunan tanh | tanh′(a) = 1 − tanh²(a) | tanh(0.35) = 0.336376 → 0.886851 | c̃, tanh(c) |

**Asal turunan sigmoid** (dari persamaan 11):

$$\sigma(a) = \frac{1}{1+e^{-a}}
\;\Rightarrow\;
\sigma'(a) = \frac{e^{-a}}{(1+e^{-a})^2}
= \frac{1}{1+e^{-a}}\cdot\frac{e^{-a}}{1+e^{-a}}
= \sigma(a)\,\bigl(1-\sigma(a)\bigr)$$

**Asal turunan tanh** (dari persamaan 13, $\tanh(a) = 2\sigma(2a) - 1$):

$$\tanh'(a) = 4\,\sigma(2a)\bigl(1-\sigma(2a)\bigr)
= 4\cdot\frac{1+\tanh(a)}{2}\cdot\frac{1-\tanh(a)}{2}
= 1-\tanh^2(a)$$

## 2. Backward di Lapisan Dense

Backward berjalan dari kanan ke kiri, mulai dari loss. Rantainya:
`h₂ → ŷ' = W_y·h₂ + b_y → L = (y' − ŷ')²`. Setiap turunan memakai rumus dari bagian 1.

```
Diketahui: h₂ = 0.217794, W_y = 0.8, b_y = 0, ŷ' = 0.174235, y' = 0.7

(1) ∂L/∂ŷ' — seberapa besar loss berubah bila prediksi berubah
    L = (y' - ŷ')²
    misalkan u = y' - ŷ'           →  L = u²
    dL/du  = 2u                     (aturan pangkat)
    du/dŷ' = 0 - 1 = -1             (y' adalah data, turunannya 0)
    ∂L/∂ŷ' = 2u × (-1) = -2(y' - ŷ')   (aturan rantai)
           = -2 × (0.7 - 0.174235)
           = -2 × 0.525765
           = -1.051530
    Cara lain tanpa aturan rantai: L = y'² - 2y'ŷ' + ŷ'²
                                   ∂L/∂ŷ' = -2y' + 2ŷ' = 2(ŷ' - y')  (sama)

(2) ∂L/∂W_y — gradien bobot dense
    ŷ' = W_y × h₂ + b_y  →  ∂ŷ'/∂W_y = h₂   (fungsi linear; h₂ dan b_y dianggap konstan)
    ∂L/∂W_y = ∂L/∂ŷ' × h₂ = (-1.051530) × 0.217794 = -0.229017

(3) ∂L/∂b_y — gradien bias dense
    ∂ŷ'/∂b_y = 1
    ∂L/∂b_y = ∂L/∂ŷ' × 1 = -1.051530

(4) ∂L/∂h₂ — sinyal kesalahan yang dikirim MASUK ke sel (awal BPTT)
    ∂ŷ'/∂h₂ = W_y            (kali ini h₂ yang menjadi variabel)
    ∂L/∂h₂ = ∂L/∂ŷ' × W_y = (-1.051530) × 0.8 = -0.841224
```

**Makna setiap angka:**

- **∂L/∂ŷ′ = -1.051530**: tanda negatif berarti menaikkan prediksi akan menurunkan loss
  (sesuai, karena prediksi masih terlalu rendah). Jika ŷ′ naik 0.001, loss turun sekitar
  0.0010515. Nilai ini adalah **sinyal kesalahan keluaran** yang dipakai ulang
  oleh tiga turunan berikutnya.
- **∂L/∂W_y = -0.229017**: pengaruh W_y terhadap prediksi "dikali" h₂. Karena h₂
  hanya 0.217794, gradien W_y lebih kecil daripada gradien ŷ′.
- **∂L/∂b_y = -1.051530**: bias langsung ditambahkan ke prediksi, jadi gradiennya
  sama dengan sinyal kesalahan keluaran.
- **∂L/∂h₂ = -0.841224**: h₂ **bukan parameter**, jadi tidak diupdate Adam.
  Nilai ini adalah pintu masuk BPTT ke dalam sel.

**Bukti dengan uji geser.** Turunan berarti "perubahan loss bila variabel digeser
sedikit". Setiap besaran digeser +0.001, lalu loss dihitung ulang:

| Yang digeser +0.001 | Perkiraan dari gradien (gradien × 0.001) | Perubahan loss sebenarnya |
|---|---|---|
| ŷ' | (-1.051530) × 0.001 = -0.0010515 | -0.0010505 |
| W_y (0.8 → 0.801) | (-0.229017) × 0.001 = -0.0002290 | -0.0002290 |
| b_y (0 → 0.001) | (-1.051530) × 0.001 = -0.0010515 | -0.0010505 |
| h₂ | (-0.841224) × 0.001 = -0.0008412 | -0.0008406 |

Selisih kecil (sekitar 0.000001) muncul karena turunan adalah pendekatan garis lurus,
sedangkan loss berbentuk kuadrat; sisanya sebesar (0.001 × ∂ŷ′/∂θ)².

## 3. Backward di Dalam Sel, t = 2

Sinyal ∂L/∂h₂ masuk ke sel lewat dua rumus forward yang diturunkan:
`h₂ = o₂ × tanh(c₂)` (menuju output gate dan cell state), lalu
`c₂ = f₂ × c₁ + i₂ × c̃₂` (menuju forget gate, input gate, dan kandidat). Setiap gerbang
lalu melewati turunan aktivasinya (σ′ atau tanh′) sehingga diperoleh sinyal kesalahan
gerbang **δ**.

```
Masukan dari bagian 2: ∂L/∂h₂ = -0.841224

(a) Output gate — turunkan h₂ = o₂ × tanh(c₂) terhadap o₂ (aturan perkalian)
    ∂L/∂o₂ = ∂L/∂h₂ × tanh(c₂) = (-0.841224) × 0.380359 = -0.319967
    δo₂    = ∂L/∂o₂ × o₂(1 - o₂)                     (turunan sigmoid)
           = (-0.319967) × 0.572601 × 0.427399
           = (-0.319967) × 0.244729
           = -0.078305

(b) Cell state — turunkan h₂ = o₂ × tanh(c₂) terhadap c₂ (perkalian + turunan tanh)
    ∂L/∂c₂ = ∂L/∂h₂ × o₂ × (1 - tanh²(c₂))
           = (-0.841224) × 0.572601 × (1 - 0.380359²)
           = (-0.841224) × 0.572601 × 0.855327
           = -0.411999
    (t = 2 adalah time step terakhir, jadi belum ada kiriman dari t = 3)

(c) Forget gate — turunkan c₂ = f₂ × c₁ + i₂ × c̃₂ terhadap f₂
    ∂L/∂f₂ = ∂L/∂c₂ × c₁ = (-0.411999) × 0.193228 = -0.079610
    δf₂    = ∂L/∂f₂ × f₂(1 - f₂) = (-0.079610) × 0.164260 = -0.013077

(d) Input gate — turunkan c₂ terhadap i₂
    ∂L/∂i₂ = ∂L/∂c₂ × c̃₂ = (-0.411999) × 0.414463 = -0.170758
    δi₂    = ∂L/∂i₂ × i₂(1 - i₂) = (-0.170758) × 0.240661 = -0.041095

(e) Kandidat — turunkan c₂ terhadap c̃₂
    ∂L/∂c̃₂ = ∂L/∂c₂ × i₂ = (-0.411999) × 0.596639 = -0.245815
    δc̃₂    = ∂L/∂c̃₂ × (1 - c̃₂²) = (-0.245815) × 0.828220 = -0.203589
```

## 4. Mengirim Kesalahan ke t = 1 (Inti Through Time)

Kesalahan di t = 2 sebagian disebabkan oleh keadaan hari sebelumnya, jadi sinyalnya
dikirim mundur ke t = 1 lewat **dua jalur**:

1. **Jalur hidden state h₁.** h₁ dipakai oleh keempat gerbang di t = 2 (lewat bobot U),
   jadi keempat kontribusinya dijumlahkan (aturan penjumlahan).
2. **Jalur cell state c₁.** Dari `c₂ = f₂ × c₁ + ...`, turunannya terhadap c₁ adalah f₂.
   Jalur ini hanya berupa perkalian sederhana, itulah sebabnya disebut "jalan tol gradien"
   (Gambar 2).

```
Jalur 1 — hidden state:
  ∂L/∂h₁ = U_f × δf₂ + U_i × δi₂ + U_c × δc̃₂ + U_o × δo₂
         = 0.4 × (-0.013077) + 0.3 × (-0.041095) + 0.2 × (-0.203589) + 0.5 × (-0.078305)
         = (-0.005231) + (-0.012328) + (-0.040718) + (-0.039153)
         = -0.097429

Jalur 2 — cell state:
  kiriman ke c₁ = ∂L/∂c₂ × f₂ = (-0.411999) × 0.792815 = -0.326639
```

## 5. Backward di Dalam Sel, t = 1

Di t = 1 rumusnya sama dengan bagian 3. Bedanya, cell state c₁ menerima **dua kiriman**
(dari jalur cell state dan dari h₁), lalu keduanya dijumlahkan.

```
Masukan dari bagian 4: ∂L/∂h₁ = -0.097429, kiriman jalur cell state = -0.326639

(a) Output gate
    ∂L/∂o₁ = ∂L/∂h₁ × tanh(c₁) = (-0.097429) × 0.190859 = -0.018595
    δo₁    = (-0.018595) × 0.247517 = -0.004603

(b) Cell state c₁ = [kiriman jalur cell state] + [kiriman lewat h₁ = o₁ × tanh(c₁)]
    ∂L/∂c₁ = ∂L/∂c₂ × f₂ + ∂L/∂h₁ × o₁ × (1 - tanh²(c₁))
           = (-0.326639) + (-0.097429) × 0.549834 × 0.963573
           = (-0.326639) + (-0.051619)
           = -0.378257
    Porsi jalur cell state = -0.326639 / -0.378257 = 86.4%

(c) Forget gate
    ∂L/∂f₁ = ∂L/∂c₁ × c₀ = (-0.378257) × 0.000000 = 0.000000
    δf₁    = 0.000000     (nol karena c₀ = 0: belum ada memori untuk dilupakan)

(d) Input gate
    ∂L/∂i₁ = ∂L/∂c₁ × c̃₁ = (-0.378257) × 0.336376 = -0.127236
    δi₁    = (-0.127236) × 0.244458 = -0.031104

(e) Kandidat
    ∂L/∂c̃₁ = ∂L/∂c₁ × i₁ = (-0.378257) × 0.574443 = -0.217287
    δc̃₁    = (-0.217287) × 0.886851 = -0.192701
```

Sekitar **86.4%** sinyal kesalahan ke c₁ datang lewat jalur cell state.
Inilah wujud nyata alasan LSTM tahan terhadap *vanishing gradient*: di jalur ini gradien
hanya dikalikan f, bukan melewati tanh dan matriks bobot berulang kali seperti pada RNN
biasa.

## 6. Gradien Total Setiap Bobot dan Bias

Bobot yang sama dipakai di t = 1 **dan** t = 2, sehingga kontribusi kedua time step
dijumlahkan (aturan penjumlahan). Dari $a_t = W x_t + U h_{t-1} + b$ diperoleh
$\partial a/\partial W = x_t$, $\partial a/\partial U = h_{t-1}$, dan
$\partial a/\partial b = 1$, sehingga:

$$\frac{\partial L}{\partial W} = \sum_t \delta_t\,x_t \qquad
\frac{\partial L}{\partial U} = \sum_t \delta_t\,h_{t-1} \qquad
\frac{\partial L}{\partial b} = \sum_t \delta_t$$

```
Forget gate:  δf₁ = 0.000000,  δf₂ = -0.013077
  ∂L/∂W_f = δf₁ × x₁ + δf₂ × x₂ = 0.000000 × 0.5 + (-0.013077) × 0.6 = 0.000000 + (-0.007846) = -0.007846
  ∂L/∂U_f = δf₁ × h₀ + δf₂ × h₁ = 0.000000 × 0.000000 + (-0.013077) × 0.104941 = -0.001372
  ∂L/∂b_f = δf₁ + δf₂ = 0.000000 + (-0.013077) = -0.013077

Input gate:  δi₁ = -0.031104,  δi₂ = -0.041095
  ∂L/∂W_i = δi₁ × x₁ + δi₂ × x₂ = (-0.031104) × 0.5 + (-0.041095) × 0.6 = (-0.015552) + (-0.024657) = -0.040209
  ∂L/∂U_i = δi₁ × h₀ + δi₂ × h₁ = (-0.031104) × 0.000000 + (-0.041095) × 0.104941 = -0.004313
  ∂L/∂b_i = δi₁ + δi₂ = (-0.031104) + (-0.041095) = -0.072199

Kandidat (c̃):  δc̃₁ = -0.192701,  δc̃₂ = -0.203589
  ∂L/∂W_c = δc̃₁ × x₁ + δc̃₂ × x₂ = (-0.192701) × 0.5 + (-0.203589) × 0.6 = (-0.096351) + (-0.122153) = -0.218504
  ∂L/∂U_c = δc̃₁ × h₀ + δc̃₂ × h₁ = (-0.192701) × 0.000000 + (-0.203589) × 0.104941 = -0.021365
  ∂L/∂b_c = δc̃₁ + δc̃₂ = (-0.192701) + (-0.203589) = -0.396290

Output gate:  δo₁ = -0.004603,  δo₂ = -0.078305
  ∂L/∂W_o = δo₁ × x₁ + δo₂ × x₂ = (-0.004603) × 0.5 + (-0.078305) × 0.6 = (-0.002301) + (-0.046983) = -0.049284
  ∂L/∂U_o = δo₁ × h₀ + δo₂ × h₁ = (-0.004603) × 0.000000 + (-0.078305) × 0.104941 = -0.008217
  ∂L/∂b_o = δo₁ + δo₂ = (-0.004603) + (-0.078305) = -0.082908

Lapisan dense (dari bagian 2):
  ∂L/∂W_y = -0.229017
  ∂L/∂b_y = -1.051530
```

**Semua gradien bernilai negatif.** Prediksi terlalu rendah, dan setiap bobot awal
bernilai positif, sehingga menaikkan bobot mana pun akan menaikkan ŷ′. Karena itu
langkah Adam berikutnya akan menaikkan semua parameter ([Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md)).
Perhatikan juga **∂L/∂U = δ₂ × h₁ saja**: pada t = 1 bobot U tidak mendapat gradien karena
h₀ = 0, sedangkan bias tetap mendapat sinyal dari kedua time step
([Tahap 7.12 (Bias pada LSTM: Peran dan Cara Menghitungnya)](tahap_07_12_bias_lstm.md)).

## 7. Pemeriksaan Gradien dengan Turunan Numerik

Untuk memastikan seluruh rantai turunan di atas benar, setiap gradien dibandingkan
dengan turunan numerik (beda pusat):
$\dfrac{L(\theta+10^{-6}) - L(\theta-10^{-6})}{2\cdot10^{-6}}$.

| Parameter | Gradien BPTT (manual) | Gradien numerik | Selisih |
|---|---|---|---|
| W_f | -0.00784600 | -0.00784600 | 4.6e-11 |
| U_f | -0.00137228 | -0.00137228 | 2.3e-11 |
| b_f | -0.01307667 | -0.01307667 | 2.2e-11 |
| W_i | -0.04020889 | -0.04020889 | 1.1e-11 |
| U_i | -0.00431252 | -0.00431252 | 9.3e-11 |
| b_i | -0.07219882 | -0.07219882 | 3.0e-11 |
| W_c | -0.21850381 | -0.21850381 | 2.9e-11 |
| U_c | -0.02136474 | -0.02136474 | 4.9e-11 |
| b_c | -0.39628990 | -0.39628990 | 8.4e-12 |
| W_o | -0.04928448 | -0.04928448 | 1.6e-11 |
| U_o | -0.00821741 | -0.00821741 | 2.3e-11 |
| b_o | -0.08290791 | -0.08290791 | 5.1e-12 |
| W_y | -0.22901679 | -0.22901679 | 6.8e-12 |
| b_y | -1.05152971 | -1.05152971 | 3.1e-11 |

Selisih terbesar hanya 9.3e-11, jadi seluruh gradien manual terbukti benar.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.2: Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.4: Simulasi Pelatihan LSTM, Langkah 4: Update Adam](tahap_07_04_simulasi_update_adam_lstm.md)
