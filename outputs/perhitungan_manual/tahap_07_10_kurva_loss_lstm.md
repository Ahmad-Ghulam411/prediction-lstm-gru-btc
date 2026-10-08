# Perhitungan Manual - Tahap 7.10: Membaca Kurva Loss Model LSTM

Simulasi pada [Tahap 7.2 (Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss)](tahap_07_02_simulasi_forward_pass_dan_loss_lstm.md) sampai [Tahap 7.4 (Simulasi Pelatihan LSTM, Langkah 4: Update Adam)](tahap_07_04_simulasi_update_adam_lstm.md) memperlihatkan
satu siklus pelatihan pada model mini. Subbagian ini membaca hasil siklus yang sama pada
**model LSTM penelitian yang sebenarnya** (40 neuron, 500 epoch),
yaitu kurva loss latih dan loss validasi.

**Daftar isi**

- [1. Apa yang Digambar Kurva Loss](#1-apa-yang-digambar-kurva-loss)
- [2. Kurva Loss Model LSTM Terbaik](#2-kurva-loss-model-lstm-terbaik)
- [3. Cara Membacanya](#3-cara-membacanya)
- [4. Mengapa Jumlah Epoch Dipilih Lewat Data Validasi](#4-mengapa-jumlah-epoch-dipilih-lewat-data-validasi)
- [5. Hubungan MSE dengan RMSE dalam USD](#5-hubungan-mse-dengan-rmse-dalam-usd)
- [6. Ringkasan: Kapan Loss dan Adam Bekerja](#6-ringkasan-kapan-loss-dan-adam-bekerja)
- [7. Catatan untuk Naskah Skripsi](#7-catatan-untuk-naskah-skripsi)

**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).

## 1. Apa yang Digambar Kurva Loss

Di akhir setiap epoch, Keras mencatat dua angka:

- **loss latih**: rata-rata MSE dari 26 batch selama epoch itu
  (25 batch × 32 sampel + 1 batch × 4 sampel);
- **loss validasi**: MSE pada 90 sampel validasi memakai bobot akhir epoch.
  Angka ini **hanya diukur; tidak ada update bobot** dari data validasi.

Keduanya dihitung pada skala ternormalisasi (persamaan 28). Selama 500 epoch,
Adam melakukan 26 × 500 = **13,000
kali update**.

## 2. Kurva Loss Model LSTM Terbaik

![Kurva loss LSTM terbaik](../gambar/gambar_03_kurva_loss_latih_dan_validasi_model_lstm_terbaik.png)

| Besaran | Nilai |
|---|---|
| Loss latih epoch 1 | 0.02209279 |
| Loss latih epoch terakhir (500) | 0.00063068 |
| Penurunan loss latih | 97.1% |
| Loss validasi epoch 1 | 0.06462061 |
| Loss validasi epoch terakhir (500) | 0.00065024 |
| Loss validasi minimum | 0.00059231 (epoch 494) |

## 3. Cara Membacanya

- Jika **loss latih dan loss validasi turun bersama**, model sedang mempelajari pola yang
  benar.
- Jika **loss latih terus turun tetapi loss validasi naik**, itu tanda *overfitting*: model
  mulai menghafal data latih.
- Pada model ini, loss latih turun 97.1% dari epoch pertama. Loss validasi mencapai
  minimum di epoch 494, sangat dekat dengan epoch terakhir; loss validasi akhir (0.00065024) hampir sama dengan minimumnya.

## 4. Mengapa Jumlah Epoch Dipilih Lewat Data Validasi

RMSE validasi model 40 neuron untuk setiap jumlah epoch pada grid (tabel tuning LSTM):

| Epoch | RMSE validasi (USD) |
|---|---|
| 100 | 3,804.03 |
| 500 | 2,503.99 ← terbaik |
| 1,000 | 2,718.20 |

Pada neuron ini, melatih sampai 1,000 epoch justru memperburuk RMSE validasi dibandingkan 500 epoch. Pelatihan yang lebih lama tidak selalu lebih baik; kemungkinan besar model mulai terlalu menyesuaikan diri dengan data latih. Itulah gunanya data validasi untuk memilih jumlah epoch, sedangkan data uji tetap netral.

## 5. Hubungan MSE dengan RMSE dalam USD

Karena normalisasi min-max bersifat linear (persamaan 6 dan 8), selisih dalam USD sama
dengan selisih ternormalisasi dikali (x_max − x_min). Akibatnya:

**RMSE (USD) = (x_max − x_min) × √MSE**

Dengan rentang harga data latih 123,359.45 − 25,162.70 = 98,196.75:

```
RMSE = 98,196.75 × √0.00065024
     = 98,196.75 × 0.025500
     = 2,503.99 USD
RMSE validasi pada tabel tuning = 2,503.99 USD
```

Jadi **meminimalkan MSE saat pelatihan sama artinya dengan meminimalkan RMSE dalam USD**.
Kalimat ini berguna untuk menjawab pertanyaan "mengapa loss-nya MSE tetapi evaluasinya
RMSE?".

## 6. Ringkasan: Kapan Loss dan Adam Bekerja

| Tahap penelitian | Loss MSE | Adam |
|---|---|---|
| Inisialisasi model | - | m = 0, v = 0, k = 0 |
| Setiap batch latih (26× per epoch) | dihitung, menjadi sumber gradien | memperbarui semua bobot dan bias |
| Akhir setiap epoch | loss latih dan loss validasi dicatat (kurva loss) | tidak ada update dari data validasi |
| Pemilihan neuron dan epoch terbaik | tidak (memakai RMSE validasi USD, setara √MSE × rentang) | tidak |
| Prediksi data uji dan evaluasi | tidak | tidak (bobot sudah beku) |

## 7. Catatan untuk Naskah Skripsi

**Opsional, hubungan loss dan evaluasi** (subbab 1.5.11 atau 1.5.12). Saran kalimat:

> Karena normalisasi min-max bersifat linear, RMSE dalam USD sama dengan
> (x_max − x_min) × √MSE, sehingga meminimalkan MSE pada skala ternormalisasi setara
> dengan meminimalkan RMSE dalam USD.

<!-- navigasi-perhitungan-manual -->

---

← Sebelumnya: [Tahap 7.8: Jumlah Parameter Model LSTM](tahap_07_08_jumlah_parameter_lstm.md)

[Daftar isi perhitungan manual](README.md)

→ Berikutnya: [Tahap 7.11: Forward Pass LSTM (Jendela Pertama Data Uji)](tahap_07_11_forward_pass_lstm.md)
