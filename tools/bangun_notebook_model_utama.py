# -*- coding: utf-8 -*-
"""
Pembangun notebook ``notebooks/btc_lstm_vs_gru_model_utama.ipynb``.

Notebook ini salinan notebook utama (``btc_lstm_vs_gru.ipynb``) dengan satu
perbedaan: seluruh perhitungan manual pelatihan LSTM dan GRU memakai MODEL
UTAMA penelitian (model terbaik hasil tuning, 7 hari x 10 fitur, batch 32,
bobot awal asli Keras), bukan model mini 1 neuron. Karena jumlah neuron model
utama baru diketahui setelah tuning, urutan Tahap 7 dan 8 disusun ulang:

    tuning -> model terbaik -> jumlah parameter -> satu siklus pelatihan model
    terbaik dihitung manual (alur sel, forward + loss, BPTT, Adam) -> kurva
    loss -> forward pass model terlatih -> bias -> prediksi

Sel-sel lain diambil apa adanya dari ``tools/bangun_notebook.py`` supaya
Tahap 0-6 dan 9-14 identik dengan notebook utama. Keluaran notebook ini
ditulis ke ``outputs_model_utama/`` sehingga keluaran notebook utama di
``outputs/`` tidak tertimpa. Jalankan dengan::

    python tools/bangun_notebook_model_utama.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bangun_notebook as dasar  # noqa: E402  (hanya menyusun daftar sel, tidak menulis berkas)

FOLDER_KELUARAN = "outputs_model_utama"
NAMA_NOTEBOOK = "notebooks/btc_lstm_vs_gru_model_utama.ipynb"

SEL = list(dasar.SEL)


# --------------------------------------------------------------------------- #
# Alat bantu pengubah sel
# --------------------------------------------------------------------------- #
def cari(penanda: str) -> int:
    """Indeks satu-satunya sel yang memuat ``penanda``."""
    hasil = [i for i, (_, isi) in enumerate(SEL) if penanda in isi]
    assert len(hasil) == 1, f"Penanda {penanda!r} ditemukan {len(hasil)} kali"
    return hasil[0]


def ganti(penanda: str, lama: str, baru: str) -> None:
    """Ganti teks ``lama`` (harus muncul tepat sekali) di sel yang memuat ``penanda``."""
    i = cari(penanda)
    jenis, isi = SEL[i]
    assert isi.count(lama) == 1, f"{lama!r} muncul {isi.count(lama)} kali di sel {i}"
    SEL[i] = (jenis, isi.replace(lama, baru))


def ganti_sel(penanda: str, sel_baru: list) -> None:
    i = cari(penanda)
    SEL[i:i + 1] = sel_baru


def hapus_sel(penanda: str) -> None:
    SEL.pop(cari(penanda))


def sisip_setelah(penanda: str, sel_baru: list) -> None:
    i = cari(penanda)
    SEL[i + 1:i + 1] = sel_baru


def md(teks: str):
    return ("markdown", teks.strip("\n"))


def kode(teks: str):
    return ("code", teks.strip("\n"))


# =========================================================================== #
# JUDUL DAN PERSIAPAN
# =========================================================================== #
ganti("**Program Studi Ilmu Aktuaria — Skripsi S1**",
      "**Program Studi Ilmu Aktuaria — Skripsi S1**",
      """**Program Studi Ilmu Aktuaria — Skripsi S1**

> **Versi model utama.** Isi notebook ini sama dengan `btc_lstm_vs_gru.ipynb`, kecuali
> perhitungan manual pelatihan LSTM (Tahap 7.6-7.9) dan GRU (Tahap 8.6-8.9). Di sini
> perhitungannya memakai **model utama penelitian**, yaitu model terbaik hasil tuning
> dengan masukan 7 hari × 10 fitur, batch 32, dan bobot awal asli Keras, bukan model mini
> 1 neuron. Karena itu tuning dikerjakan lebih dulu, baru kemudian satu siklus pelatihan
> model terbaiknya dihitung ulang langkah demi langkah dan dicocokkan dengan Keras.
> Semua tabel, gambar, dan perhitungan manual notebook ini disimpan di folder
> `outputs_model_utama/`.""")

ganti("import manual_pelatihan           # simulasi manual pelatihan LSTM & GRU (model mini)",
      "import manual_pelatihan           # simulasi manual pelatihan LSTM & GRU (model mini)",
      f"""import manual_model_utama as manual_utama   # perhitungan manual pelatihan pada model utama

# Keluaran notebook versi model utama disimpan terpisah dari notebook utama
utils.atur_folder_keluaran("{FOLDER_KELUARAN}", "{NAMA_NOTEBOOK}")""")


# =========================================================================== #
# TAHAP 7 — LSTM
# =========================================================================== #
ganti_sel("### 7.E Simulasi manual satu siklus pelatihan LSTM (model mini)", [md(r'''
### 7.E Alur Tahap 7 pada notebook ini

Perhitungan manual di tahap ini memakai **model LSTM utama**, yaitu model terbaik hasil
tuning. Jumlah neuronnya baru diketahui setelah tuning selesai, jadi urutannya begini:

| Subbagian | Isi |
|---|---|
| 7.1-7.2 | Tuning 15 kombinasi neuron × epoch, lalu memilih model terbaik dari RMSE validasi |
| 7.3-7.5 | Arsitektur, jumlah parameter, dan bentuk bobot model terbaik |
| 7.6 | Sigmoid dan tanh (persamaan 11-13), lalu alur sel LSTM (Gambar 1-6) dengan satu jendela data asli |
| 7.7 | Langkah 1-2 pelatihan: *forward pass* batch pertama (32 sampel) dan loss MSE |
| 7.8 | Langkah 3: *backpropagation through time*, gradien seluruh bobot dan bias |
| 7.9 | Langkah 4: update Adam, dicocokkan dengan `model.fit()` Keras |
| 7.10 | Kurva loss hasil seluruh iterasi pelatihan |
| 7.11 | *Forward pass* model terlatih pada data uji |
| 7.12 | Bias: perannya dan cara nilainya dipelajari |
| 7.13 | Prediksi data uji |

Subbagian 7.6-7.9 memakai **bobot awal** model terbaik (dibangun ulang dengan seed 42,
sehingga sama persis dengan bobot awal saat tuning), sedangkan 7.11 memakai bobot
**akhir** setelah pelatihan. Setiap perhitungan manual disimpan sebagai
`outputs_model_utama/perhitungan_manual/tahap_07_SS_*.md`.
''')])
for penanda in ("7.1 PERHITUNGAN MANUAL — sigmoid dan tanh",
                "7.2 PERHITUNGAN MANUAL — simulasi langkah 1-2",
                "7.3 PERHITUNGAN MANUAL — simulasi langkah 3",
                "7.4 PERHITUNGAN MANUAL — simulasi langkah 4"):
    hapus_sel(penanda)

ganti('utils.cetak_sub("7.5 Tuning hyperparameter LSTM', '"7.5 Tuning', '"7.1 Tuning')
ganti('utils.cetak_sub("7.6 Tabel 9', '"7.6 Tabel 9', '"7.2 Tabel 9')
ganti('utils.cetak_sub("7.7 Ringkasan arsitektur model LSTM terbaik")',
      '"7.7 Ringkasan arsitektur', '"7.3 Ringkasan arsitektur')
ganti('utils.cetak_sub("7.3 Ringkasan arsitektur model LSTM terbaik")',
      '"7.8 PERHITUNGAN MANUAL — jumlah parameter LSTM"', '"7.4 PERHITUNGAN MANUAL — jumlah parameter LSTM"')
ganti('utils.cetak_sub("7.3 Ringkasan arsitektur model LSTM terbaik")',
      '"7.9 Bentuk dan contoh nilai bobot asli Keras"', '"7.5 Bentuk dan contoh nilai bobot asli Keras (bobot akhir)"')

sisip_setelah('utils.cetak_sub("7.3 Ringkasan arsitektur model LSTM terbaik")', [
    md(r'''
### 7.F Satu siklus pelatihan model LSTM utama, dihitung manual

Sekarang model terbaik sudah diketahui. Subbagian 7.6-7.9 menghitung ulang **apa yang
dikerjakan `model.fit()` di iterasi pertama** saat model ini dilatih: alur sel, *forward
pass* batch pertama, loss MSE, BPTT, lalu update Adam. Semua angkanya dicek ke Keras:

1. **Bobot awal** dibangun ulang dengan `bangun_model()` dan seed 42, dua kali, lalu
   dipastikan identik. Jadi bobot ini sama persis dengan bobot awal model terbaik saat tuning.
2. **Forward pass dan loss** manual dibandingkan dengan `model.predict()` dan fungsi loss Keras.
3. **Gradien BPTT** manual dibandingkan dengan `tf.GradientTape` dan turunan numerik.
4. **Update Adam** manual dibandingkan dengan bobot yang dihasilkan `model.fit()`
   sungguhan setelah batch ke-1 dan ke-2.

Perhitungannya ada di `src/manual_model_utama.py`.
'''),
    kode(r'''
utils.cetak_sub("7.6 PERHITUNGAN MANUAL — sigmoid dan tanh (persamaan 11-13), alur sel LSTM (Gambar 1-6) pada model utama")

# Konteks penelitian, dipakai bersama oleh Tahap 7, 8, dan 9
konteks_pelatihan = manual_utama.konteks_penelitian(
    n_latih=len(X_latih), n_val=len(X_val), lookback=LOOKBACK, n_fitur=JUMLAH_FITUR,
    batch=CONFIG["BATCH_SIZE"], lr=CONFIG["LEARNING_RATE"],
    neuron_grid=CONFIG["NEURON_GRID"], epoch_grid=CONFIG["EPOCH_GRID"],
    x_min=penskala_y.data_min_[0], x_max=penskala_y.data_max_[0],
)
info_dasar = {
    "kode": [v["kode"].split("/")[-1].strip() for v in modul_data.DEFINISI_VARIABEL],
    "nama": [v["nama"] for v in modul_data.DEFINISI_VARIABEL],
    "tanggal_jendela": [str(pd.Timestamp(t).date())
                        for t in df_bersih["date"].to_numpy()[idx_latih[0] - LOOKBACK:idx_latih[0]]],
    "tanggal_target": [str(pd.Timestamp(t).date()) for t in tanggal_latih],
}


def siapkan_siklus_manual(tipe: str, jumlah_neuron: int):
    """
    Hitung satu siklus pelatihan model terbaik secara manual dan cocokkan
    setiap langkahnya dengan Keras.

    Model dibangun ulang dengan seed penelitian, sehingga bobot awalnya sama
    persis dengan bobot awal model terbaik saat tuning.
    """
    ukuran_batch = CONFIG["BATCH_SIZE"]
    model_awal = bangun_model(tipe, jumlah_neuron, CONFIG["SEED"])
    bobot_awal = [w.copy() for w in model_awal.get_weights()]
    model_kembar = bangun_model(tipe, jumlah_neuron, CONFIG["SEED"])
    assert all(np.array_equal(w1, w2) for w1, w2 in zip(bobot_awal, model_kembar.get_weights())), \
        "Bobot awal tidak reproducible"

    siklus = manual_utama.SiklusModelUtama(tipe, bobot_awal, X_latih, y_latih,
                                           ukuran_batch, CONFIG["LEARNING_RATE"])

    # (1) Forward pass dan loss batch pertama vs Keras
    X_b1, y_b1 = X_latih[:ukuran_batch], y_latih[:ukuran_batch]
    fungsi_loss = tf.keras.losses.MeanSquaredError()
    siklus.periksa_forward(model_awal.predict(X_latih[:1], verbose=0),
                           model_awal.predict(X_b1, verbose=0),
                           float(fungsi_loss(y_b1, model_awal(X_b1, training=False))))

    # (2) Gradien BPTT vs TensorFlow GradientTape dan turunan numerik
    with tf.GradientTape() as pita:
        loss_b1 = fungsi_loss(y_b1, model_awal(X_b1, training=True))
    siklus.periksa_gradien([g.numpy() for g in pita.gradient(loss_b1, model_awal.trainable_weights)])
    g1 = "f" if tipe == "LSTM" else "z"
    siklus.gradien_numerik([("b_y", (0,)), ("W_y", (0, 0)), (f"W_{g1}", (0, 0)), (f"U_{g1}", (0, 0))])

    # (3) Update Adam vs model.fit() sungguhan: ambil bobot setelah batch ke-1 dan ke-2
    bobot_fit = {}

    class AmbilBobot(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch < 2:
                bobot_fit[batch] = [w.copy() for w in self.model.get_weights()]

    model_fit = bangun_model(tipe, jumlah_neuron, CONFIG["SEED"])
    model_fit.fit(X_latih, y_latih, epochs=1, batch_size=ukuran_batch,
                  shuffle=False, verbose=0, callbacks=[AmbilBobot()])
    siklus.periksa_adam(bobot_fit[0], bobot_fit[1])

    print(f"  Model {tipe} utama: {jumlah_neuron} neuron, {utils.fmt_int(siklus.n_param)} parameter")
    print(f"  Pemeriksaan terhadap Keras (selisih absolut terbesar):")
    print(f"    forward pass 1 sampel  : {siklus.selisih_satu:.2e}")
    print(f"    forward pass 32 sampel : {siklus.selisih_batch:.2e}")
    print(f"    loss batch pertama     : {siklus.selisih_loss:.2e}")
    print(f"    gradien BPTT           : {siklus.selisih_gradien:.2e}")
    print(f"    bobot setelah Adam k=1 : {siklus.selisih_adam1:.2e}")
    print(f"    bobot setelah Adam k=2 : {siklus.selisih_adam2:.2e}")
    print("  OK Satu siklus pelatihan manual SAMA dengan model.fit() Keras\n")
    return siklus


siklus_lstm = siapkan_siklus_manual("LSTM", neuron_lstm)
info_lstm = dict(info_dasar, epoch=epoch_lstm)

isi = manual_utama.alur_sel(siklus_lstm, konteks_pelatihan, info_lstm)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("LSTM", "alur_sel"), isi)
'''),
    kode(r'''
utils.cetak_sub("7.7 PERHITUNGAN MANUAL — langkah 1-2 pelatihan: forward pass batch pertama dan loss (LSTM)")
isi = manual_utama.forward_loss(siklus_lstm, konteks_pelatihan, info_lstm)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("LSTM", "forward_loss"), isi)
'''),
    kode(r'''
utils.cetak_sub("7.8 PERHITUNGAN MANUAL — langkah 3 pelatihan: backpropagation through time (LSTM)")
isi = manual_utama.bptt(siklus_lstm, konteks_pelatihan, info_lstm)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("LSTM", "bptt"), isi)
'''),
    kode(r'''
utils.cetak_sub("7.9 PERHITUNGAN MANUAL — langkah 4 pelatihan: update Adam (LSTM)")
isi = manual_utama.adam(siklus_lstm, konteks_pelatihan, info_lstm, model_lstm.get_weights())
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("LSTM", "adam"), isi)
'''),
])

ganti('utils.cetak_sub("7.10 Gambar 3', '"  validasi pada Tahap 7.5 sudah memperhitungkan hal ini."',
      '"  validasi pada Tahap 7.1 sudah memperhitungkan hal ini."')
ganti('utils.cetak_sub("7.10 (lanjutan) PERHITUNGAN MANUAL — membaca kurva loss LSTM")',
      "manual_pelatihan.hasil_model(", "manual_utama.hasil_model(")
ganti('utils.cetak_sub("7.10 (lanjutan) PERHITUNGAN MANUAL — membaca kurva loss LSTM")',
      'manual_pelatihan.kurva_loss("LSTM", konteks_pelatihan, hasil_lstm)',
      'manual_utama.kurva_loss("LSTM", konteks_pelatihan, hasil_lstm)')
ganti("### 7.F Verifikasi manual *forward pass* LSTM", "### 7.F Verifikasi", "### 7.G Verifikasi")
ganti('utils.cetak_sub("7.12 PERHITUNGAN MANUAL — bias pada LSTM',
      "manual_pelatihan.bias(sim_lstm, konteks_pelatihan, hasil_lstm)",
      "manual_utama.bias(siklus_lstm, konteks_pelatihan, hasil_lstm, info_lstm, model_lstm.get_weights())")
ganti('utils.cetak_sub("7.13 Prediksi model LSTM pada data uji")',
      "Dengan demikian seluruh rumus LSTM pada Bab III sudah tervalidasi secara",
      """Satu siklus pelatihan model ini juga sudah dihitung ulang dari bobot awalnya:
forward pass, loss MSE, gradien BPTT untuk seluruh parameter, dan update Adam
cocok dengan model.fit() Keras (selisih bobot setelah dua iterasi
{siklus_lstm.selisih_adam2:.1e}).

Dengan demikian seluruh rumus LSTM pada Bab III sudah tervalidasi secara""")


# =========================================================================== #
# TAHAP 8 — GRU
# =========================================================================== #
ganti_sel("### 8.F Simulasi manual satu siklus pelatihan GRU (model mini)", [md(r'''
### 8.F Alur Tahap 8 pada notebook ini

Urutannya sama dengan Tahap 7: tuning dulu, lalu satu siklus pelatihan **model GRU utama**
(model terbaik hasil tuning) dihitung ulang langkah demi langkah.

| Subbagian | Isi |
|---|---|
| 8.1-8.2 | Tuning 15 kombinasi neuron × epoch, lalu memilih model terbaik dari RMSE validasi |
| 8.3-8.5 | Arsitektur, jumlah parameter, dan bentuk bobot model terbaik |
| 8.6 | Alur sel GRU (Gambar 7) dengan satu jendela data asli |
| 8.7 | Langkah 1-2 pelatihan: *forward pass* batch pertama dan loss MSE |
| 8.8 | Langkah 3: *backpropagation through time* |
| 8.9 | Langkah 4: update Adam, dicocokkan dengan `model.fit()` Keras |
| 8.10 | Kurva loss hasil seluruh iterasi pelatihan |
| 8.11 | *Forward pass* model terlatih pada data uji |
| 8.12 | Bias, termasuk dua jenis bias GRU |
| 8.13 | Prediksi data uji |

Setiap perhitungan manual disimpan sebagai
`outputs_model_utama/perhitungan_manual/tahap_08_SS_*.md`.
''')])
hapus_sel("8.1 PERHITUNGAN MANUAL — alur sel GRU (Gambar 7) dengan model mini")
for penanda in ("8.2 PERHITUNGAN MANUAL — simulasi langkah 1-2",
                "8.3 PERHITUNGAN MANUAL — simulasi langkah 3",
                "8.4 PERHITUNGAN MANUAL — simulasi langkah 4"):
    hapus_sel(penanda)

ganti('utils.cetak_sub("8.5 Tuning hyperparameter GRU',
      'utils.cetak_sub("8.5 Tuning', 'utils.cetak_header(8, "Model GRU")\n\nutils.cetak_sub("8.1 Tuning')
ganti('utils.cetak_sub("8.6 Tabel 11', '"8.6 Tabel 11', '"8.2 Tabel 11')
ganti('utils.cetak_sub("8.7 Ringkasan arsitektur model GRU terbaik")',
      '"8.7 Ringkasan arsitektur', '"8.3 Ringkasan arsitektur')
ganti('utils.cetak_sub("8.8 PERHITUNGAN MANUAL — jumlah parameter GRU")',
      '"8.8 PERHITUNGAN MANUAL — jumlah parameter GRU"', '"8.4 PERHITUNGAN MANUAL — jumlah parameter GRU"')
ganti('utils.cetak_sub("8.4 PERHITUNGAN MANUAL — jumlah parameter GRU")',
      '"8.9 Bentuk dan contoh nilai bobot asli Keras (GRU)"',
      '"8.5 Bentuk dan contoh nilai bobot asli Keras (GRU, bobot akhir)"')

sisip_setelah('utils.cetak_sub("8.4 PERHITUNGAN MANUAL — jumlah parameter GRU")', [
    md(r'''
### 8.G Satu siklus pelatihan model GRU utama, dihitung manual

Sama seperti 7.F: bobot awal model GRU terbaik dibangun ulang dengan seed 42, lalu satu
siklus pelatihannya (alur sel, *forward pass* + loss, BPTT, Adam) dihitung manual dan
dicocokkan dengan Keras. Rumus sel mengikuti Keras `reset_after=True` (8.A).
'''),
    kode(r'''
utils.cetak_sub("8.6 PERHITUNGAN MANUAL — alur sel GRU (Gambar 7) pada model utama")
siklus_gru = siapkan_siklus_manual("GRU", neuron_gru)
info_gru = dict(info_dasar, epoch=epoch_gru)

isi = manual_utama.alur_sel(siklus_gru, konteks_pelatihan, info_gru)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("GRU", "alur_sel"), isi)
'''),
    kode(r'''
utils.cetak_sub("8.7 PERHITUNGAN MANUAL — langkah 1-2 pelatihan: forward pass batch pertama dan loss (GRU)")
isi = manual_utama.forward_loss(siklus_gru, konteks_pelatihan, info_gru)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("GRU", "forward_loss"), isi)
'''),
    kode(r'''
utils.cetak_sub("8.8 PERHITUNGAN MANUAL — langkah 3 pelatihan: backpropagation through time (GRU)")
isi = manual_utama.bptt(siklus_gru, konteks_pelatihan, info_gru)
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("GRU", "bptt"), isi)
'''),
    kode(r'''
utils.cetak_sub("8.9 PERHITUNGAN MANUAL — langkah 4 pelatihan: update Adam (GRU)")
isi = manual_utama.adam(siklus_gru, konteks_pelatihan, info_gru, model_gru.get_weights())
utils.tampilkan_markdown(isi)
utils.tulis_manual(*manual_utama.berkas("GRU", "adam"), isi)
'''),
])

ganti('utils.cetak_sub("8.10 (lanjutan) PERHITUNGAN MANUAL — membaca kurva loss GRU")',
      "manual_pelatihan.hasil_model(", "manual_utama.hasil_model(")
ganti('utils.cetak_sub("8.10 (lanjutan) PERHITUNGAN MANUAL — membaca kurva loss GRU")',
      'manual_pelatihan.kurva_loss("GRU", konteks_pelatihan, hasil_gru)',
      'manual_utama.kurva_loss("GRU", konteks_pelatihan, hasil_gru)')
ganti("### 8.G Verifikasi manual *forward pass* GRU", "### 8.G Verifikasi", "### 8.H Verifikasi")
ganti('utils.cetak_sub("8.12 PERHITUNGAN MANUAL — bias pada GRU',
      "manual_pelatihan.bias(sim_gru, konteks_pelatihan, hasil_gru)",
      "manual_utama.bias(siklus_gru, konteks_pelatihan, hasil_gru, info_gru, model_gru.get_weights())")
ganti('utils.cetak_sub("8.13 Prediksi model GRU pada data uji")',
      "Sebagai tambahan, telah dibuktikan secara numerik bahwa rumus GRU Cho dkk.",
      """Satu siklus pelatihan model GRU utama juga sudah dihitung ulang dari bobot
awalnya dan cocok dengan model.fit() Keras (selisih bobot setelah dua iterasi
{siklus_gru.selisih_adam2:.1e}).

Sebagai tambahan, telah dibuktikan secara numerik bahwa rumus GRU Cho dkk.""")


# =========================================================================== #
# TAHAP 9 DAN PENUTUP
# =========================================================================== #
ganti('utils.cetak_sub("9.2 PERHITUNGAN MANUAL — perbandingan struktur LSTM dan GRU")',
      "manual_pelatihan.perbandingan(sim_lstm, sim_gru, konteks_pelatihan, hasil_lstm, hasil_gru)",
      "manual_utama.perbandingan(siklus_lstm, siklus_gru, konteks_pelatihan, hasil_lstm, hasil_gru)")

ganti('print("G. CATATAN METODOLOGIS UNTUK BAB III DAN BAB IV")',
      '''print("     simulasi manual satu siklus pelatihan (forward, loss, BPTT, Adam) pada")
print("     model mini LSTM & GRU lolos pemeriksaan gradien numerik.")''',
      '''print("     satu siklus pelatihan (forward, loss, BPTT, Adam) model LSTM dan GRU")
print("     utama yang dihitung manual sama dengan model.fit() Keras.")''')

# Nama berkas perhitungan manual mengikuti penomoran notebook ini, dan seluruh
# penyebutan folder keluaran diarahkan ke folder versi model utama
for i, (jenis, isi) in enumerate(SEL):
    isi = isi.replace("manual_pelatihan.berkas(", "manual_utama.berkas(")
    SEL[i] = (jenis, isi.replace("outputs/tabel/", f"{FOLDER_KELUARAN}/tabel/")
                        .replace("outputs/gambar/", f"{FOLDER_KELUARAN}/gambar/")
                        .replace("outputs/perhitungan_manual/", f"{FOLDER_KELUARAN}/perhitungan_manual/"))

# Pemeriksaan akhir: tidak ada lagi sisa model mini
for jenis, isi in SEL:
    assert "manual_pelatihan" not in isi and "sim_lstm" not in isi and "sim_gru" not in isi, isi[:200]


if __name__ == "__main__":
    dasar.SEL[:] = SEL
    akar = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dasar.tulis_notebook(os.path.join(akar, NAMA_NOTEBOOK))
