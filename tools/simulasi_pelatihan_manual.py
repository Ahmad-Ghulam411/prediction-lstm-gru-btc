# -*- coding: utf-8 -*-
"""
Simulasi pelatihan bobot LSTM dan GRU yang dihitung manual langkah demi
langkah: forward pass, loss MSE, backpropagation through time (BPTT), gradien
bobot dan bias, lalu pembaruan parameter dengan Adam.

Agar setiap angka dapat diikuti dengan tangan, simulasi memakai model mini
(1 neuron, 1 fitur, 2 time step, 1 sampel). Rumus yang dipakai sama dengan
rumus Keras pada model penelitian (persamaan 15-31 pada draf skripsi).

Keluaran:
    outputs/perhitungan_manual/tahap_7_8_simulasi_pelatihan_lstm_gru.md

Skrip hanya memakai pustaka standar Python sehingga dapat dijalankan tanpa
TensorFlow. Angka konteks penelitian (jumlah sampel latih, batch size, rentang
harga, loss validasi, model terbaik, dan bias model terlatih) dibaca dari
keluaran notebook yang sudah ada di folder notebooks/ dan outputs/.

Jalankan dari akar proyek:  python tools/simulasi_pelatihan_manual.py
"""
import csv
import glob
import json
import math
import os
import re

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MANUAL = os.path.join(AKAR, "outputs", "perhitungan_manual")
DIR_TABEL = os.path.join(AKAR, "outputs", "tabel")
NOTEBOOK = os.path.join(AKAR, "notebooks", "btc_lstm_vs_gru.ipynb")
BERKAS_KELUARAN = os.path.join(DIR_MANUAL, "tahap_7_8_simulasi_pelatihan_lstm_gru.md")

# --------------------------------------------------------------------------- #
# Pengaturan simulasi
# --------------------------------------------------------------------------- #
X = [0.5, 0.6]   # x_1, x_2 : fitur ternormalisasi hari 1 dan hari 2
Y = 0.7          # y'      : target ternormalisasi hari 3
LR, B1, B2, EPS = 0.001, 0.9, 0.999, 1e-7   # Adam bawaan Keras (sama dengan penelitian)
ITERASI = 500
TITIK_RIWAYAT = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500]

# Bobot awal dibuat bulat agar mudah dihitung. Bias mengikuti aturan Keras:
# bias forget gate LSTM = 1 (unit_forget_bias), bias lainnya = 0.
LSTM_AWAL = {
    "W_f": 0.5, "U_f": 0.4, "b_f": 1.0,
    "W_i": 0.6, "U_i": 0.3, "b_i": 0.0,
    "W_c": 0.7, "U_c": 0.2, "b_c": 0.0,
    "W_o": 0.4, "U_o": 0.5, "b_o": 0.0,
    "W_y": 0.8, "b_y": 0.0,
}
GRU_AWAL = {
    "W_z": 0.5, "U_z": 0.4, "b_z_in": 0.0, "b_z_rec": 0.0,
    "W_r": 0.6, "U_r": 0.3, "b_r_in": 0.0, "b_r_rec": 0.0,
    "W_h": 0.7, "U_h": 0.2, "b_h_in": 0.0, "b_h_rec": 0.0,
    "W_y": 0.8, "b_y": 0.0,
}


def nama(kunci: str) -> str:
    """b_z_in -> b_z(in), W_f -> W_f."""
    for akhiran in ("_in", "_rec"):
        if kunci.endswith(akhiran):
            return f"{kunci[: -len(akhiran)]}({akhiran[1:]})"
    return kunci


# --------------------------------------------------------------------------- #
# Format angka (titik desimal, sama dengan berkas perhitungan manual lain)
# --------------------------------------------------------------------------- #
def _nol(x, d):
    return 0.0 if abs(x) < 0.5 * 10 ** (-d) else x


def a(x, d=6):
    """Angka biasa dengan d desimal."""
    return f"{_nol(x, d):.{d}f}"


def kr(x, d=6):
    """Angka untuk perkalian/penjumlahan: diberi kurung bila negatif."""
    x = _nol(x, d)
    return f"({x:.{d}f})" if x < 0 else f"{x:.{d}f}"


def bt(x):
    """Bobot awal yang bulat: 0.5, 1, 0."""
    return f"{x:g}"


def fl(x):
    """Nilai loss: 6 desimal, atau 10 desimal bila sudah sangat kecil."""
    return a(x, 6) if x >= 1e-4 else a(x, 10)


def ribu(n):
    return f"{n:,}"


def usd(x):
    return f"{x:,.2f}"


def persen(x, d=1):
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- #
# Model mini LSTM (persamaan 15-21)
# --------------------------------------------------------------------------- #
def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def lstm_maju(p):
    h, c = 0.0, 0.0
    langkah = []
    for x in X:
        s = {"x": x, "h_prev": h, "c_prev": c}
        for g in "fio":
            s["a_" + g] = p["W_" + g] * x + p["U_" + g] * h + p["b_" + g]
            s[g] = sig(s["a_" + g])
        s["a_c"] = p["W_c"] * x + p["U_c"] * h + p["b_c"]
        s["cc"] = math.tanh(s["a_c"])
        s["c"] = s["f"] * c + s["i"] * s["cc"]
        s["tc"] = math.tanh(s["c"])
        s["h"] = s["o"] * s["tc"]
        langkah.append(s)
        h, c = s["h"], s["c"]
    yhat = p["W_y"] * h + p["b_y"]
    return {"langkah": langkah, "yhat": yhat, "L": (Y - yhat) ** 2}


def lstm_mundur(p, maju):
    s_all = maju["langkah"]
    dy = -2.0 * (Y - maju["yhat"])
    g = {k: 0.0 for k in p}
    g["W_y"] = dy * s_all[-1]["h"]
    g["b_y"] = dy
    dh = dy * p["W_y"]
    dc_lanjut = 0.0
    rincian = [None] * len(s_all)
    for t in reversed(range(len(s_all))):
        s = s_all[t]
        jalur_h = dh * s["o"] * (1 - s["tc"] ** 2)
        dc = dc_lanjut + jalur_h
        r = {"dh": dh, "dc_lanjut": dc_lanjut, "jalur_h": jalur_h, "dc": dc}
        r["d_o"] = dh * s["tc"]
        r["delta_o"] = r["d_o"] * s["o"] * (1 - s["o"])
        r["d_f"] = dc * s["c_prev"]
        r["delta_f"] = r["d_f"] * s["f"] * (1 - s["f"])
        r["d_i"] = dc * s["cc"]
        r["delta_i"] = r["d_i"] * s["i"] * (1 - s["i"])
        r["d_cc"] = dc * s["i"]
        r["delta_c"] = r["d_cc"] * (1 - s["cc"] ** 2)
        for gerbang in "fico":
            d = r["delta_" + gerbang]
            g["W_" + gerbang] += d * s["x"]
            g["U_" + gerbang] += d * s["h_prev"]
            g["b_" + gerbang] += d
        r["suku_h"] = [p["U_" + gb] * r["delta_" + gb] for gb in "fico"]
        r["dh_prev"] = sum(r["suku_h"])
        r["dc_prev"] = dc * s["f"]
        rincian[t] = r
        dh, dc_lanjut = r["dh_prev"], r["dc_prev"]
    return {"dy": dy, "dh_T": dy * p["W_y"], "g": g, "rincian": rincian}


# --------------------------------------------------------------------------- #
# Model mini GRU, konvensi Keras reset_after=True (persamaan 23-26)
# --------------------------------------------------------------------------- #
def gru_maju(p):
    h = 0.0
    langkah = []
    for x in X:
        s = {"x": x, "h_prev": h}
        s["a_z"] = p["W_z"] * x + p["U_z"] * h + p["b_z_in"] + p["b_z_rec"]
        s["z"] = sig(s["a_z"])
        s["a_r"] = p["W_r"] * x + p["U_r"] * h + p["b_r_in"] + p["b_r_rec"]
        s["r"] = sig(s["a_r"])
        s["q"] = p["U_h"] * h + p["b_h_rec"]            # bagian rekuren kandidat
        s["masuk"] = p["W_h"] * x + p["b_h_in"]          # bagian masukan kandidat
        s["rq"] = s["r"] * s["q"]
        s["a_h"] = s["masuk"] + s["rq"]
        s["hh"] = math.tanh(s["a_h"])
        s["h"] = s["z"] * h + (1 - s["z"]) * s["hh"]
        langkah.append(s)
        h = s["h"]
    yhat = p["W_y"] * h + p["b_y"]
    return {"langkah": langkah, "yhat": yhat, "L": (Y - yhat) ** 2}


def gru_mundur(p, maju):
    s_all = maju["langkah"]
    dy = -2.0 * (Y - maju["yhat"])
    g = {k: 0.0 for k in p}
    g["W_y"] = dy * s_all[-1]["h"]
    g["b_y"] = dy
    dh = dy * p["W_y"]
    rincian = [None] * len(s_all)
    for t in reversed(range(len(s_all))):
        s = s_all[t]
        r = {"dh": dh}
        r["d_z"] = dh * (s["h_prev"] - s["hh"])
        r["delta_z"] = r["d_z"] * s["z"] * (1 - s["z"])
        r["d_hh"] = dh * (1 - s["z"])
        r["delta_h"] = r["d_hh"] * (1 - s["hh"] ** 2)
        r["d_r"] = r["delta_h"] * s["q"]
        r["delta_r"] = r["d_r"] * s["r"] * (1 - s["r"])
        r["d_q"] = r["delta_h"] * s["r"]
        g["W_z"] += r["delta_z"] * s["x"]
        g["U_z"] += r["delta_z"] * s["h_prev"]
        g["b_z_in"] += r["delta_z"]
        g["b_z_rec"] += r["delta_z"]
        g["W_r"] += r["delta_r"] * s["x"]
        g["U_r"] += r["delta_r"] * s["h_prev"]
        g["b_r_in"] += r["delta_r"]
        g["b_r_rec"] += r["delta_r"]
        g["W_h"] += r["delta_h"] * s["x"]
        g["b_h_in"] += r["delta_h"]
        g["U_h"] += r["d_q"] * s["h_prev"]
        g["b_h_rec"] += r["d_q"]
        r["suku_h"] = [dh * s["z"], p["U_z"] * r["delta_z"],
                       p["U_r"] * r["delta_r"], p["U_h"] * r["d_q"]]
        r["dh_prev"] = sum(r["suku_h"])
        rincian[t] = r
        dh = r["dh_prev"]
    return {"dy": dy, "dh_T": dy * p["W_y"], "g": g, "rincian": rincian}


# --------------------------------------------------------------------------- #
# Pemeriksaan gradien numerik dan pelatihan Adam
# --------------------------------------------------------------------------- #
def gradien_numerik(fungsi_maju, p, langkah=1e-6):
    hasil = {}
    for k in p:
        naik, turun = dict(p), dict(p)
        naik[k] += langkah
        turun[k] -= langkah
        hasil[k] = (fungsi_maju(naik)["L"] - fungsi_maju(turun)["L"]) / (2 * langkah)
    return hasil


def latih_adam(fungsi_maju, fungsi_mundur, p_awal, iterasi):
    p = dict(p_awal)
    m = {k: 0.0 for k in p}
    v = {k: 0.0 for k in p}
    riwayat = []
    for k in range(1, iterasi + 1):
        maju = fungsi_maju(p)
        g = fungsi_mundur(p, maju)["g"]
        catat = {"k": k, "L": maju["L"], "yhat": maju["yhat"], "p": dict(p), "g": g,
                 "m_lama": dict(m), "v_lama": dict(v), "m": {}, "v": {},
                 "mh": {}, "vh": {}, "delta": {}}
        for n in p:
            m[n] = B1 * m[n] + (1 - B1) * g[n]
            v[n] = B2 * v[n] + (1 - B2) * g[n] ** 2
            mh = m[n] / (1 - B1 ** k)
            vh = v[n] / (1 - B2 ** k)
            delta = LR * mh / (math.sqrt(vh) + EPS)
            catat["m"][n], catat["v"][n] = m[n], v[n]
            catat["mh"][n], catat["vh"][n], catat["delta"][n] = mh, vh, delta
            p[n] -= delta
        riwayat.append(catat)
    akhir = fungsi_maju(p)
    # urutan_setelah[j] = keadaan setelah j kali update (j = 0..iterasi)
    urutan_p = [r["p"] for r in riwayat] + [dict(p)]
    urutan_L = [r["L"] for r in riwayat] + [akhir["L"]]
    urutan_y = [r["yhat"] for r in riwayat] + [akhir["yhat"]]
    return {"riwayat": riwayat, "p": urutan_p, "L": urutan_L, "yhat": urutan_y}


# --------------------------------------------------------------------------- #
# Angka konteks penelitian (dibaca dari keluaran notebook)
# --------------------------------------------------------------------------- #
def _csv(pola):
    berkas = sorted(glob.glob(os.path.join(DIR_TABEL, pola)))
    assert berkas, f"Tabel {pola} tidak ditemukan di {DIR_TABEL}"
    with open(berkas[0], encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _model_terbaik(pola):
    baris = [b for b in _csv(pola) if "TERBAIK" in (b.get("Keterangan") or "")]
    assert len(baris) == 1, f"Model terbaik pada {pola} tidak ditemukan"
    b = baris[0]
    return {"neuron": int(b["Neuron"]), "epoch": int(b["Epoch"]),
            "rmse_val": float(b["RMSE Validasi (USD)"]),
            "param": int(b["Jumlah Parameter"])}


def _angka(pola, teks, jumlah):
    hasil = [float(v) for v in re.findall(pola, teks, flags=re.M)]
    assert len(hasil) == jumlah, f"Pola {pola!r}: ditemukan {len(hasil)}, diharapkan {jumlah}"
    return hasil


def baca_konteks():
    k = {}
    close = [b for b in _csv("tabel_08_*.csv") if b["Nama Kolom"] == "close_price"][0]
    k["x_min"] = float(close["x_min (data latih)"])
    k["x_max"] = float(close["x_max (data latih)"])
    k["rentang"] = k["x_max"] - k["x_min"]
    k["lstm"] = _model_terbaik("tabel_09_*.csv")
    k["gru"] = _model_terbaik("tabel_11_*.csv")

    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    sumber, keluaran = [], []
    for sel in nb["cells"]:
        if sel["cell_type"] != "code":
            continue
        sumber.append("".join(sel["source"]))
        for o in sel.get("outputs", []):
            if "text" in o:
                keluaran.append("".join(o["text"]))
    sumber, keluaran = "\n".join(sumber), "\n".join(keluaran)
    bentuk = re.findall(r"shape X_latih\s+=\s+\((\d+), (\d+), (\d+)\)", keluaran)
    assert len(bentuk) == 1, "Bentuk X_latih tidak ditemukan pada keluaran notebook"
    k["n_latih"], k["lookback"], k["n_fitur"] = (int(v) for v in bentuk[0])
    k["batch"] = int(_angka(r'"BATCH_SIZE":\s*(\d+)', sumber, 1)[0])
    k["lr"] = _angka(r'"LEARNING_RATE":\s*([0-9.]+)', sumber, 1)[0]
    k["val_loss_lstm"], k["val_loss_gru"] = _angka(
        r"Loss validasi epoch terakhir:\s+([0-9.]+)", keluaran, 2)

    with open(os.path.join(DIR_MANUAL, "tahap_7_forward_pass_lstm.md"), encoding="utf-8") as f:
        teks = f.read()
    b_lstm = _angka(r"^[ \t]*b\[1\][ \t]+\(bias\)[ \t]+=[ \t]+([-+]?\d+\.\d+)", teks, 4)   # urutan Keras i, f, c, o
    k["bias_lstm"] = dict(zip("ifco", b_lstm))
    k["by_lstm"] = _angka(r"b_y \(bias Dense\)\s+=\s+([-+]?\d+\.\d+)", teks, 1)[0]

    with open(os.path.join(DIR_MANUAL, "tahap_8_forward_pass_gru.md"), encoding="utf-8") as f:
        teks = f.read()
    b_in = _angka(r"^[ \t]*b_in\[1\][ \t]+=[ \t]+([-+]?\d+\.\d+)", teks, 3)              # urutan Keras z, r, h
    b_rec = _angka(r"^[ \t]*b_rec\[1\][ \t]+=[ \t]+([-+]?\d+\.\d+)", teks, 3)
    k["bias_gru"] = {g: (bi, br) for g, bi, br in zip("zrh", b_in, b_rec)}
    k["by_gru"] = _angka(r"b_y \(bias Dense\)\s+=\s+([-+]?\d+\.\d+)", teks, 1)[0]

    k["iter_epoch"] = math.ceil(k["n_latih"] / k["batch"])
    k["batch_akhir"] = k["n_latih"] - (k["iter_epoch"] - 1) * k["batch"]
    return k


# --------------------------------------------------------------------------- #
# Penyusun markdown
# --------------------------------------------------------------------------- #
class Dokumen:
    def __init__(self):
        self.baris = []

    def teks(self, s=""):
        self.baris.append(s.strip("\n"))
        self.baris.append("")

    def kode(self, isi):
        if isinstance(isi, str):
            isi = isi.strip("\n").split("\n")
        while isi and not isi[-1].strip():
            isi = isi[:-1]
        self.baris.append("```")
        self.baris.extend(isi)
        self.baris.append("```")
        self.baris.append("")

    def tabel(self, kepala, isi):
        self.baris.append("| " + " | ".join(kepala) + " |")
        self.baris.append("|" + "|".join("---" for _ in kepala) + "|")
        for b in isi:
            self.baris.append("| " + " | ".join(str(x) for x in b) + " |")
        self.baris.append("")

    def isi(self):
        return "\n".join(self.baris).rstrip() + "\n"


def jangkar(judul):
    """Jangkar tautan judul ala GitHub."""
    s = judul.strip().lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch in " -_")
    return s.replace(" ", "-")


# --------------------------------------------------------------------------- #
# Bagian dokumen
# --------------------------------------------------------------------------- #
JUDUL = {
    0: "0. Gambaran Besar: Satu Siklus Pembobotan",
    1: "1. Asumsi Simulasi dan Notasi",
    2: "2. Rumus Turunan Dasar yang Dipakai",
    3: "3. LSTM: Satu Siklus Pelatihan Langkah demi Langkah",
    4: "4. GRU: Satu Siklus Pelatihan Langkah demi Langkah",
    5: "5. Bias: Mengapa Perlu dan Bagaimana Dihitung",
    6: "6. Dari Model Mini ke Model Penelitian",
    7: "7. Ringkasan Alur dan Hasil Verifikasi",
}


def bagian_pembuka(d, k):
    d.teks("# Perhitungan Manual - Tahap 7 & 8: Simulasi Pelatihan Bobot LSTM dan GRU")
    d.teks("""
> Berkas ini dibuat otomatis oleh `tools/simulasi_pelatihan_manual.py`.
> Semua angka dihitung ulang oleh skrip itu dan diperiksa dengan `assert`
> (lihat bagian 7). Untuk membuat ulang: `python tools/simulasi_pelatihan_manual.py`.
""")
    d.teks("""
Berkas `tahap_7_forward_pass_lstm.md` dan `tahap_8_forward_pass_gru.md`
menunjukkan bagaimana model yang **sudah dilatih** menghasilkan prediksi.
Berkas ini menjawab pertanyaan sebelumnya: **bagaimana bobot dan bias itu
diperoleh?** Jawabannya adalah pelatihan, yaitu satu siklus yang diulang
ribuan kali: *forward pass* → *loss* → *backpropagation through time* (BPTT)
→ Adam.

Agar setiap angka bisa diikuti dengan tangan, simulasi memakai **model mini**:
1 neuron, 1 fitur, 2 time step, dan 1 sampel. Rumus dan urutan langkahnya
**identik** dengan model penelitian; yang berbeda hanya jumlah angkanya.
Bagian 6 menjelaskan cara memperbesarnya ke model penelitian
(10 fitur, 7 time step, batch 32, 40/30 neuron).
""")
    d.teks("**Daftar isi**")
    d.teks("\n".join(f"- [{JUDUL[i]}](#{jangkar(JUDUL[i])})" for i in sorted(JUDUL)))
    d.teks("""
**Cara membaca angka.** Seperti berkas perhitungan manual lainnya, angka memakai
titik sebagai pemisah desimal dan koma sebagai pemisah ribuan. Angka
ditampilkan 6 desimal, tetapi skrip menghitung dengan presisi penuh, sehingga
selisih pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka
yang tampil. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.8 sampai 1.5.11).
""")


def bagian_0(d, k):
    total_lstm = k["iter_epoch"] * k["lstm"]["epoch"]
    total_gru = k["iter_epoch"] * k["gru"]["epoch"]
    d.teks(f"## {JUDUL[0]}")
    d.teks("""
Model "belajar" dengan mengulang satu siklus yang sama. Satu siklus disebut
**satu iterasi k**:
""")
    d.kode("""
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
 Bobot & bias θ ─► (1) FORWARD ─► (2) LOSS ─► (3) BACKWARD (BPTT) ─► (4) ADAM
                    prediksi ŷ'    seberapa     gradien g = ∂L/∂θ      θ baru
                                   salah?       untuk setiap θ
""")
    d.tabel(
        ["Langkah", "Pertanyaan yang dijawab", "Persamaan skripsi", "Di berkas ini"],
        [["(1) Forward pass", "Dengan bobot sekarang, berapa prediksinya?",
          "(15)-(21) LSTM, (23)-(26) GRU", "3.2, 4.2"],
         ["(2) Loss", "Seberapa salah prediksinya?", "(28)", "3.3, 4.3"],
         ["(3) Backward (BPTT)", "Bobot mana yang menyebabkan salah, ke arah mana?",
          "aturan rantai (bagian 2)", "3.4-3.9, 4.4-4.9"],
         ["(4) Adam", "Seberapa jauh setiap bobot digeser?", "(29)-(31)",
          "3.10-3.13, 4.10-4.13"]])
    d.teks(f"""
**Berapa kali siklus ini terjadi di penelitian?** Data latih berisi
{ribu(k['n_latih'])} sampel (jendela {k['lookback']} hari × {k['n_fitur']} fitur) dengan
batch size {k['batch']}. Jadi satu epoch terdiri dari
⌈{ribu(k['n_latih'])} / {k['batch']}⌉ = **{k['iter_epoch']} iterasi**
({k['iter_epoch'] - 1} batch berisi {k['batch']} sampel dan 1 batch terakhir berisi
{k['batch_akhir']} sampel).

- LSTM terbaik ({k['lstm']['neuron']} neuron, {ribu(k['lstm']['epoch'])} epoch):
  {k['iter_epoch']} × {ribu(k['lstm']['epoch'])} = **{ribu(total_lstm)} kali update**.
- GRU terbaik ({k['gru']['neuron']} neuron, {ribu(k['gru']['epoch'])} epoch):
  {k['iter_epoch']} × {ribu(k['gru']['epoch'])} = **{ribu(total_gru)} kali update**.
""")
    d.teks("**Kapan loss dan Adam dipakai?** Keduanya hanya bekerja saat pelatihan.")
    d.tabel(
        ["Tahap penelitian", "Loss MSE", "Adam"],
        [["Inisialisasi model", "-", "m = 0, v = 0, k = 0"],
         [f"Setiap batch latih ({k['iter_epoch']}× per epoch)",
          "dihitung, menjadi sumber gradien", "memperbarui semua bobot dan bias"],
         ["Akhir setiap epoch", "loss latih dan loss validasi dicatat (kurva loss)",
          "tidak ada update dari data validasi"],
         ["Pemilihan neuron dan epoch terbaik", "tidak (memakai RMSE validasi USD)", "tidak"],
         ["Prediksi data uji dan evaluasi", "tidak", "tidak (bobot sudah beku)"]])


def bagian_1(d, k):
    d.teks(f"## {JUDUL[1]}")
    d.tabel(["Komponen", "Model mini (berkas ini)", "Model penelitian"],
            [["Neuron", "1", f"LSTM {k['lstm']['neuron']}, GRU {k['gru']['neuron']}"],
             ["Fitur per hari", "1 (harga penutupan ternormalisasi)", str(k["n_fitur"])],
             ["Time step (window)", "2", str(k["lookback"])],
             ["Sampel per iterasi", "1 (N = 1)", f"{k['batch']} (batch size)"],
             ["Optimizer", "Adam, η = 0.001", f"Adam, η = {k['lr']:g}"]])
    d.teks(f"""
**Data.** Harga penutupan ternormalisasi dua hari dipakai untuk memprediksi hari
ketiga (angka ilustrasi):

- hari 1: x₁ = {X[0]}
- hari 2: x₂ = {X[1]}
- target hari 3: y′ = {Y}

**Keadaan awal.** h₀ = 0 dan c₀ = 0, sama seperti bawaan Keras.

**Adam.** η = {LR}, β₁ = {B1}, β₂ = {B2}, ε = 10⁻⁷ (bawaan Keras, sama dengan penelitian).

**Bobot awal.** Dibuat bulat agar mudah dihitung. Keras sebenarnya mengisi
bobot secara acak (Glorot uniform untuk W, ortogonal untuk U), tetapi aturan
biasnya diikuti: **bias forget gate LSTM = 1**, bias lain = 0.
""")
    d.teks("**Notasi yang dipakai:**")
    d.tabel(["Simbol", "Arti"],
            [["a", "pra-aktivasi: nilai sebelum σ atau tanh, misalnya a = W·x + U·h + b"],
             ["σ, tanh", "fungsi aktivasi sigmoid (persamaan 11) dan tanh (persamaan 12)"],
             ["∂L/∂θ", "turunan parsial loss terhadap θ; variabel lain dianggap konstan"],
             ["δ", "sinyal kesalahan sebuah gerbang = ∂L/∂a (turunan loss terhadap pra-aktivasinya)"],
             ["g", "gradien sebuah parameter = ∂L/∂θ"],
             ["m, v", "momen pertama dan kedua Adam (persamaan 29)"],
             ["k", "nomor iterasi (satu kali update = satu batch)"]])


def bagian_2(d, k):
    s = sig(1.25)
    d.teks(f"## {JUDUL[2]}")
    d.teks("""
Seluruh backward pass hanya mengulang beberapa rumus turunan berikut. Polanya
selalu sama: **kalikan sinyal kesalahan dengan turunan lokal, lalu teruskan ke
kiri (ke langkah sebelumnya).**
""")
    d.tabel(["Rumus", "Bentuk", "Contoh", "Dipakai di"],
            [["Aturan pangkat", "d(u²)/du = 2u", "d(x²)/dx = 2x", "∂L/∂ŷ′"],
             ["Fungsi linear", "d(a·u + b)/du = a", "d(3x + 5)/dx = 3",
              "pra-aktivasi dan dense: ∂a/∂W = x, ∂a/∂U = hₜ₋₁, ∂a/∂b = 1"],
             ["Aturan perkalian", "∂(u·v)/∂u = v", "∂(3u)/∂u = 3",
              "h = o·tanh(c), c = f·cₜ₋₁ + i·c̃, GRU h = z·hₜ₋₁ + (1−z)·h̃"],
             ["Aturan rantai", "∂L/∂a = ∂L/∂b · ∂b/∂a", "-", "semua langkah"],
             ["Aturan penjumlahan", "variabel yang dipakai di beberapa tempat: gradiennya dijumlah",
              "-", "h₁ dipakai 4 gerbang; bobot dipakai di setiap t; c₁ punya 2 jalur"],
             ["Turunan sigmoid", "σ′(a) = σ(a)·(1 − σ(a))",
              f"σ(1.25) = {a(s)} → σ′ = {a(s * (1 - s))}", "gerbang f, i, o, z, r"],
             ["Turunan tanh", "tanh′(a) = 1 − tanh²(a)", f"tanh(0.35) = {a(math.tanh(0.35))} → "
              f"{a(1 - math.tanh(0.35) ** 2)}", "c̃, h̃, tanh(c)"]])
    d.teks("**Asal turunan sigmoid** (dari persamaan 11):")
    d.teks(r"""
$$\sigma(a) = \frac{1}{1+e^{-a}}
\;\Rightarrow\;
\sigma'(a) = \frac{e^{-a}}{(1+e^{-a})^2}
= \frac{1}{1+e^{-a}}\cdot\frac{e^{-a}}{1+e^{-a}}
= \sigma(a)\,\bigl(1-\sigma(a)\bigr)$$
""")
    d.teks("**Asal turunan tanh** (dari persamaan 13, $\\tanh(a) = 2\\sigma(2a) - 1$):")
    d.teks(r"""
$$\tanh'(a) = 4\,\sigma(2a)\bigl(1-\sigma(2a)\bigr)
= 4\cdot\frac{1+\tanh(a)}{2}\cdot\frac{1-\tanh(a)}{2}
= 1-\tanh^2(a)$$
""")


# ------------------------------- LSTM -------------------------------------- #
def blok_gerbang(baris, judul, simbol, kode_bobot, x, h_prev, akt, aktivasi, x_nama, h_nama, bias_suku):
    """Rincian pra-aktivasi satu gerbang LSTM/GRU pada satu time step."""
    W, U = kode_bobot
    bagian_bias = " + ".join(bt(b) for b in bias_suku)
    baris.append(judul)
    nama_bias = " + ".join(n for n, _ in akt["bias_nama"])
    baris.append(f"  a = {W} × {x_nama} + {U} × {h_nama} + {nama_bias}")
    baris.append(f"    = {bt(akt['W'])} × {a(x)} + {bt(akt['U'])} × {a(h_prev)} + {bagian_bias}")
    baris.append(f"    = {a(akt['W'] * x)} + {a(akt['U'] * h_prev)} + {a(sum(bias_suku))}")
    baris.append(f"    = {a(akt['a'])}")
    if aktivasi == "sigmoid":
        baris.append(f"  {simbol} = σ({a(akt['a'])}) = 1 / (1 + e^(-{kr(akt['a'])})) = {a(akt['nilai'])}")
    else:
        baris.append(f"  {simbol} = tanh({a(akt['a'])}) = {a(akt['nilai'])}")


def bagian_3(d, k, p, maju, mundur, num, latih):
    s1, s2 = maju["langkah"]
    r1, r2 = mundur["rincian"]
    g = mundur["g"]
    d.teks(f"## {JUDUL[3]}")

    # 3.1 ------------------------------------------------------------------
    d.teks("### 3.1 Bobot Awal dan Jumlah Parameter")
    d.tabel(["Gerbang", "W (bobot masukan)", "U (bobot rekuren)", "b (bias)"],
            [["forget (f)", bt(p["W_f"]), bt(p["U_f"]), bt(p["b_f"]) + " (unit forget bias)"],
             ["input (i)", bt(p["W_i"]), bt(p["U_i"]), bt(p["b_i"])],
             ["kandidat (c̃)", bt(p["W_c"]), bt(p["U_c"]), bt(p["b_c"])],
             ["output (o)", bt(p["W_o"]), bt(p["U_o"]), bt(p["b_o"])],
             ["dense", f"W_y = {bt(p['W_y'])}", "-", f"b_y = {bt(p['b_y'])}"]])
    d.kode([
        "Jumlah parameter (persamaan 22) dengan n_u = 1 neuron dan n_f = 1 fitur:",
        "  P_LSTM = 4 × [n_u × (n_u + n_f) + n_u] + (n_u + 1)",
        "         = 4 × [1 × (1 + 1) + 1] + (1 + 1)",
        "         = 4 × 3 + 2",
        f"         = {len(p)} parameter  (12 di lapisan LSTM + 2 di lapisan dense)",
    ])
    assert len(p) == 4 * (1 * 2 + 1) + 2

    # 3.2 ------------------------------------------------------------------
    d.teks("### 3.2 Langkah 1: Forward Pass")
    d.teks(r"""
Rumus versi 1 neuron (persamaan 15-21). Karena hanya ada satu neuron, semua
bobot berupa angka tunggal (skalar), bukan matriks:

$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \qquad
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \qquad
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

$$c_t = f_t\,c_{t-1} + i_t\,\tilde{c}_t \qquad
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \qquad
h_t = o_t \tanh(c_t) \qquad
\hat{y}' = W_y h_T + b_y$$
""")
    baris = []
    for t, s in ((1, s1), (2, s2)):
        xn, hn, cn = f"x{'₁₂'[t - 1]}", f"h{'₀₁'[t - 1]}", f"c{'₀₁'[t - 1]}"
        sub = "₁₂"[t - 1]
        baris.append("=" * 70)
        baris.append(f"TIME STEP t = {t}   ({xn} = {a(s['x'])}, {hn} = {a(s['h_prev'])}, "
                     f"{cn} = {a(s['c_prev'])})")
        baris.append("=" * 70)
        for gb, judul, akt in (("f", "Forget gate (persamaan 15, Gambar 3)", "sigmoid"),
                               ("i", "Input gate (persamaan 16, Gambar 4)", "sigmoid"),
                               ("c", "Kandidat cell state (persamaan 17, Gambar 4)", "tanh"),
                               ("o", "Output gate (persamaan 19, Gambar 6)", "sigmoid")):
            simbol = {"f": "f", "i": "i", "c": "c̃", "o": "o"}[gb] + sub
            nilai = s["cc"] if gb == "c" else s[gb]
            blok_gerbang(baris, judul, simbol, (f"W_{gb}", f"U_{gb}"), s["x"], s["h_prev"],
                         {"W": p["W_" + gb], "U": p["U_" + gb], "a": s["a_" + gb], "nilai": nilai,
                          "bias_nama": [(f"b_{gb}", p["b_" + gb])]},
                         akt, xn, hn, [p["b_" + gb]])
        baris.append("Pembaruan cell state (persamaan 18, Gambar 5)")
        baris.append(f"  c{sub} = f{sub} × {cn} + i{sub} × c̃{sub}")
        baris.append(f"     = {a(s['f'])} × {a(s['c_prev'])} + {a(s['i'])} × {a(s['cc'])}")
        baris.append(f"     = {a(s['f'] * s['c_prev'])} + {a(s['i'] * s['cc'])}")
        baris.append(f"     = {a(s['c'])}")
        baris.append("Hidden state (persamaan 20, Gambar 6)")
        baris.append(f"  h{sub} = o{sub} × tanh(c{sub}) = {a(s['o'])} × tanh({a(s['c'])})"
                     f" = {a(s['o'])} × {a(s['tc'])} = {a(s['h'])}")
        baris.append("")
    baris.append("=" * 70)
    baris.append("LAPISAN DENSE (persamaan 21) — hanya h₂ (time step terakhir) yang dipakai")
    baris.append("=" * 70)
    baris.append(f"  ŷ' = W_y × h₂ + b_y = {bt(p['W_y'])} × {a(s2['h'])} + {bt(p['b_y'])} = {a(maju['yhat'])}")
    d.kode(baris)
    d.tabel(["t", "x", "f", "i", "c̃", "o", "c", "h"],
            [[t, a(s["x"]), a(s["f"]), a(s["i"]), a(s["cc"]), a(s["o"]), a(s["c"]), a(s["h"])]
             for t, s in ((1, s1), (2, s2))])

    # 3.3 ------------------------------------------------------------------
    d.teks("### 3.3 Langkah 2: Loss")
    d.kode([
        "Persamaan (28) dengan N = 1 sampel:",
        "  L = (y' - ŷ')²",
        f"    = ({a(Y, 1)} - {a(maju['yhat'])})²",
        f"    = {a(Y - maju['yhat'])}²",
        f"    = {a(maju['L'])}",
    ])
    d.teks(f"""
Prediksi masih **terlalu rendah** ({a(maju['yhat'])} padahal seharusnya {Y}).
Langkah berikutnya mencari tahu bobot mana yang perlu diubah agar loss ini turun.
""")

    # 3.4 ------------------------------------------------------------------
    d.teks("### 3.4 Langkah 3a: Backward di Lapisan Dense")
    d.teks("""
Backward berjalan dari kanan ke kiri, mulai dari loss. Rantainya:
`h₂ → ŷ' = W_y·h₂ + b_y → L = (y' − ŷ')²`. Setiap turunan memakai rumus
dari bagian 2.
""")
    dy = mundur["dy"]
    d.kode([
        f"Diketahui: h₂ = {a(s2['h'])}, W_y = {bt(p['W_y'])}, b_y = {bt(p['b_y'])}, "
        f"ŷ' = {a(maju['yhat'])}, y' = {Y}",
        "",
        "(1) ∂L/∂ŷ' — seberapa besar loss berubah bila prediksi berubah",
        "    L = (y' - ŷ')²",
        "    misalkan u = y' - ŷ'           →  L = u²",
        "    dL/du  = 2u                     (aturan pangkat)",
        "    du/dŷ' = 0 - 1 = -1             (y' adalah data, turunannya 0)",
        "    ∂L/∂ŷ' = 2u × (-1) = -2(y' - ŷ')   (aturan rantai)",
        f"           = -2 × ({Y} - {a(maju['yhat'])})",
        f"           = -2 × {a(Y - maju['yhat'])}",
        f"           = {a(dy)}",
        "    Cara lain tanpa aturan rantai: L = y'² - 2y'ŷ' + ŷ'²",
        "                                   ∂L/∂ŷ' = -2y' + 2ŷ' = 2(ŷ' - y')  (sama)",
        "",
        "(2) ∂L/∂W_y — gradien bobot dense",
        "    ŷ' = W_y × h₂ + b_y  →  ∂ŷ'/∂W_y = h₂   (fungsi linear; h₂ dan b_y dianggap konstan)",
        f"    ∂L/∂W_y = ∂L/∂ŷ' × h₂ = {kr(dy)} × {a(s2['h'])} = {a(g['W_y'])}",
        "",
        "(3) ∂L/∂b_y — gradien bias dense",
        "    ∂ŷ'/∂b_y = 1",
        f"    ∂L/∂b_y = ∂L/∂ŷ' × 1 = {a(g['b_y'])}",
        "",
        "(4) ∂L/∂h₂ — sinyal kesalahan yang dikirim MASUK ke sel LSTM (awal BPTT)",
        "    ∂ŷ'/∂h₂ = W_y            (kali ini h₂ yang menjadi variabel)",
        f"    ∂L/∂h₂ = ∂L/∂ŷ' × W_y = {kr(dy)} × {bt(p['W_y'])} = {a(mundur['dh_T'])}",
    ])
    d.teks(f"""
**Makna setiap angka:**

- **∂L/∂ŷ′ = {a(dy)}**: tanda negatif berarti menaikkan prediksi akan
  menurunkan loss (sesuai, karena prediksi masih terlalu rendah). Jika ŷ′ naik
  0.001, loss turun sekitar {a(abs(dy) * 0.001, 7)}. Nilai ini adalah **sinyal
  kesalahan keluaran** yang dipakai ulang oleh tiga turunan berikutnya.
- **∂L/∂W_y = {a(g['W_y'])}**: pengaruh W_y terhadap prediksi "dikali" h₂.
  Karena h₂ hanya {a(s2['h'])}, gradien W_y lebih kecil daripada gradien ŷ′.
- **∂L/∂b_y = {a(g['b_y'])}**: bias langsung ditambahkan ke prediksi, jadi
  gradiennya sama dengan sinyal kesalahan keluaran.
- **∂L/∂h₂ = {a(mundur['dh_T'])}**: h₂ **bukan parameter**, jadi tidak
  diupdate Adam. Nilai ini adalah pintu masuk BPTT ke dalam sel LSTM.
""")
    # uji geser
    L0 = maju["L"]
    geser = 0.001
    uji = [
        ("ŷ'", dy, (Y - (maju["yhat"] + geser)) ** 2 - L0),
        ("W_y (0.8 → 0.801)", g["W_y"], (Y - ((p["W_y"] + geser) * s2["h"] + p["b_y"])) ** 2 - L0),
        ("b_y (0 → 0.001)", g["b_y"], (Y - (p["W_y"] * s2["h"] + p["b_y"] + geser)) ** 2 - L0),
        ("h₂", mundur["dh_T"], (Y - (p["W_y"] * (s2["h"] + geser) + p["b_y"])) ** 2 - L0),
    ]
    for _, grad, beda in uji:
        assert abs(grad * geser - beda) < 2e-6
    d.teks("""
**Bukti dengan uji geser.** Turunan berarti "perubahan loss bila variabel
digeser sedikit". Setiap besaran digeser +0.001, lalu loss dihitung ulang:
""")
    d.tabel(["Yang digeser +0.001", "Perkiraan dari gradien (gradien × 0.001)", "Perubahan loss sebenarnya"],
            [[n, f"{kr(gr)} × 0.001 = {a(gr * geser, 7)}", a(bd, 7)] for n, gr, bd in uji])
    d.teks("""
Selisih kecil (sekitar 0.000001) muncul karena turunan adalah pendekatan garis
lurus, sedangkan loss berbentuk kuadrat; sisanya sebesar (0.001 × ∂ŷ′/∂θ)².
""")

    # 3.5 ------------------------------------------------------------------
    d.teks("### 3.5 Langkah 3b: Backward di Dalam Sel, t = 2")
    d.teks("""
Sinyal ∂L/∂h₂ masuk ke sel lewat dua rumus forward yang diturunkan:
`h₂ = o₂ × tanh(c₂)` (menuju output gate dan cell state), lalu
`c₂ = f₂ × c₁ + i₂ × c̃₂` (menuju forget gate, input gate, dan kandidat).
Setiap gerbang lalu melewati turunan aktivasinya (σ′ atau tanh′) sehingga
diperoleh sinyal kesalahan gerbang **δ**.
""")
    tc2 = s2["tc"]
    d.kode([
        f"Masukan dari 3.4: ∂L/∂h₂ = {a(r2['dh'])}",
        "",
        "(a) Output gate — turunkan h₂ = o₂ × tanh(c₂) terhadap o₂ (aturan perkalian)",
        f"    ∂L/∂o₂ = ∂L/∂h₂ × tanh(c₂) = {kr(r2['dh'])} × {a(tc2)} = {a(r2['d_o'])}",
        "    δo₂    = ∂L/∂o₂ × o₂(1 - o₂)                     (turunan sigmoid)",
        f"           = {kr(r2['d_o'])} × {a(s2['o'])} × {a(1 - s2['o'])}",
        f"           = {kr(r2['d_o'])} × {a(s2['o'] * (1 - s2['o']))}",
        f"           = {a(r2['delta_o'])}",
        "",
        "(b) Cell state — turunkan h₂ = o₂ × tanh(c₂) terhadap c₂ (perkalian + turunan tanh)",
        "    ∂L/∂c₂ = ∂L/∂h₂ × o₂ × (1 - tanh²(c₂))",
        f"           = {kr(r2['dh'])} × {a(s2['o'])} × (1 - {a(tc2)}²)",
        f"           = {kr(r2['dh'])} × {a(s2['o'])} × {a(1 - tc2 ** 2)}",
        f"           = {a(r2['dc'])}",
        "    (t = 2 adalah time step terakhir, jadi belum ada kiriman dari t = 3)",
        "",
        "(c) Forget gate — turunkan c₂ = f₂ × c₁ + i₂ × c̃₂ terhadap f₂",
        f"    ∂L/∂f₂ = ∂L/∂c₂ × c₁ = {kr(r2['dc'])} × {a(s2['c_prev'])} = {a(r2['d_f'])}",
        f"    δf₂    = ∂L/∂f₂ × f₂(1 - f₂) = {kr(r2['d_f'])} × {a(s2['f'] * (1 - s2['f']))} = {a(r2['delta_f'])}",
        "",
        "(d) Input gate — turunkan c₂ terhadap i₂",
        f"    ∂L/∂i₂ = ∂L/∂c₂ × c̃₂ = {kr(r2['dc'])} × {a(s2['cc'])} = {a(r2['d_i'])}",
        f"    δi₂    = ∂L/∂i₂ × i₂(1 - i₂) = {kr(r2['d_i'])} × {a(s2['i'] * (1 - s2['i']))} = {a(r2['delta_i'])}",
        "",
        "(e) Kandidat — turunkan c₂ terhadap c̃₂",
        f"    ∂L/∂c̃₂ = ∂L/∂c₂ × i₂ = {kr(r2['dc'])} × {a(s2['i'])} = {a(r2['d_cc'])}",
        f"    δc̃₂    = ∂L/∂c̃₂ × (1 - c̃₂²) = {kr(r2['d_cc'])} × {a(1 - s2['cc'] ** 2)} = {a(r2['delta_c'])}",
    ])

    # 3.6 ------------------------------------------------------------------
    d.teks("### 3.6 Langkah 3c: Mengirim Kesalahan ke t = 1 (Inti \"Through Time\")")
    d.teks("""
Kesalahan di t = 2 sebagian disebabkan oleh keadaan hari sebelumnya, jadi
sinyalnya dikirim mundur ke t = 1 lewat **dua jalur**:

1. **Jalur hidden state h₁.** h₁ dipakai oleh keempat gerbang di t = 2
   (lewat bobot U), jadi keempat kontribusinya dijumlahkan (aturan penjumlahan).
2. **Jalur cell state c₁.** Dari `c₂ = f₂ × c₁ + ...`, turunannya terhadap c₁
   adalah f₂. Jalur ini hanya berupa perkalian sederhana, itulah sebabnya
   disebut "jalan tol gradien".
""")
    su = r2["suku_h"]
    d.kode([
        "Jalur 1 — hidden state:",
        "  ∂L/∂h₁ = U_f × δf₂ + U_i × δi₂ + U_c × δc̃₂ + U_o × δo₂",
        f"         = {bt(p['U_f'])} × {kr(r2['delta_f'])} + {bt(p['U_i'])} × {kr(r2['delta_i'])}"
        f" + {bt(p['U_c'])} × {kr(r2['delta_c'])} + {bt(p['U_o'])} × {kr(r2['delta_o'])}",
        f"         = {kr(su[0])} + {kr(su[1])} + {kr(su[2])} + {kr(su[3])}",
        f"         = {a(r2['dh_prev'])}",
        "",
        "Jalur 2 — cell state:",
        f"  kiriman ke c₁ = ∂L/∂c₂ × f₂ = {kr(r2['dc'])} × {a(s2['f'])} = {a(r2['dc_prev'])}",
    ])

    # 3.7 ------------------------------------------------------------------
    porsi = r1["dc_lanjut"] / r1["dc"]
    d.teks("### 3.7 Langkah 3d: Backward di Dalam Sel, t = 1")
    d.teks("""
Di t = 1 rumusnya sama dengan 3.5. Bedanya, cell state c₁ menerima **dua
kiriman** (dari jalur cell state dan dari h₁), lalu keduanya dijumlahkan.
""")
    d.kode([
        f"Masukan dari 3.6: ∂L/∂h₁ = {a(r1['dh'])}, kiriman jalur cell state = {a(r1['dc_lanjut'])}",
        "",
        "(a) Output gate",
        f"    ∂L/∂o₁ = ∂L/∂h₁ × tanh(c₁) = {kr(r1['dh'])} × {a(s1['tc'])} = {a(r1['d_o'])}",
        f"    δo₁    = {kr(r1['d_o'])} × {a(s1['o'] * (1 - s1['o']))} = {a(r1['delta_o'])}",
        "",
        "(b) Cell state c₁ = [kiriman jalur cell state] + [kiriman lewat h₁ = o₁ × tanh(c₁)]",
        "    ∂L/∂c₁ = ∂L/∂c₂ × f₂ + ∂L/∂h₁ × o₁ × (1 - tanh²(c₁))",
        f"           = {kr(r1['dc_lanjut'])} + {kr(r1['dh'])} × {a(s1['o'])} × {a(1 - s1['tc'] ** 2)}",
        f"           = {kr(r1['dc_lanjut'])} + {kr(r1['jalur_h'])}",
        f"           = {a(r1['dc'])}",
        f"    Porsi jalur cell state = {a(r1['dc_lanjut'])} / {a(r1['dc'])} = {persen(porsi)}",
        "",
        "(c) Forget gate",
        f"    ∂L/∂f₁ = ∂L/∂c₁ × c₀ = {kr(r1['dc'])} × {a(s1['c_prev'])} = {a(r1['d_f'])}",
        f"    δf₁    = {a(r1['delta_f'])}     (nol karena c₀ = 0: belum ada memori untuk dilupakan)",
        "",
        "(d) Input gate",
        f"    ∂L/∂i₁ = ∂L/∂c₁ × c̃₁ = {kr(r1['dc'])} × {a(s1['cc'])} = {a(r1['d_i'])}",
        f"    δi₁    = {kr(r1['d_i'])} × {a(s1['i'] * (1 - s1['i']))} = {a(r1['delta_i'])}",
        "",
        "(e) Kandidat",
        f"    ∂L/∂c̃₁ = ∂L/∂c₁ × i₁ = {kr(r1['dc'])} × {a(s1['i'])} = {a(r1['d_cc'])}",
        f"    δc̃₁    = {kr(r1['d_cc'])} × {a(1 - s1['cc'] ** 2)} = {a(r1['delta_c'])}",
    ])
    d.teks(f"""
Sekitar **{persen(porsi)}** sinyal kesalahan ke c₁ datang lewat jalur cell
state. Inilah wujud nyata alasan LSTM tahan terhadap *vanishing gradient*: di
jalur ini gradien hanya dikalikan f (bukan melewati tanh dan matriks bobot
berulang kali seperti pada RNN biasa).
""")

    # 3.8 ------------------------------------------------------------------
    d.teks("### 3.8 Langkah 3e: Gradien Total Setiap Bobot dan Bias")
    d.teks(r"""
Bobot yang sama dipakai di t = 1 **dan** t = 2, sehingga kontribusi kedua time
step dijumlahkan (aturan penjumlahan). Dari $a_t = W x_t + U h_{t-1} + b$
diperoleh $\partial a/\partial W = x_t$, $\partial a/\partial U = h_{t-1}$, dan
$\partial a/\partial b = 1$, sehingga:

$$\frac{\partial L}{\partial W} = \sum_t \delta_t\,x_t \qquad
\frac{\partial L}{\partial U} = \sum_t \delta_t\,h_{t-1} \qquad
\frac{\partial L}{\partial b} = \sum_t \delta_t$$
""")
    baris = []
    for gb, judul in (("f", "Forget gate"), ("i", "Input gate"), ("c", "Kandidat (c̃)"), ("o", "Output gate")):
        d1, d2 = r1["delta_" + gb], r2["delta_" + gb]
        sim = "δc̃" if gb == "c" else "δ" + gb
        baris.append(f"{judul}:  {sim}₁ = {a(d1)},  {sim}₂ = {a(d2)}")
        baris.append(f"  ∂L/∂W_{gb} = {sim}₁ × x₁ + {sim}₂ × x₂ = {kr(d1)} × {X[0]} + {kr(d2)} × {X[1]}"
                     f" = {kr(d1 * X[0])} + {kr(d2 * X[1])} = {a(g['W_' + gb])}")
        baris.append(f"  ∂L/∂U_{gb} = {sim}₁ × h₀ + {sim}₂ × h₁ = {kr(d1)} × {a(s1['h_prev'])} + {kr(d2)} × {a(s2['h_prev'])}"
                     f" = {a(g['U_' + gb])}")
        baris.append(f"  ∂L/∂b_{gb} = {sim}₁ + {sim}₂ = {kr(d1)} + {kr(d2)} = {a(g['b_' + gb])}")
        baris.append("")
    baris.append("Lapisan dense (dari 3.4):")
    baris.append(f"  ∂L/∂W_y = {a(g['W_y'])}")
    baris.append(f"  ∂L/∂b_y = {a(g['b_y'])}")
    d.kode(baris)
    assert all(g[n] < 0 for n in p), "Teks 3.8 menyatakan semua gradien LSTM negatif"
    d.teks("""
**Semua gradien bernilai negatif.** Prediksi terlalu rendah, dan setiap bobot
awal bernilai positif, sehingga menaikkan bobot mana pun akan menaikkan ŷ′.
Karena itu langkah Adam berikutnya akan menaikkan semua parameter. Perhatikan
juga **∂L/∂U = δ₂ × h₁ saja**: pada t = 1 bobot U tidak mendapat gradien karena
h₀ = 0, sedangkan bias tetap mendapat sinyal dari kedua time step.
""")

    # 3.9 ------------------------------------------------------------------
    d.teks("### 3.9 Pemeriksaan Gradien dengan Turunan Numerik")
    d.teks(r"""
Untuk memastikan seluruh rantai turunan di atas benar, setiap gradien
dibandingkan dengan turunan numerik (beda pusat):
$\dfrac{L(\theta+10^{-6}) - L(\theta-10^{-6})}{2\cdot10^{-6}}$.
""")
    d.tabel(["Parameter", "Gradien BPTT (manual)", "Gradien numerik", "Selisih"],
            [[nama(n), a(g[n], 8), a(num[n], 8), f"{abs(g[n] - num[n]):.1e}"] for n in p])

    bagian_adam(d, p, latih, contoh=("W_c", "b_y"), urutan_bagian=3)


def bagian_adam(d, p_awal, latih, contoh, urutan_bagian, ringkas=False):
    """Langkah 4 (Adam) beserta iterasi lanjutan. ``ringkas`` dipakai untuk GRU
    agar penjelasan yang sama dengan LSTM cukup dirujuk, tidak diulang."""
    rw1, rw2 = latih["riwayat"][0], latih["riwayat"][1]
    n_ = urutan_bagian
    d.teks(f"### {n_}.10 Langkah 4: Update Adam, Iterasi k = 1")
    if ringkas:
        d.teks("""
Rumus Adam sama dengan bagian 3.10 (persamaan 29-31). Setiap parameter GRU juga
punya m dan v sendiri yang diawali 0.
""")
    else:
        d.teks(r"""
Rumus Adam (persamaan 29-31) diterapkan **untuk setiap parameter secara
terpisah**; setiap parameter punya m dan v sendiri yang diawali 0:

$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\,g_k \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)\,g_k^2$$

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad
\hat{v}_k = \frac{v_k}{1-\beta_2^k} \qquad
\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}$$
""")
    baris = []
    for i, n in enumerate(contoh, 1):
        gk, mk, vk = rw1["g"][n], rw1["m"][n], rw1["v"][n]
        mh, vh, de = rw1["mh"][n], rw1["vh"][n], rw1["delta"][n]
        th = rw1["p"][n]
        if i > 1:
            baris.append("")
        baris += [
            f"Contoh {i} — {nama(n)}  (g₁ = {a(gk)}, nilai awal {nama(n)} = {bt(th)})",
            f"  m₁  = 0.9 × 0 + 0.1 × {kr(gk)} = {a(mk, 7)}",
            f"  v₁  = 0.999 × 0 + 0.001 × {kr(gk)}² = {a(vk, 10)}",
            f"  m̂₁  = {a(mk, 7)} / (1 - 0.9¹) = {a(mk, 7)} / 0.1 = {a(mh)}",
            f"  v̂₁  = {a(vk, 10)} / (1 - 0.999¹) = {a(vk, 10)} / 0.001 = {a(vh, 7)}",
            f"  √v̂₁ = √{a(vh, 7)} = {a(math.sqrt(vh))}",
            f"  Δ   = η × m̂₁ / (√v̂₁ + ε) = 0.001 × {kr(mh)} / ({a(math.sqrt(vh))} + 0.0000001) = {a(de, 7)}",
            f"  {nama(n)} baru = {nama(n)} lama - Δ = {bt(th)} - {kr(de, 7)} = {a(th - de)}",
        ]
    d.kode(baris)

    g1 = rw1["g"]
    besar = max(p_awal, key=lambda n: abs(g1[n]))
    kecil = min(p_awal, key=lambda n: abs(g1[n]))
    turun = [nama(n) for n in p_awal if g1[n] > 0]
    if turun:
        arah = (f"Parameter dengan gradien **positif** ({', '.join(turun)}) **turun** 0.001, "
                "sedangkan parameter lain (gradien negatif) naik 0.001.")
    else:
        arah = "Semua gradien negatif, sehingga **semua parameter naik 0.001**."
    d.teks(f"""
**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias
membuat m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap
parameter bergeser sebesar 0.001** (= η) berlawanan arah dengan tanda
gradiennya, berapa pun besar gradiennya. Selisih kecil pada digit ke-7 (misalnya
pada {nama(kecil)}) berasal dari ε yang ditambahkan ke penyebut. {arah}

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding,
SGD biasa (Δ = η × g) akan menggeser {nama(besar)} sebesar
{a(LR * abs(g1[besar]), 7)} tetapi {nama(kecil)} hanya {a(LR * abs(g1[kecil]), 7)},
yaitu {ribu(round(abs(g1[besar]) / abs(g1[kecil])))} kali lebih kecil. Adam menyamakan
kecepatan belajar semua parameter.
""")
    d.tabel(["Parameter", "Nilai lama", "Gradien g₁", "m₁", "v₁", "Δ = η·m̂₁/(√v̂₁+ε)", "Nilai baru"],
            [[nama(n), a(rw1["p"][n]), a(rw1["g"][n]), a(rw1["m"][n], 7), a(rw1["v"][n], 10),
              a(rw1["delta"][n], 7), a(rw1["p"][n] - rw1["delta"][n])] for n in p_awal])

    # forward ulang -------------------------------------------------------
    d.teks(f"### {n_}.11 Forward Ulang dengan Bobot Baru")
    d.kode([
        "Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:",
        f"  ŷ' = {a(latih['yhat'][1])}     (sebelumnya {a(latih['yhat'][0])})",
        f"  L  = ({Y} - {a(latih['yhat'][1])})² = {a(latih['L'][1])}     (sebelumnya {a(latih['L'][0])})",
        f"  Loss turun {a(latih['L'][0] - latih['L'][1])} hanya dengan satu kali update.",
    ])
    assert latih["L"][1] < latih["L"][0]

    # iterasi 2 ------------------------------------------------------------
    d.teks(f"### {n_}.12 Iterasi k = 2: Momentum Mulai Bekerja")
    d.teks("""
Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT
menghasilkan gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol,
sehingga langkah Adam merupakan **campuran** gradien sekarang dan gradien
sebelumnya. Berikut perhitungan Adam untuk dua parameter yang sama:
""")
    baris = []
    for i, n in enumerate(contoh, 1):
        g2, m1, v1 = rw2["g"][n], rw2["m_lama"][n], rw2["v_lama"][n]
        m2, v2, mh, vh, de = rw2["m"][n], rw2["v"][n], rw2["mh"][n], rw2["vh"][n], rw2["delta"][n]
        th = rw2["p"][n]
        if i > 1:
            baris.append("")
        baris += [
            f"Contoh {i} — {nama(n)}  (g₂ = {a(g2)}, nilai sekarang = {a(th)}, m₁ = {a(m1, 7)}, v₁ = {a(v1, 10)})",
            f"  m₂  = 0.9 × m₁ + 0.1 × g₂ = 0.9 × {kr(m1, 7)} + 0.1 × {kr(g2)}"
            f" = {kr(B1 * m1, 7)} + {kr((1 - B1) * g2, 7)} = {a(m2, 7)}",
            f"  v₂  = 0.999 × v₁ + 0.001 × g₂² = 0.999 × {a(v1, 10)} + 0.001 × {kr(g2)}²"
            f" = {a(B2 * v1, 10)} + {a((1 - B2) * g2 ** 2, 10)} = {a(v2, 10)}",
            f"  m̂₂  = {a(m2, 7)} / (1 - 0.9²) = {a(m2, 7)} / 0.19 = {a(mh)}",
            f"  v̂₂  = {a(v2, 10)} / (1 - 0.999²) = {a(v2, 10)} / 0.001999 = {a(vh, 7)}",
            f"  √v̂₂ = {a(math.sqrt(vh))}",
            f"  Δ   = 0.001 × {kr(mh)} / ({a(math.sqrt(vh))} + 0.0000001) = {a(de, 7)}",
            f"  {nama(n)} baru = {a(th)} - {kr(de, 7)} = {a(th - de)}",
        ]
    d.kode(baris)
    if ringkas:
        d.teks("Tabel koreksi bias (1 − βᵏ) pada bagian 3.12 berlaku sama untuk GRU.")
    else:
        d.teks("""
m₂ menggabungkan arah gradien iterasi 1 dan 2. Jika suatu saat gradien
berbalik arah (misalnya ketika bobot melewati titik minimum), m akan mengecil
dan langkahnya otomatis melambat. Koreksi bias (1 − βᵏ) makin lama makin
mendekati 1, sehingga pengaruhnya hilang setelah banyak iterasi:
""")
        d.tabel(["Iterasi k", "Keterangan", "1 - 0.9ᵏ (pembagi m)", "1 - 0.999ᵏ (pembagi v)"],
                [[ribu(kk), ket, a(1 - B1 ** kk, 6), a(1 - B2 ** kk, 6)]
                 for kk, ket in ((1, "iterasi pertama"), (2, "iterasi kedua"),
                                 (26, "akhir epoch 1 penelitian"), (1000, "± epoch 38"),
                                 (2600, "akhir epoch 100"), (13000, "akhir epoch 500"))])

    # perjalanan loss ------------------------------------------------------
    d.teks(f"### {n_}.13 Siklus Diulang: Perjalanan Loss")
    kolom_bias = [n for n in p_awal if n.startswith("b_")][:1]
    d.teks(f"""
Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah
update. Bobot berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom
b_y dan {nama(kolom_bias[0])} memperlihatkan bias ikut dipelajari seperti bobot.
""")
    d.tabel(["Setelah k update", "Loss L", "Prediksi ŷ′", "b_y"] + [nama(n) for n in kolom_bias],
            [[ribu(j), fl(latih["L"][j]), a(latih["yhat"][j]), a(latih["p"][j]["b_y"])]
             + [a(latih["p"][j][n]) for n in kolom_bias]
             for j in TITIK_RIWAYAT])
    akhir = latih["p"][-1]
    d.teks(f"Nilai seluruh parameter sebelum dan sesudah {ITERASI} iterasi:")
    d.tabel(["Parameter", "Awal", f"Setelah {ITERASI} iterasi", "Perubahan"],
            [[nama(n), bt(p_awal[n]), a(akhir[n]), f"{akhir[n] - p_awal[n]:+.6f}"] for n in p_awal])


# ------------------------------- GRU --------------------------------------- #
def bagian_4(d, k, p, maju, mundur, num, latih):
    s1, s2 = maju["langkah"]
    r1, r2 = mundur["rincian"]
    g = mundur["g"]
    d.teks(f"## {JUDUL[4]}")
    d.teks("""
Urutan langkahnya sama persis dengan LSTM. Yang berbeda hanya rumus di dalam
sel: GRU tidak punya cell state, punya tiga himpunan bobot (z, r, kandidat), dan
setiap gerbang punya **dua bias** (bias masukan dan bias rekuren) karena Keras
memakai `reset_after=True` (lihat bagian 5.4).
""")

    # 4.1 ------------------------------------------------------------------
    d.teks("### 4.1 Bobot Awal dan Jumlah Parameter")
    d.tabel(["Gerbang", "W (bobot masukan)", "U (bobot rekuren)", "b(in) (bias masukan)", "b(rec) (bias rekuren)"],
            [["update (z)", bt(p["W_z"]), bt(p["U_z"]), bt(p["b_z_in"]), bt(p["b_z_rec"])],
             ["reset (r)", bt(p["W_r"]), bt(p["U_r"]), bt(p["b_r_in"]), bt(p["b_r_rec"])],
             ["kandidat (h̃)", bt(p["W_h"]), bt(p["U_h"]), bt(p["b_h_in"]), bt(p["b_h_rec"])],
             ["dense", f"W_y = {bt(p['W_y'])}", "-", f"b_y = {bt(p['b_y'])}", "-"]])
    d.kode([
        "Jumlah parameter (persamaan 27) dengan n_u = 1 dan n_f = 1:",
        "  P_GRU = 3 × [n_u × (n_u + n_f) + 2 × n_u] + (n_u + 1)",
        "        = 3 × [1 × (1 + 1) + 2] + (1 + 1)",
        "        = 3 × 4 + 2",
        f"        = {len(p)} parameter  (12 di lapisan GRU + 2 di lapisan dense)",
        "  Suku 2 × n_u = dua bias per gerbang (bias masukan dan bias rekuren).",
    ])
    assert len(p) == 3 * (1 * 2 + 2) + 2

    # 4.2 ------------------------------------------------------------------
    d.teks("### 4.2 Langkah 1: Forward Pass")
    d.teks(r"""
Rumus versi 1 neuron (persamaan 23-26). Untuk kandidat, bagian rekurennya
diberi nama $q_t$ supaya terlihat jelas apa yang dikalikan reset gate:

$$z_t = \sigma\bigl(W_z x_t + U_z h_{t-1} + b_z^{(in)} + b_z^{(rec)}\bigr) \qquad
r_t = \sigma\bigl(W_r x_t + U_r h_{t-1} + b_r^{(in)} + b_r^{(rec)}\bigr)$$

$$q_t = U_h h_{t-1} + b_h^{(rec)} \qquad
\tilde{h}_t = \tanh\bigl(W_h x_t + b_h^{(in)} + r_t\,q_t\bigr) \qquad
h_t = z_t\,h_{t-1} + (1-z_t)\,\tilde{h}_t$$
""")
    d.teks("""
Catatan konvensi: pada Keras (dan pada persamaan 7 makalah Cho et al., 2014),
**zₜ mengalikan memori lama hₜ₋₁**. Bentuk dengan peran z tertukar,
hₜ = (1 − zₜ)hₜ₋₁ + zₜh̃ₜ, berasal dari Chung et al. (2014).
""")
    baris = []
    for t, s in ((1, s1), (2, s2)):
        sub = "₁₂"[t - 1]
        xn, hn = f"x{sub}", f"h{'₀₁'[t - 1]}"
        baris.append("=" * 70)
        baris.append(f"TIME STEP t = {t}   ({xn} = {a(s['x'])}, {hn} = {a(s['h_prev'])})")
        baris.append("=" * 70)
        for gb, judul in (("z", "Update gate (persamaan 23, Gambar 7: kotak σ kiri)"),
                          ("r", "Reset gate (persamaan 24, Gambar 7: kotak σ tengah)")):
            blok_gerbang(baris, judul, gb + sub, (f"W_{gb}", f"U_{gb}"), s["x"], s["h_prev"],
                         {"W": p["W_" + gb], "U": p["U_" + gb], "a": s["a_" + gb], "nilai": s[gb],
                          "bias_nama": [(f"b_{gb}(in)", 0), (f"b_{gb}(rec)", 0)]},
                         "sigmoid", xn, hn, [p[f"b_{gb}_in"], p[f"b_{gb}_rec"]])
        baris.append("Kandidat hidden state (persamaan 25, Gambar 7: kotak tanh)")
        baris.append(f"  bagian rekuren  q{sub} = U_h × {hn} + b_h(rec) = {bt(p['U_h'])} × {a(s['h_prev'])}"
                     f" + {bt(p['b_h_rec'])} = {a(s['q'])}")
        baris.append(f"  bagian masukan       = W_h × {xn} + b_h(in) = {bt(p['W_h'])} × {a(s['x'])}"
                     f" + {bt(p['b_h_in'])} = {a(s['masuk'])}")
        baris.append(f"  a = bagian masukan + r{sub} × q{sub} = {a(s['masuk'])} + {a(s['r'])} × {a(s['q'])}"
                     f" = {a(s['masuk'])} + {a(s['rq'])} = {a(s['a_h'])}")
        baris.append(f"  h̃{sub} = tanh({a(s['a_h'])}) = {a(s['hh'])}")
        baris.append("Hidden state baru (persamaan 26, Gambar 7: lingkaran × , 1−, dan +)")
        baris.append(f"  h{sub} = z{sub} × {hn} + (1 - z{sub}) × h̃{sub}")
        baris.append(f"     = {a(s['z'])} × {a(s['h_prev'])} + {a(1 - s['z'])} × {a(s['hh'])}")
        baris.append(f"     = {a(s['z'] * s['h_prev'])} + {a((1 - s['z']) * s['hh'])}")
        baris.append(f"     = {a(s['h'])}")
        if t == 1:
            baris.append("  (q₁ = 0 karena h₀ = 0, jadi reset gate belum berpengaruh pada t = 1)")
        baris.append("")
    baris.append("=" * 70)
    baris.append("LAPISAN DENSE (persamaan 21)")
    baris.append("=" * 70)
    baris.append(f"  ŷ' = W_y × h₂ + b_y = {bt(p['W_y'])} × {a(s2['h'])} + {bt(p['b_y'])} = {a(maju['yhat'])}")
    d.kode(baris)
    d.tabel(["t", "x", "z", "r", "q", "h̃", "h"],
            [[t, a(s["x"]), a(s["z"]), a(s["r"]), a(s["q"]), a(s["hh"]), a(s["h"])]
             for t, s in ((1, s1), (2, s2))])

    # 4.3 ------------------------------------------------------------------
    d.teks("### 4.3 Langkah 2: Loss")
    d.kode([
        f"  L = (y' - ŷ')² = ({Y} - {a(maju['yhat'])})² = {a(Y - maju['yhat'])}² = {a(maju['L'])}",
    ])

    # 4.4 ------------------------------------------------------------------
    dy = mundur["dy"]
    d.teks("### 4.4 Langkah 3a: Backward di Lapisan Dense")
    d.teks("Rumusnya identik dengan 3.4; hanya nilai h₂ dan ŷ′ yang berbeda.")
    d.kode([
        f"  ∂L/∂ŷ'  = -2(y' - ŷ') = -2 × ({Y} - {a(maju['yhat'])}) = {a(dy)}",
        f"  ∂L/∂W_y = ∂L/∂ŷ' × h₂ = {kr(dy)} × {a(s2['h'])} = {a(g['W_y'])}",
        f"  ∂L/∂b_y = ∂L/∂ŷ' × 1  = {a(g['b_y'])}",
        f"  ∂L/∂h₂  = ∂L/∂ŷ' × W_y = {kr(dy)} × {bt(p['W_y'])} = {a(mundur['dh_T'])}   (masuk ke sel GRU)",
    ])

    # 4.5 ------------------------------------------------------------------
    d.teks("### 4.5 Langkah 3b: Backward di Dalam Sel, t = 2")
    d.teks("""
Rumus forward yang diturunkan: `h₂ = z₂ × h₁ + (1 − z₂) × h̃₂` (menuju update
gate dan kandidat), lalu `h̃₂ = tanh(W_h x₂ + b_h(in) + r₂ × q₂)` (menuju reset
gate dan bagian rekuren q₂).
""")
    d.kode([
        f"Masukan dari 4.4: ∂L/∂h₂ = {a(r2['dh'])}",
        "",
        "(a) Update gate — turunkan h₂ terhadap z₂: ∂h₂/∂z₂ = h₁ - h̃₂",
        f"    ∂L/∂z₂ = ∂L/∂h₂ × (h₁ - h̃₂) = {kr(r2['dh'])} × ({a(s2['h_prev'])} - {a(s2['hh'])})"
        f" = {kr(r2['dh'])} × {kr(s2['h_prev'] - s2['hh'])} = {a(r2['d_z'])}",
        f"    δz₂    = ∂L/∂z₂ × z₂(1 - z₂) = {kr(r2['d_z'])} × {a(s2['z'] * (1 - s2['z']))} = {a(r2['delta_z'])}",
        "",
        "(b) Kandidat — turunkan h₂ terhadap h̃₂: ∂h₂/∂h̃₂ = 1 - z₂",
        f"    ∂L/∂h̃₂ = ∂L/∂h₂ × (1 - z₂) = {kr(r2['dh'])} × {a(1 - s2['z'])} = {a(r2['d_hh'])}",
        f"    δh̃₂    = ∂L/∂h̃₂ × (1 - h̃₂²) = {kr(r2['d_hh'])} × {a(1 - s2['hh'] ** 2)} = {a(r2['delta_h'])}",
        "",
        "(c) Reset gate — dalam pra-aktivasi kandidat, r₂ dikalikan q₂",
        f"    ∂L/∂r₂ = δh̃₂ × q₂ = {kr(r2['delta_h'])} × {a(s2['q'])} = {a(r2['d_r'])}",
        f"    δr₂    = ∂L/∂r₂ × r₂(1 - r₂) = {kr(r2['d_r'])} × {a(s2['r'] * (1 - s2['r']))} = {a(r2['delta_r'])}",
        "",
        "(d) Bagian rekuren kandidat q₂ = U_h × h₁ + b_h(rec)",
        f"    ∂L/∂q₂ = δh̃₂ × r₂ = {kr(r2['delta_h'])} × {a(s2['r'])} = {a(r2['d_q'])}",
    ])
    assert r2["d_z"] > 0 and g["W_z"] > 0, "Teks 4.5 menyatakan gradien z positif"
    d.teks(f"""
**Makna tanda gradien z.** ∂L/∂z₂ bernilai **positif** karena kandidat baru
(h̃₂ = {a(s2['hh'])}) lebih besar daripada memori lama (h₁ = {a(s2['h_prev'])}),
sedangkan prediksi masih terlalu rendah. Adam nanti akan **menurunkan** z,
artinya GRU belajar untuk lebih mempercayai informasi baru.
""")

    # 4.6 ------------------------------------------------------------------
    su = r2["suku_h"]
    porsi = su[0] / r2["dh_prev"]
    d.teks("### 4.6 Langkah 3c: Mengirim Kesalahan ke t = 1")
    d.teks("""
h₁ dipakai di empat tempat pada t = 2: langsung di `z₂ × h₁`, di update gate
(lewat U_z), di reset gate (lewat U_r), dan di bagian rekuren q₂ (lewat U_h).
Keempat kontribusinya dijumlahkan.
""")
    d.kode([
        "  ∂L/∂h₁ = ∂L/∂h₂ × z₂ + U_z × δz₂ + U_r × δr₂ + U_h × ∂L/∂q₂",
        f"         = {kr(r2['dh'])} × {a(s2['z'])} + {bt(p['U_z'])} × {kr(r2['delta_z'])}"
        f" + {bt(p['U_r'])} × {kr(r2['delta_r'])} + {bt(p['U_h'])} × {kr(r2['d_q'])}",
        f"         = {kr(su[0])} + {kr(su[1])} + {kr(su[2])} + {kr(su[3])}",
        f"         = {a(r2['dh_prev'])}",
        f"  Porsi jalur langsung z₂ × h₁ = {a(su[0])} / {a(r2['dh_prev'])} = {persen(porsi)}",
    ])
    d.teks(f"""
Sekitar **{persen(porsi)}** sinyal mengalir lewat jalur langsung `zₜ × hₜ₋₁`.
Jalur ini adalah "jalan tol gradien" pada GRU, padanan cell state pada LSTM.
""")

    # 4.7 ------------------------------------------------------------------
    d.teks("### 4.7 Langkah 3d: Backward di Dalam Sel, t = 1")
    d.kode([
        f"Masukan dari 4.6: ∂L/∂h₁ = {a(r1['dh'])}",
        "",
        "(a) Update gate",
        f"    ∂L/∂z₁ = ∂L/∂h₁ × (h₀ - h̃₁) = {kr(r1['dh'])} × ({a(s1['h_prev'])} - {a(s1['hh'])}) = {a(r1['d_z'])}",
        f"    δz₁    = {kr(r1['d_z'])} × {a(s1['z'] * (1 - s1['z']))} = {a(r1['delta_z'])}",
        "",
        "(b) Kandidat",
        f"    ∂L/∂h̃₁ = ∂L/∂h₁ × (1 - z₁) = {kr(r1['dh'])} × {a(1 - s1['z'])} = {a(r1['d_hh'])}",
        f"    δh̃₁    = {kr(r1['d_hh'])} × {a(1 - s1['hh'] ** 2)} = {a(r1['delta_h'])}",
        "",
        "(c) Reset gate",
        f"    ∂L/∂r₁ = δh̃₁ × q₁ = {kr(r1['delta_h'])} × {a(s1['q'])} = {a(r1['d_r'])}",
        f"    δr₁    = {a(r1['delta_r'])}     (nol karena q₁ = 0: belum ada memori untuk di-reset)",
        "",
        "(d) Bagian rekuren q₁",
        f"    ∂L/∂q₁ = δh̃₁ × r₁ = {kr(r1['delta_h'])} × {a(s1['r'])} = {a(r1['d_q'])}",
    ])

    # 4.8 ------------------------------------------------------------------
    d.teks("### 4.8 Langkah 3e: Gradien Total Setiap Bobot dan Bias")
    d.teks(r"""
Sama seperti LSTM, kontribusi kedua time step dijumlahkan. Untuk kandidat,
bobot $U_h$ dan bias $b_h^{(rec)}$ berada di dalam $q_t$, sehingga gradiennya
memakai $\partial L/\partial q_t$; bobot $W_h$ dan bias $b_h^{(in)}$ memakai
$\delta\tilde{h}_t$.
""")
    baris = []
    for gb, judul in (("z", "Update gate"), ("r", "Reset gate")):
        d1, d2 = r1["delta_" + gb], r2["delta_" + gb]
        baris.append(f"{judul}:  δ{gb}₁ = {a(d1)},  δ{gb}₂ = {a(d2)}")
        baris.append(f"  ∂L/∂W_{gb} = δ{gb}₁ × x₁ + δ{gb}₂ × x₂ = {kr(d1)} × {X[0]} + {kr(d2)} × {X[1]}"
                     f" = {kr(d1 * X[0])} + {kr(d2 * X[1])} = {a(g['W_' + gb])}")
        baris.append(f"  ∂L/∂U_{gb} = δ{gb}₁ × h₀ + δ{gb}₂ × h₁ = {kr(d1)} × {a(s1['h_prev'])} + {kr(d2)} × {a(s2['h_prev'])}"
                     f" = {a(g['U_' + gb])}")
        baris.append(f"  ∂L/∂b_{gb}(in)  = δ{gb}₁ + δ{gb}₂ = {kr(d1)} + {kr(d2)} = {a(g[f'b_{gb}_in'])}")
        baris.append(f"  ∂L/∂b_{gb}(rec) = δ{gb}₁ + δ{gb}₂ = {a(g[f'b_{gb}_rec'])}   (sama persis dengan bias masukan)")
        baris.append("")
    d1, d2 = r1["delta_h"], r2["delta_h"]
    q1, q2 = r1["d_q"], r2["d_q"]
    baris += [
        f"Kandidat:  δh̃₁ = {a(d1)},  δh̃₂ = {a(d2)},  ∂L/∂q₁ = {a(q1)},  ∂L/∂q₂ = {a(q2)}",
        f"  ∂L/∂W_h      = δh̃₁ × x₁ + δh̃₂ × x₂ = {kr(d1)} × {X[0]} + {kr(d2)} × {X[1]}"
        f" = {kr(d1 * X[0])} + {kr(d2 * X[1])} = {a(g['W_h'])}",
        f"  ∂L/∂b_h(in)  = δh̃₁ + δh̃₂ = {kr(d1)} + {kr(d2)} = {a(g['b_h_in'])}",
        f"  ∂L/∂U_h      = ∂L/∂q₁ × h₀ + ∂L/∂q₂ × h₁ = {kr(q1)} × {a(s1['h_prev'])} + {kr(q2)} × {a(s2['h_prev'])}"
        f" = {a(g['U_h'])}",
        f"  ∂L/∂b_h(rec) = ∂L/∂q₁ + ∂L/∂q₂ = {kr(q1)} + {kr(q2)} = {a(g['b_h_rec'])}",
        "  (∂L/∂q = δh̃ × r, jadi bias rekuren kandidat = Σ δh̃ × r, BERBEDA dari bias masukan)",
        "",
        "Lapisan dense (dari 4.4):",
        f"  ∂L/∂W_y = {a(g['W_y'])}",
        f"  ∂L/∂b_y = {a(g['b_y'])}",
    ]
    d.kode(baris)

    # 4.9 ------------------------------------------------------------------
    d.teks("### 4.9 Pemeriksaan Gradien dengan Turunan Numerik")
    d.tabel(["Parameter", "Gradien BPTT (manual)", "Gradien numerik", "Selisih"],
            [[nama(n), a(g[n], 8), a(num[n], 8), f"{abs(g[n] - num[n]):.1e}"] for n in p])

    bagian_adam(d, p, latih, contoh=("W_h", "b_y"), urutan_bagian=4, ringkas=True)
    akhir = latih["p"][-1]
    d.teks(f"""
Perhatikan pasangan bias pada tabel terakhir: **b_z(in) = b_z(rec) = {a(akhir['b_z_in'])}**
dan **b_r(in) = b_r(rec) = {a(akhir['b_r_in'])}** tetap kembar sampai akhir,
sedangkan **b_h(in) = {a(akhir['b_h_in'])}** dan **b_h(rec) = {a(akhir['b_h_rec'])}**
berbeda. Bagian 5.4 menjelaskan sebabnya.
""")


# ------------------------------- Bias -------------------------------------- #
def bagian_5(d, k, lstm, gru):
    gl, gg = lstm["mundur"]["g"], gru["mundur"]["g"]
    rl = lstm["mundur"]["rincian"]
    d.teks(f"## {JUDUL[5]}")

    d.teks("### 5.1 Intinya")
    d.teks("""
Bias adalah **titik awal (intercept)** setiap gerbang dan neuron, sama seperti
intercept *a* pada regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan.
Tanpa bias, ketika masukannya nol, setiap gerbang dipaksa bernilai tetap
(σ(0) = 0.5, selalu setengah terbuka; tanh(0) = 0). Dengan bias, setiap gerbang
dapat menentukan **posisi bawaannya sendiri**. Bias tidak dihitung dengan satu
rumus langsung; bias **dipelajari** lewat siklus yang sama dengan bobot
(bagian 5.3).
""")

    # 5.2 ------------------------------------------------------------------
    d.teks("### 5.2 Mengapa Bias Diperlukan")
    d.teks("""
**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Ini
sering terjadi di penelitian: h₀ = 0 di awal setiap jendela 7 hari, dan
normalisasi min-max membuat fitur bernilai dekat 0 ketika nilainya mendekati
minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W
mengatur kecuraman kurva, bias b mengatur posisinya (gerbang = 0.5 saat
x = −b/W). Contoh satu fitur ternormalisasi x ∈ [0, 1] dengan W = 5:
""")
    d.tabel(["x", "Dengan bias: σ(5x − 2.5)", "Tanpa bias: σ(5x)"],
            [[f"{x:g}", a(sig(5 * x - 2.5), 3), a(sig(5 * x), 3)] for x in (0, 0.25, 0.5, 0.75, 1)])
    d.teks("""
Tanpa bias, gerbang **tidak pernah turun di bawah 0.5** untuk data
ternormalisasi yang positif, sehingga model tidak bisa menyatakan aturan "tutup
gerbang saat nilai fitur rendah, buka saat tinggi".

**(c) Bias menentukan perilaku bawaan setiap gerbang.** Contoh terpenting:
forget gate LSTM. Jika gerbang hanya ditentukan oleh biasnya, sisa memori
setelah 7 hari adalah f⁷:
""")
    bf = k["bias_lstm"]["f"]
    d.tabel(["Bias forget", "f = σ(b)", "Memori tersisa setelah 7 hari (f⁷)"],
            [[lbl, a(sig(b), 3), persen(sig(b) ** 7)]
             for lbl, b in (("0 (tanpa bias)", 0.0), (f"{a(bf)} (neuron ke-1 LSTM terlatih)", bf),
                            ("1 (inisialisasi Keras)", 1.0), ("2", 2.0))])
    d.teks("""
Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras
mengisi **b_f = 1** di awal pelatihan (*unit forget bias*).

Nilai bawaan gerbang pada **model terlatih penelitian** (neuron ke-1, saat
kontribusi xW + hU = 0), dibaca dari `tahap_7_forward_pass_lstm.md` dan
`tahap_8_forward_pass_gru.md`:
""")
    bl, bg = k["bias_lstm"], k["bias_gru"]
    d.tabel(["Model", "Gerbang", "Bias", "Nilai bawaan", "Arti"],
            [["LSTM", "input i", a(bl["i"]), f"σ = {a(sig(bl['i']), 3)}", "cenderung hemat menerima informasi baru"
              if sig(bl["i"]) < 0.5 else "cenderung menerima informasi baru"],
             ["LSTM", "forget f", a(bl["f"]), f"σ = {a(sig(bl['f']), 3)}", "cenderung mempertahankan memori"
              if sig(bl["f"]) > 0.5 else "cenderung membuang memori"],
             ["LSTM", "kandidat c̃", a(bl["c"]), f"tanh = {a(math.tanh(bl['c']), 3)}", "isi bawaan hampir netral"
              if abs(bl["c"]) < 0.1 else "isi bawaan tidak netral"],
             ["LSTM", "output o", a(bl["o"]), f"σ = {a(sig(bl['o']), 3)}", "cenderung menahan sebagian keluaran"
              if sig(bl["o"]) < 0.5 else "cenderung mengeluarkan memori"],
             ["GRU", "update z", f"{a(bg['z'][0])} + {kr(bg['z'][1])}", f"σ = {a(sig(sum(bg['z'])), 3)}",
              f"cenderung mempertahankan {persen(sig(sum(bg['z'])), 0)} memori lama"],
             ["GRU", "reset r", f"{a(bg['r'][0])} + {kr(bg['r'][1])}", f"σ = {a(sig(sum(bg['r'])), 3)}",
              f"memakai sekitar {persen(sig(sum(bg['r'])), 0)} masa lalu untuk kandidat"]])
    d.teks(f"""
**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di
rentang (−1, 1), jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y
cukup menangani variasinya. Pada model terlatih, b_y LSTM = {a(k['by_lstm'])}
(setara {a(k['by_lstm'])} × {usd(k['rentang'])} ≈ **{usd(k['by_lstm'] * k['rentang'])} USD**)
dan b_y GRU = {a(k['by_gru'])} (≈ **{usd(k['by_gru'] * k['rentang'])} USD**).
Tanpa b_y, prediksi dipaksa jatuh ke harga terendah data latih
({usd(k['x_min'])} USD) setiap kali h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁
bernilai nol bila masukannya nol (lihat ∂L/∂U di 3.8: pada t = 1 tidak ada
kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.

**(f) Biayanya kecil.** Pada model penelitian, bias hanya
{4 * k['lstm']['neuron'] + 1} dari {ribu(k['lstm']['param'])} parameter LSTM
({persen((4 * k['lstm']['neuron'] + 1) / k['lstm']['param'])}) dan
{6 * k['gru']['neuron'] + 1} dari {ribu(k['gru']['param'])} parameter GRU
({persen((6 * k['gru']['neuron'] + 1) / k['gru']['param'])}).
""")

    # 5.3 ------------------------------------------------------------------
    d.teks("### 5.3 Cara Menghitung Bias Langkah demi Langkah")
    d.teks("""
Bias diperlakukan **persis seperti bobot**. Berikut lima langkahnya, dengan
rujukan ke bagian yang sudah dihitung di atas:
""")
    lat = lstm["latih"]
    d.kode([
        "Langkah 1 — Nilai awal (3.1 dan 4.1)",
        "  Bias gerbang = 0, kecuali bias forget gate LSTM = 1. Bias dense b_y = 0.",
        "",
        "Langkah 2 — Dipakai di forward pass (3.2 dan 4.2)",
        "  Gerbang : a = W × x + U × h + b",
        "  Dense   : ŷ' = W_y × h_T + b_y",
        "",
        "Langkah 3 — Hitung gradiennya (3.8 dan 4.8)",
        "  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga",
        "  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ",
        f"  Contoh b_y   : ∂L/∂b_y = ∂L/∂ŷ' × 1 = {a(gl['b_y'])}",
        f"  Contoh b_c   : ∂L/∂b_c = δc̃₁ + δc̃₂ = {kr(rl[0]['delta_c'])} + {kr(rl[1]['delta_c'])} = {a(gl['b_c'])}",
        "",
        "Langkah 4 — Update dengan Adam (3.10 dan 3.12)",
        f"  b_y: 0 → {a(lat['p'][1]['b_y'])} (k = 1) → {a(lat['p'][2]['b_y'])} (k = 2)",
        "",
        "Langkah 5 — Ulangi siklus",
        f"  Setelah {ITERASI} iterasi b_y = {a(lat['p'][-1]['b_y'])}. Pada model penelitian, b_y LSTM = {a(k['by_lstm'])}",
        f"  adalah hasil akhir proses yang sama setelah {ribu(k['iter_epoch'] * k['lstm']['epoch'])} iterasi.",
    ])
    d.teks("Ringkasan rumus gradien seluruh bias pada simulasi ini (iterasi k = 1):")
    d.tabel(["Bias", "Rumus gradien", "Nilai pada simulasi"],
            [["Dense b_y (LSTM)", "∂L/∂ŷ′", a(gl["b_y"])],
             ["LSTM b_f", "δf₁ + δf₂", a(gl["b_f"])],
             ["LSTM b_i", "δi₁ + δi₂", a(gl["b_i"])],
             ["LSTM b_c", "δc̃₁ + δc̃₂", a(gl["b_c"])],
             ["LSTM b_o", "δo₁ + δo₂", a(gl["b_o"])],
             ["Dense b_y (GRU)", "∂L/∂ŷ′", a(gg["b_y"])],
             ["GRU b_z(in) dan b_z(rec)", "keduanya δz₁ + δz₂", f"{a(gg['b_z_in'])} dan {a(gg['b_z_rec'])}"],
             ["GRU b_r(in) dan b_r(rec)", "keduanya δr₁ + δr₂", f"{a(gg['b_r_in'])} dan {a(gg['b_r_rec'])}"],
             ["GRU b_h(in)", "δh̃₁ + δh̃₂", a(gg["b_h_in"])],
             ["GRU b_h(rec)", "δh̃₁·r₁ + δh̃₂·r₂", a(gg["b_h_rec"])]])

    # 5.4 ------------------------------------------------------------------
    d.teks("### 5.4 Dua Bias pada GRU: Asal dan Buktinya")
    d.teks("""
**Letaknya pada Gambar 7.** Bias tidak digambar; bias berada di dalam setiap
kotak kuning (σ, σ, tanh). Setiap kotak menerima dua garis masuk, yaitu xₜ dari
bawah dan hₜ₋₁ dari garis vertikal kiri. Dengan `reset_after=True`, Keras
menghitung kedua garis itu terpisah, masing-masing dengan biasnya sendiri:
garis xₜ membawa **bias masukan** (xₜW + b(in)) dan garis hₜ₋₁ membawa **bias
rekuren** (hₜ₋₁U + b(rec)).
""")
    d.kode("""
Kotak σ untuk z (dan r, sama persis): kedua cabang langsung dijumlahkan
  h_(t-1) ──► h_(t-1) × U_z + b_z(rec) ──┐
                                         (+) ──► σ ──► z_t
  x_t ─────► x_t × W_z + b_z(in) ────────┘

Kotak tanh untuk kandidat: cabang h_(t-1) melewati lingkaran × milik r_t dulu
  h_(t-1) ──► h_(t-1) × U_h + b_h(rec) ──► (× r_t) ──┐
                                                     (+) ──► tanh ──► h̃_t
  x_t ─────► x_t × W_h + b_h(in) ─────────────────────┘
""")
    bg = k["bias_gru"]
    akhir = gru["latih"]["p"][-1]
    d.teks(f"""
**Pada z dan r, dua bias sebenarnya berlebih.** b(in) + b(rec) langsung
dijumlahkan, jadi gradien keduanya selalu sama. Karena nilai awalnya juga sama
(0), keduanya akan selalu kembar:

- simulasi: setelah {ITERASI} iterasi, b_z(in) = b_z(rec) = {a(akhir['b_z_in'])} dan
  b_r(in) = b_r(rec) = {a(akhir['b_r_in'])};
- model GRU terlatih penelitian (neuron ke-1): b_z(in) = {a(bg['z'][0])},
  b_z(rec) = {a(bg['z'][1])}; b_r(in) = {a(bg['r'][0])}, b_r(rec) = {a(bg['r'][1])}.

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikalikan rₜ (jika
rₜ mendekati 0, bias ini ikut "dimatikan" bersama memori lama), sedangkan
b_h(in) selalu aktif. Gradiennya berbeda (Σ δh̃·r vs Σ δh̃), sehingga nilainya
juga berbeda:

- simulasi: gradien {a(gru['mundur']['g']['b_h_in'])} vs {a(gru['mundur']['g']['b_h_rec'])};
  setelah {ITERASI} iterasi b_h(in) = {a(akhir['b_h_in'])} dan b_h(rec) = {a(akhir['b_h_rec'])};
- model terlatih (neuron ke-1): b_h(in) = {a(bg['h'][0])} dan b_h(rec) = {a(bg['h'][1])}.

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak memuat bias sama sekali
(penulisnya menyebut *"to make the equations uncluttered, we omit biases"*).
Bentuk dua bias berasal dari implementasi GRU pada pustaka **NVIDIA cuDNN**, yang
memisahkan bagian masukan dan bagian rekuren setiap gerbang. Keras memakainya
sebagai bawaan (`reset_after=True`, *"cuDNN compatible"*). LSTM di Keras cukup
memakai satu bias karena pada LSTM semua bias hanya dijumlahkan sehingga selalu
bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU {k['gru']['neuron']} neuron
memiliki 2 × 3 × {k['gru']['neuron']} = {6 * k['gru']['neuron']} bias gerbang
(Tabel 12: bias berbentuk (2, {3 * k['gru']['neuron']})). Dengan satu bias
jumlahnya hanya {3 * k['gru']['neuron']}, sehingga total parameter menjadi
{ribu(k['gru']['param'] - 3 * k['gru']['neuron'])}, bukan {ribu(k['gru']['param'])}.
""")
    assert akhir["b_z_in"] == akhir["b_z_rec"] and akhir["b_r_in"] == akhir["b_r_rec"]
    assert abs(akhir["b_h_in"] - akhir["b_h_rec"]) > 1e-3
    assert bg["z"][0] == bg["z"][1] and bg["r"][0] == bg["r"][1]


# ----------------------- Model mini → model penelitian --------------------- #
def bagian_6(d, k, lstm, gru):
    d.teks(f"## {JUDUL[6]}")
    nl, ng = k["lstm"]["neuron"], k["gru"]["neuron"]
    d.tabel(["Aspek", "Model mini", "Model penelitian"],
            [["Masukan per time step", "1 angka", f"vektor {k['n_fitur']} fitur"],
             ["Time step (BPTT mundur sejauh)", "2", str(k["lookback"])],
             ["Bobot per gerbang", "angka tunggal",
              f"matriks W ({k['n_fitur']} × n_u) dan U (n_u × n_u), vektor b (n_u)"],
             ["Jumlah parameter", "14", f"LSTM {ribu(k['lstm']['param'])}, GRU {ribu(k['gru']['param'])}"],
             ["Loss per iterasi", "1 sampel", f"rata-rata {k['batch']} sampel (1 batch)"],
             ["Jumlah iterasi", ribu(ITERASI),
              f"{k['iter_epoch']} per epoch × {ribu(k['lstm']['epoch'])} epoch = "
              f"{ribu(k['iter_epoch'] * k['lstm']['epoch'])}"]])
    d.teks("""
Yang sama: urutan forward → loss → BPTT → Adam, rumus setiap langkah, dan
pengaturan Adam. Setiap parameter (termasuk setiap elemen matriks) punya
gradien, m, dan v sendiri.
""")

    s_lstm = lstm["maju"]["langkah"][-1]["h"]
    y_lstm = lstm["maju"]["yhat"]
    h2b, y2b = 0.30, 0.65
    yh2 = LSTM_AWAL["W_y"] * h2b + LSTM_AWAL["b_y"]
    d1, d2 = -(2 / 2) * (Y - y_lstm), -(2 / 2) * (y2b - yh2)
    d.teks("### 6.1 Jika Memakai Batch (N > 1)")
    d.teks(r"""
Persamaan (28) memakai rata-rata $\mathcal{L} = \frac{1}{N}\sum_k (y'_k - \hat{y}'_k)^2$.
Setiap $\hat{y}'_k$ hanya muncul di suku ke-$k$, sehingga:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}'_k} = -\frac{2}{N}\,(y'_k - \hat{y}'_k) \qquad
\frac{\partial \mathcal{L}}{\partial W_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}\,h_{T,k} \qquad
\frac{\partial \mathcal{L}}{\partial b_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}$$

Setiap sampel menerima sinyal $\partial\mathcal{L}/\partial h_{T,k} =
\partial\mathcal{L}/\partial\hat{y}'_k \cdot W_y$ yang menjalankan BPTT-nya
sendiri, lalu gradien semua sampel dijumlahkan sebelum Adam dipanggil sekali.
""")
    d.kode([
        "Contoh N = 2 (sampel 1 = simulasi LSTM di atas; sampel 2 = ilustrasi h_T = 0.30, y' = 0.65)",
        f"  ŷ'₁ = {a(y_lstm)},   ŷ'₂ = 0.8 × 0.30 + 0 = {a(yh2)}",
        f"  ∂L/∂ŷ'₁ = -(2/2) × ({Y} - {a(y_lstm)}) = {a(d1)}",
        f"  ∂L/∂ŷ'₂ = -(2/2) × ({y2b} - {a(yh2)}) = {a(d2)}",
        f"  ∂L/∂W_y = {kr(d1)} × {a(s_lstm)} + {kr(d2)} × {h2b} = {kr(d1 * s_lstm)} + {kr(d2 * h2b)} = {a(d1 * s_lstm + d2 * h2b)}",
        f"  ∂L/∂b_y = {kr(d1)} + {kr(d2)} = {a(d1 + d2)}",
    ])

    d.teks("### 6.2 Versi Vektor pada Lapisan Dense")
    d.teks(f"""
Pada model penelitian, h_T berisi n_u elemen dan W_y juga berisi n_u elemen,
sehingga ŷ′ = Σⱼ h_T,ⱼ·W_y,ⱼ + b_y. Rumus 3.4 berlaku untuk setiap elemen j:
∂L/∂W_y,ⱼ = Σₖ ∂L/∂ŷ′ₖ·h_T,ₖ,ⱼ. Hasilnya n_u gradien bobot + 1 gradien bias,
yaitu **{nl + 1} parameter dense pada LSTM** dan **{ng + 1} pada GRU** (suku
(n_u + 1) pada persamaan 22 dan 27). Sinyal yang masuk ke neuron j adalah
∂L/∂h_T,ⱼ = ∂L/∂ŷ′·W_y,ⱼ.
""")

    d.teks("### 6.3 Hubungan Loss MSE dengan RMSE dalam USD")
    rmse_l = k["rentang"] * math.sqrt(k["val_loss_lstm"])
    rmse_g = k["rentang"] * math.sqrt(k["val_loss_gru"])
    assert abs(rmse_l / k["lstm"]["rmse_val"] - 1) < 1e-3
    assert abs(rmse_g / k["gru"]["rmse_val"] - 1) < 1e-3
    d.teks(f"""
Karena normalisasi min-max bersifat linear, RMSE dalam USD = (x_max − x_min) × √MSE.
Dengan rentang harga data latih {usd(k['x_max'])} − {usd(k['x_min'])} = {usd(k['rentang'])}
dan loss validasi epoch terakhir dari notebook:
""")
    d.tabel(["Model", "Loss validasi (MSE)", "Rentang × √MSE", "RMSE validasi di tabel tuning"],
            [["LSTM", f"{k['val_loss_lstm']:.8f}", f"{usd(rmse_l)} USD", f"{usd(k['lstm']['rmse_val'])} USD"],
             ["GRU", f"{k['val_loss_gru']:.8f}", f"{usd(rmse_g)} USD", f"{usd(k['gru']['rmse_val'])} USD"]])
    d.teks("""
Jadi meminimalkan MSE saat pelatihan sama artinya dengan meminimalkan RMSE dalam
USD (selisih kecil hanya karena loss validasi dicetak 8 desimal).
""")

    d.teks("### 6.4 Batasan Simulasi Ini")
    d.teks(f"""
1. Loss simulasi turun hampir ke nol karena hanya ada **1 sampel**, sehingga
   model bisa "menghafal" targetnya. Pada data penelitian dengan
   {ribu(k['n_latih'])} sampel, loss validasi LSTM terbaik berhenti di sekitar
   {k['val_loss_lstm']:.5f} karena model harus menemukan pola umum, bukan menghafal.
2. Angka akhir simulasi **tidak dapat dipakai untuk membandingkan** LSTM dan
   GRU. Bobot awalnya dipilih bulat agar mudah dihitung, sedangkan Keras
   memakai bobot acak. Perbandingan yang sah tetap hasil tahap 9-12.
""")


def bagian_7(d, k, lstm, gru, cek):
    d.teks(f"## {JUDUL[7]}")
    ll, lg = lstm["latih"], gru["latih"]
    d.kode([
        "SATU SIKLUS PELATIHAN (berlaku untuk LSTM dan GRU)",
        "",
        " 1. Inisialisasi : bobot acak, bias 0 (forget LSTM = 1), Adam m = v = 0",
        " 2. Forward      : x₁ → gerbang → h₁ (dan c₁) → x₂ → ... → h_T → ŷ' = W_y h_T + b_y",
        " 3. Loss         : L = (y' - ŷ')²  (rata-rata satu batch)",
        " 4. Dense        : ∂L/∂ŷ' = -2(y' - ŷ') → ∂L/∂W_y, ∂L/∂b_y, ∂L/∂h_T",
        " 5. Sel, t = T   : ∂L/∂h_T → turunan setiap gerbang → δ setiap gerbang",
        " 6. Through time : δ dikirim ke t-1 lewat U (dan lewat cell state / z × h_(t-1))",
        " 7. Ulangi 5-6 sampai t = 1",
        " 8. Gradien      : ∂L/∂W = Σ δx,  ∂L/∂U = Σ δh_(t-1),  ∂L/∂b = Σ δ",
        " 9. Adam         : m, v → m̂, v̂ → θ baru = θ - η m̂/(√v̂ + ε)",
        "10. Kembali ke langkah 2 dengan batch berikutnya",
        "",
        "HASIL SIMULASI",
        f"  LSTM: loss {a(ll['L'][0])} → {a(ll['L'][1])} (k = 1) → {fl(ll['L'][-1])} (k = {ITERASI}),"
        f" ŷ' {a(ll['yhat'][0])} → {a(ll['yhat'][-1])}",
        f"  GRU : loss {a(lg['L'][0])} → {a(lg['L'][1])} (k = 1) → {fl(lg['L'][-1])} (k = {ITERASI}),"
        f" ŷ' {a(lg['yhat'][0])} → {a(lg['yhat'][-1])}",
    ])
    d.teks("**Pemeriksaan otomatis yang lolos saat berkas ini dibuat:**")
    d.teks("\n".join(f"- {c}" for c in cek))


# --------------------------------------------------------------------------- #
def main():
    k = baca_konteks()
    assert abs(k["lr"] - LR) < 1e-12, "Learning rate simulasi harus sama dengan penelitian"

    hasil = {}
    for model, maju_fn, mundur_fn, p0 in (("LSTM", lstm_maju, lstm_mundur, LSTM_AWAL),
                                          ("GRU", gru_maju, gru_mundur, GRU_AWAL)):
        maju = maju_fn(p0)
        mundur = mundur_fn(p0, maju)
        num = gradien_numerik(maju_fn, p0)
        selisih = max(abs(mundur["g"][n] - num[n]) for n in p0)
        assert selisih < 1e-8, f"{model}: gradien BPTT tidak cocok dengan gradien numerik"
        latih = latih_adam(maju_fn, mundur_fn, p0, ITERASI)
        langkah1 = latih["riwayat"][0]["delta"]
        assert all(abs(abs(v) - LR) < 1e-6 for v in langkah1.values()), \
            f"{model}: langkah Adam pertama harus ±η"
        assert latih["L"][-1] < 1e-4 < latih["L"][0]
        hasil[model] = {"maju": maju, "mundur": mundur, "num": num, "latih": latih, "selisih": selisih}

    cek = [
        f"Gradien BPTT manual = gradien numerik untuk 14 parameter LSTM "
        f"(selisih maksimum {hasil['LSTM']['selisih']:.1e}) dan 14 parameter GRU "
        f"(selisih maksimum {hasil['GRU']['selisih']:.1e}).",
        "Uji geser pada lapisan dense: perubahan loss sebenarnya = gradien × 0.001 "
        "(toleransi 2e-6).",
        "Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter.",
        "Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah "
        f"{ITERASI} iterasi, untuk LSTM maupun GRU.",
        "Bias z dan r GRU tetap kembar (bias masukan = bias rekuren), bias kandidat "
        "berbeda, baik pada simulasi maupun model terlatih.",
        "Rentang harga × √(loss validasi) = RMSE validasi pada tabel tuning "
        "(toleransi 0.1%) untuk LSTM dan GRU.",
    ]

    d = Dokumen()
    bagian_pembuka(d, k)
    bagian_0(d, k)
    bagian_1(d, k)
    bagian_2(d, k)
    h = hasil["LSTM"]
    bagian_3(d, k, LSTM_AWAL, h["maju"], h["mundur"], h["num"], h["latih"])
    h = hasil["GRU"]
    bagian_4(d, k, GRU_AWAL, h["maju"], h["mundur"], h["num"], h["latih"])
    bagian_5(d, k, hasil["LSTM"], hasil["GRU"])
    bagian_6(d, k, hasil["LSTM"], hasil["GRU"])
    bagian_7(d, k, hasil["LSTM"], hasil["GRU"], cek)

    with open(BERKAS_KELUARAN, "w", encoding="utf-8") as f:
        f.write(d.isi())
    print(f"[tersimpan] {os.path.relpath(BERKAS_KELUARAN, AKAR)}")
    for c in cek:
        print(f"  OK {c}")


if __name__ == "__main__":
    main()
