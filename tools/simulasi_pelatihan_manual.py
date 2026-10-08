# -*- coding: utf-8 -*-
"""
Simulasi pelatihan bobot LSTM dan GRU yang dihitung manual langkah demi
langkah: forward pass, loss MSE, backpropagation through time (BPTT), gradien
bobot dan bias, lalu pembaruan parameter dengan Adam.

Agar setiap angka dapat diikuti dengan tangan, simulasi memakai model mini
(1 neuron, 1 fitur, 2 time step, 1 sampel). Rumus yang dipakai sama dengan
rumus Keras pada model penelitian (persamaan 15-31 pada draf skripsi).

Keluaran berupa seri empat berkas yang dibaca berurutan, semuanya memakai angka
simulasi yang sama sehingga saling menyambung:
    outputs/perhitungan_manual/tahap_7_8_1_alur_sel_lstm_gru.md      (Gambar 1-7)
    outputs/perhitungan_manual/tahap_7_8_2_fungsi_loss_dan_adam.md   (subbab 1.5.11)
    outputs/perhitungan_manual/tahap_7_8_3_simulasi_pelatihan_lstm_gru.md
    outputs/perhitungan_manual/tahap_7_8_4_bias_lstm_gru.md

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
DIR_GAMBAR = os.path.join(AKAR, "outputs", "gambar")

# Seri berkas keluaran, dibaca berurutan
SERI = [
    ("tahap_7_8_1_alur_sel_lstm_gru.md", "Alur Sel LSTM dan GRU (Gambar 1-7)"),
    ("tahap_7_8_2_fungsi_loss_dan_adam.md", "Fungsi Loss dan Adam"),
    ("tahap_7_8_3_simulasi_pelatihan_lstm_gru.md", "Simulasi Satu Siklus Pelatihan LSTM dan GRU"),
    ("tahap_7_8_4_bias_lstm_gru.md", "Bias pada LSTM dan GRU"),
]

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
    semua = _csv(pola)
    baris = [b for b in semua if "TERBAIK" in (b.get("Keterangan") or "")]
    assert len(baris) == 1, f"Model terbaik pada {pola} tidak ditemukan"
    b = baris[0]
    hasil = {"neuron": int(b["Neuron"]), "epoch": int(b["Epoch"]),
             "rmse_val": float(b["RMSE Validasi (USD)"]),
             "param": int(b["Jumlah Parameter"])}
    # RMSE validasi neuron terbaik untuk setiap jumlah epoch pada grid
    hasil["per_epoch"] = {int(x["Epoch"]): float(x["RMSE Validasi (USD)"])
                          for x in semua if int(x["Neuron"]) == hasil["neuron"]}
    return hasil


def _gambar(awalan):
    berkas = sorted(glob.glob(os.path.join(DIR_GAMBAR, awalan + "*.png")))
    assert berkas, f"Gambar {awalan}*.png tidak ditemukan di {DIR_GAMBAR}"
    return "../gambar/" + os.path.basename(berkas[0])


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
    bentuk_val = re.findall(r"shape X_val\s+=\s+\((\d+), (\d+), (\d+)\)", keluaran)
    assert len(bentuk_val) == 1, "Bentuk X_val tidak ditemukan pada keluaran notebook"
    k["n_val"] = int(bentuk_val[0][0])
    # Teks Bagian 2 menyatakan data tidak diacak dan tanpa early stopping
    assert "shuffle=False" in sumber and "EarlyStopping" not in sumber
    kurva = {kunci: _angka(pola, keluaran, 2) for kunci, pola in (
        ("latih_awal", r"Loss latih[ \t]+epoch pertama[ \t]*:[ \t]*([0-9.]+)"),
        ("latih_akhir", r"Loss latih[ \t]+epoch terakhir[ \t]*:[ \t]*([0-9.]+)"),
        ("val_awal", r"Loss validasi epoch pertama[ \t]*:[ \t]*([0-9.]+)"),
        ("val_akhir", r"Loss validasi epoch terakhir[ \t]*:[ \t]*([0-9.]+)"),
        ("val_min", r"Loss validasi minimum[ \t]*:[ \t]*([0-9.]+)"),
        ("epoch_min", r"Loss validasi minimum[ \t]*:[ \t]*[0-9.]+ \(epoch (\d+)\)"))}
    k["kurva"] = {m: {kunci: nilai[i] for kunci, nilai in kurva.items()}
                  for i, m in enumerate(("lstm", "gru"))}
    for m in ("lstm", "gru"):
        k["kurva"][m]["epoch_min"] = int(k["kurva"][m]["epoch_min"])
    k["gambar_loss_lstm"] = _gambar("gambar_03_")
    k["gambar_loss_gru"] = _gambar("gambar_04_")

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
    1: "1. Asumsi Simulasi, Notasi, dan Siklus yang Dihitung",
    2: "2. Rumus Turunan Dasar yang Dipakai",
    3: "3. LSTM: Satu Siklus Pelatihan Langkah demi Langkah",
    4: "4. GRU: Satu Siklus Pelatihan Langkah demi Langkah",
    5: "5. Dari Model Mini ke Model Penelitian",
    6: "6. Ringkasan Alur dan Hasil Verifikasi",
}

CATATAN_ANGKA = """
**Cara membaca angka.** Seperti berkas perhitungan manual lainnya, angka memakai
titik sebagai pemisah desimal dan koma sebagai pemisah ribuan. Angka ditampilkan
6 desimal, tetapi skrip menghitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).
"""


def tautan(nomor, sub=None):
    """Tautan Markdown ke bagian lain dalam seri (opsional ke judul tertentu)."""
    berkas, judul = SERI[nomor - 1]
    if sub is None:
        return f"[Bagian {nomor}]({berkas})"
    return f"[Bagian {nomor}, subbagian {sub.split('.')[0]}]({berkas}#{jangkar(sub)})"


def navigasi(d, nomor):
    baris = ["**Seri perhitungan manual pelatihan LSTM dan GRU** (baca berurutan):", ""]
    for i, (berkas, judul) in enumerate(SERI, 1):
        if i == nomor:
            baris.append(f"{i}. **{judul}** ← sedang dibaca")
        else:
            baris.append(f"{i}. [{judul}]({berkas})")
    d.teks("\n".join(baris))


def kepala(d, nomor):
    d.teks(f"# Perhitungan Manual - Tahap 7 & 8 (Bagian {nomor} dari {len(SERI)}): {SERI[nomor - 1][1]}")
    d.teks("""
> Berkas ini dibuat otomatis oleh `tools/simulasi_pelatihan_manual.py`. Semua
> angka dihitung ulang oleh skrip itu dan diperiksa dengan `assert`. Untuk membuat
> ulang seluruh seri: `python tools/simulasi_pelatihan_manual.py`.
""")
    navigasi(d, nomor)


def penutup(d, nomor):
    baris = ["---", ""]
    if nomor > 1:
        berkas, judul = SERI[nomor - 2]
        baris.append(f"← Sebelumnya: [Bagian {nomor - 1}. {judul}]({berkas})")
        baris.append("")
    if nomor < len(SERI):
        berkas, judul = SERI[nomor]
        baris.append(f"→ Berikutnya: [Bagian {nomor + 1}. {judul}]({berkas})")
    else:
        baris.append(f"Kembali ke awal seri: [Bagian 1. {SERI[0][1]}]({SERI[0][0]})")
    d.teks("\n".join(baris))


def daftar_isi(d, judul):
    d.teks("**Daftar isi**")
    d.teks("\n".join(f"- [{j}](#{jangkar(j)})" for j in judul.values()))


# ------------------------------- Bagian 3 ---------------------------------- #
def bagian_pembuka(d, k):
    kepala(d, 3)
    d.teks(f"""
{tautan(1)} menjelaskan makna setiap gerbang pada Gambar 1-7, dan {tautan(2)}
menjelaskan konsep fungsi loss serta Adam. Bagian ini **menghitung satu siklus
pelatihan secara lengkap**: *forward pass* → *loss* → *backpropagation through time*
(BPTT) → Adam, sampai bobot dan bias berubah dan loss turun.

Berkas `tahap_7_forward_pass_lstm.md` dan `tahap_8_forward_pass_gru.md`
menunjukkan bagaimana model yang **sudah dilatih** menghasilkan prediksi. Bagian
ini menjawab pertanyaan sebelumnya: **bagaimana bobot dan bias itu diperoleh?**

Agar setiap angka bisa diikuti dengan tangan, simulasi memakai **model mini**:
1 neuron, 1 fitur, 2 time step, dan 1 sampel. Rumus dan urutan langkahnya
**identik** dengan model penelitian; yang berbeda hanya jumlah angkanya.
Subbagian 5 menjelaskan cara memperbesarnya ke model penelitian
({k['n_fitur']} fitur, {k['lookback']} time step, batch {k['batch']},
{k['lstm']['neuron']}/{k['gru']['neuron']} neuron).
""")
    daftar_isi(d, JUDUL)
    d.teks(CATATAN_ANGKA)


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
    d.teks("""
**Siklus yang dihitung.** Satu siklus di bawah ini disebut **satu iterasi k**.
Model penelitian mengulangnya ribuan kali; berapa kali dan kapan persisnya
dijelaskan di """ + tautan(2, JUDUL_2[2]) + ".")
    d.kode("""
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
 Bobot & bias θ ─► (1) FORWARD ─► (2) LOSS ─► (3) BACKWARD (BPTT) ─► (4) ADAM
                    prediksi ŷ'    seberapa     gradien g = ∂L/∂θ      θ baru
                                   salah?       untuk setiap θ
""")
    d.tabel(
        ["Langkah", "Pertanyaan yang dijawab", "Persamaan skripsi", "LSTM", "GRU"],
        [["(1) Forward pass", "Dengan bobot sekarang, berapa prediksinya?",
          "(15)-(21) LSTM, (23)-(26) GRU", "3.2", "4.2"],
         ["(2) Loss", "Seberapa salah prediksinya?", "(28)", "3.3", "4.3"],
         ["(3) Backward (BPTT)", "Bobot mana yang menyebabkan salah, ke arah mana?",
          "aturan rantai (subbagian 2)", "3.4-3.9", "4.4-4.9"],
         ["(4) Adam", "Seberapa jauh setiap bobot digeser?", "(29)-(31)", "3.10-3.13", "4.10-4.13"]])


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
    d.teks(f"Makna setiap gerbang dan alurnya pada Gambar 1-6 dijelaskan di {tautan(1)}; "
           "di sini dihitung angkanya untuk kedua time step.")
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
Konsep MSE dan alasan pemakaiannya dibahas di {tautan(2, JUDUL_2[3])}.
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
        d.teks("Penjelasan momentum dan koreksi bias pada 3.12 berlaku sama untuk GRU.")
    else:
        d.teks(f"""
m₂ menggabungkan arah gradien iterasi 1 dan 2. Jika suatu saat gradien berbalik arah
(misalnya ketika bobot melewati titik minimum), m akan mengecil dan langkahnya
otomatis melambat. Koreksi bias (1 − βᵏ) makin lama makin mendekati 1, sehingga
pengaruhnya hilang setelah banyak iterasi; tabelnya ada di {tautan(2, JUDUL_2[8])}.
""")

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
memakai `reset_after=True` (lihat """ + tautan(4, JUDUL_4[4]) + """).
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
    d.teks(f"Alur Gambar 7 langkah demi langkah dijelaskan di {tautan(1, JUDUL_1[9])}; "
           "di sini dihitung angkanya untuk kedua time step.")
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
berbeda. {tautan(4, JUDUL_4[4])} menjelaskan sebabnya.
""")


# ------------------------------- Bias -------------------------------------- #
# ------------------------------- Bagian 1 ---------------------------------- #
JUDUL_1 = {
    1: "1. Cara Membaca Gambar",
    2: "2. Gambar 1: Struktur LSTM",
    3: "3. Gambar 2: Cell State",
    4: "4. Gambar 3: Forget Gate",
    5: "5. Gambar 4: Input Gate",
    6: "6. Gambar 5: Pembaruan Cell State",
    7: "7. Gambar 6: Output Gate",
    8: "8. Ringkasan Satu Langkah LSTM",
    9: "9. Gambar 7: Struktur Sel GRU",
    10: "10. Hubungan LSTM dan GRU",
    11: "11. Catatan untuk Naskah Skripsi",
}


def seri_1_alur_sel(d, k, lstm, gru):
    p, q = LSTM_AWAL, GRU_AWAL
    s2 = lstm["maju"]["langkah"][1]
    g2 = gru["maju"]["langkah"][1]
    r1 = lstm["mundur"]["rincian"][0]
    rg2 = gru["mundur"]["rincian"][1]
    porsi_c = r1["dc_lanjut"] / r1["dc"]
    porsi_z = rg2["suku_h"][0] / rg2["dh_prev"]
    kepala(d, 1)
    d.teks(f"""
Bagian ini menjelaskan alur **Gambar 1 sampai Gambar 7** pada subbab 1.5.8 (LSTM)
dan 1.5.9 (GRU) langkah demi langkah: apa yang mengalir di setiap garis, apa yang
dikerjakan setiap kotak, dan apa fungsinya.

Supaya setiap langkah punya angka nyata, contoh angka diambil dari **simulasi
model mini** yang dihitung lengkap di {tautan(3)}: 1 neuron, 1 fitur, data
x₁ = {X[0]} dan x₂ = {X[1]}, bobot awal bulat (W_f = {bt(p['W_f'])}, U_f = {bt(p['U_f'])},
b_f = {bt(p['b_f'])}, dan seterusnya; tabel lengkapnya di {tautan(3, JUDUL[3])}).
Contoh memakai **time step t = 2**, karena di sana memori dari t = 1 sudah ikut
bekerja. Angka yang sama dipakai lagi di Bagian 2 dan 3 saat menghitung loss,
gradien, dan Adam, sehingga seluruh seri saling menyambung.
""")
    daftar_isi(d, JUDUL_1)
    d.teks(CATATAN_ANGKA)

    # 1 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[1]}")
    d.teks("Gambar 1-6 berasal dari Olah (2015), dan Gambar 7 mengikuti gaya yang sama. Simbolnya:")
    d.tabel(["Simbol", "Arti"],
            [["Kotak kuning (σ atau tanh)",
              "lapisan jaringan saraf yang **punya bobot dan bias** (W, U, b) dan dipelajari saat pelatihan"],
             ["Lingkaran merah muda (×, +, 1−)",
              "operasi elemen demi elemen **tanpa bobot**; × = perkalian Hadamard (⊙), + = penjumlahan"],
             ["Oval merah muda \"tanh\" (Gambar 1 dan 6)",
              "fungsi tanh biasa **tanpa bobot**, hanya memampatkan nilai ke rentang -1 sampai 1"],
             ["Dua garis menyatu",
              "hₜ₋₁ dan xₜ menjadi masukan bersama sebuah gerbang (di skripsi ditulis xₜW + hₜ₋₁U)"],
             ["Satu garis bercabang", "nilai yang sama disalin ke dua tujuan"],
             ["Cₜ (huruf besar) pada gambar Olah", "sama dengan cₜ (cell state) pada persamaan skripsi"]])
    d.teks("""
**Mengapa σ dipakai untuk gerbang dan tanh untuk isi?**

- **Sigmoid (0 sampai 1)** berfungsi seperti **keran**: 0 berarti tertutup, 1 berarti
  terbuka penuh, 0.5 berarti setengah. Nilainya adalah *proporsi*, jadi selalu dipakai
  untuk **mengalikan** sesuatu.
- **Tanh (-1 sampai 1)** berfungsi sebagai **isi informasi**. Nilainya bisa positif
  atau negatif sehingga memori bisa dinaikkan atau diturunkan, dan tetap terbatas
  sehingga tidak meledak.
""")

    # 2 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[2]}")
    d.teks(f"""
**Alurnya:**

1. Ketiga kotak hijau berlabel **A** adalah **sel yang sama** (bobot yang sama) yang
   digambar berulang untuk setiap waktu: kiri untuk t − 1, tengah untuk t, kanan
   untuk t + 1. Cara menggambar ini disebut *unrolling*.
2. Setiap sel menerima **masukan dari bawah** (xₜ₋₁, xₜ, xₜ₊₁) dan menghasilkan
   **keluaran ke atas** (hₜ₋₁, hₜ, hₜ₊₁).
3. Di antara sel ada **dua garis horizontal** yang membawa memori ke langkah
   berikutnya: garis **atas** adalah **cell state cₜ** (memori jangka panjang), garis
   **bawah** adalah **hidden state hₜ** (memori jangka pendek sekaligus keluaran).
4. Isi sel tengah adalah 4 kotak kuning, dari kiri: **σ (forget gate), σ (input
   gate), tanh (kandidat), σ (output gate)**. Gambar 3-6 membedahnya satu per satu.

**Fungsinya:** RNN biasa (persamaan 10) hanya punya satu garis, yaitu h. LSTM
menambahkan garis cₜ dan tiga gerbang yang mengatur apa yang dibuang, disimpan,
dan dikeluarkan.

**Kaitan dengan penelitian:** dengan *window* {k['lookback']} hari, sel A dijalankan
**{k['lookback']} kali berturut-turut** (xₜ₋₆ sampai xₜ), dan setiap xₜ berisi
{k['n_fitur']} fitur blockchain ternormalisasi. Di awal, h₀ = c₀ = 0 (bawaan Keras).
Hanya hidden state **langkah terakhir h_T** yang diteruskan ke lapisan dense
(persamaan 21). Model mini di seri ini sama, hanya dijalankan 2 kali (t = 1 dan
t = 2), lalu h₂ masuk ke dense.
""")

    # 3 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[3]}")
    d.teks(f"""
**Alurnya:**

1. cₜ₋₁ masuk dari kiri.
2. cₜ₋₁ melewati lingkaran **×**: sebagian memori lama dihapus (dikalikan fₜ).
3. Hasilnya melewati lingkaran **+**: informasi baru ditambahkan (iₜ × c̃ₜ).
4. Keluar ke kanan sebagai **cₜ** dan masuk ke langkah waktu berikutnya.

**Fungsinya:** cell state adalah **jalan tol memori**. Di sepanjang garis ini hanya
ada perkalian dan penjumlahan sederhana; tidak ada perkalian matriks bobot dan tidak
ada fungsi aktivasi yang berulang.

Inilah alasan LSTM tahan terhadap *vanishing gradient*. Saat *backpropagation*,
gradien yang mengalir mundur di jalur ini hanya dikalikan fₜ. Jika model memutuskan
untuk mengingat (fₜ mendekati 1), gradien hampir tidak mengecil. Buktinya ada pada
perhitungan {tautan(3)} subbagian 3.7: **{persen(porsi_c)}** sinyal kesalahan yang
sampai ke c₁ datang lewat jalur cell state ini.
""")

    # 4 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[4]}")
    d.teks("""
**Alurnya:**

1. **hₜ₋₁** (masuk dari kiri) dan **xₜ** (masuk dari bawah) bertemu dan menjadi
   masukan bersama.
2. Keduanya masuk ke **kotak kuning σ pertama** dan dihitung pra-aktivasinya:
   xₜW_f + hₜ₋₁U_f + b_f (persamaan 15).
3. Hasilnya dilewatkan ke sigmoid sehingga menjadi **fₜ**, bernilai 0 sampai 1.
4. Panah fₜ naik ke lingkaran **×** di garis cell state dan **mengalikan cₜ₋₁**.

**Fungsinya:** menentukan **berapa banyak memori lama yang dipertahankan**. fₜ
mendekati 0 berarti memori dihapus; fₜ mendekati 1 berarti memori dipertahankan utuh.
""")
    d.kode([
        f"Contoh angka (t = 2). Dari t = 1 sudah diperoleh h₁ = {a(s2['h_prev'])} dan c₁ = {a(s2['c_prev'])}.",
        f"  a  = W_f × x₂ + U_f × h₁ + b_f = {bt(p['W_f'])} × {X[1]} + {bt(p['U_f'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_f'])} = {a(s2['a_f'])}",
        f"  f₂ = σ({a(s2['a_f'])}) = {a(s2['f'])}",
        f"  Memori lama yang dipertahankan: f₂ × c₁ = {a(s2['f'])} × {a(s2['c_prev'])} = {a(s2['f'] * s2['c_prev'])}",
        f"  Artinya {persen(s2['f'])} isi memori lama dipertahankan dan {persen(1 - s2['f'])} dibuang.",
    ])
    d.teks(f"""
Bias forget bernilai 1 (*unit forget bias* Keras). Karena itu, walaupun xₜ dan hₜ₋₁
bernilai nol, gerbang ini tetap bernilai σ(1) = {a(sig(1.0), 3)}: secara bawaan model
cenderung **mengingat**. Peran bias dijelaskan lengkap di {tautan(4)}.
""")

    # 5 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[5]}")
    d.teks("""
Gambar ini memiliki **dua cabang paralel** dari masukan yang sama (hₜ₋₁ dan xₜ).

**Alurnya:**

1. **Cabang kiri, kotak σ kedua**, menghasilkan **iₜ** (0 sampai 1, persamaan 16):
   keran yang menentukan **berapa banyak** informasi baru boleh masuk.
2. **Cabang kanan, kotak tanh**, menghasilkan **c̃ₜ** (-1 sampai 1, persamaan 17):
   **isi informasi baru** yang diusulkan.
3. Kedua cabang bertemu di lingkaran **×** sehingga menjadi iₜ × c̃ₜ.
4. Hasilnya naik ke lingkaran **+** di garis cell state.

**Fungsinya:** menentukan **informasi baru apa yang ditulis ke memori**. c̃ₜ menjawab
"apa isinya?", sedangkan iₜ menjawab "seberapa penting untuk disimpan?". Karena c̃ₜ
bisa negatif, informasi baru bisa menaikkan atau menurunkan memori.
""")
    d.kode([
        "Contoh angka (t = 2):",
        f"  i₂ = σ(W_i × x₂ + U_i × h₁ + b_i) = σ({bt(p['W_i'])} × {X[1]} + {bt(p['U_i'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_i'])}) = σ({a(s2['a_i'])}) = {a(s2['i'])}",
        f"  c̃₂ = tanh(W_c × x₂ + U_c × h₁ + b_c) = tanh({bt(p['W_c'])} × {X[1]} + {bt(p['U_c'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_c'])}) = tanh({a(s2['a_c'])}) = {a(s2['cc'])}",
        f"  Informasi baru yang ditulis: i₂ × c̃₂ = {a(s2['i'])} × {a(s2['cc'])} = {a(s2['i'] * s2['cc'])}",
        f"  Artinya usulan isi {a(s2['cc'])} hanya masuk {persen(s2['i'])}.",
    ])

    # 6 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[6]}")
    d.teks("""
Gambar ini menggabungkan hasil Gambar 3 dan Gambar 4 di garis atas (persamaan 18):

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

**Alurnya:** cₜ₋₁ dikalikan fₜ (memori lama terseleksi), lalu ditambah iₜ × c̃ₜ
(memori baru terseleksi). Hasilnya adalah **cₜ**, memori jangka panjang yang sudah
diperbarui, yang dikirim ke langkah waktu berikutnya.

**Fungsinya:** di sinilah memori benar-benar ditulis ulang. Karena fₜ dan iₜ
**independen**, LSTM bisa sekaligus mempertahankan banyak memori lama dan menambah
banyak memori baru. Ini salah satu pembeda utama dengan GRU (subbagian 10).
""")
    d.kode([
        "Contoh angka (t = 2), memakai hasil Gambar 3 dan Gambar 4:",
        f"  c₂ = f₂ × c₁ + i₂ × c̃₂ = {a(s2['f'] * s2['c_prev'])} + {a(s2['i'] * s2['cc'])} = {a(s2['c'])}",
        f"  Memori berubah dari c₁ = {a(s2['c_prev'])} menjadi c₂ = {a(s2['c'])}.",
    ])

    # 7 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[7]}")
    d.teks("""
**Alurnya:**

1. hₜ₋₁ dan xₜ masuk ke **kotak σ keempat** dan menghasilkan **oₜ** (0 sampai 1,
   persamaan 19).
2. Sementara itu, **cₜ** dari garis atas bercabang turun ke **oval tanh** (tanpa bobot)
   yang memampatkan cₜ ke rentang -1 sampai 1.
3. tanh(cₜ) dan oₜ bertemu di lingkaran **×** sehingga menjadi **hₜ = oₜ ⊙ tanh(cₜ)**
   (persamaan 20).
4. hₜ **bercabang dua**: naik ke atas sebagai keluaran waktu t, dan ke kanan sebagai
   hₜ₋₁ untuk langkah berikutnya.

**Fungsinya:** menentukan **bagian memori mana yang ditampilkan** sebagai keluaran
saat ini. Tidak semua isi cₜ relevan untuk keluaran sekarang; sebagian cukup
disimpan untuk dipakai nanti.
""")
    d.kode([
        "Contoh angka (t = 2):",
        f"  o₂ = σ(W_o × x₂ + U_o × h₁ + b_o) = σ({bt(p['W_o'])} × {X[1]} + {bt(p['U_o'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_o'])}) = σ({a(s2['a_o'])}) = {a(s2['o'])}",
        f"  tanh(c₂) = tanh({a(s2['c'])}) = {a(s2['tc'])}        (oval tanh, tanpa bobot)",
        f"  h₂ = o₂ × tanh(c₂) = {a(s2['o'])} × {a(s2['tc'])} = {a(s2['h'])}",
        "  t = 2 adalah langkah terakhir, jadi h₂ masuk ke lapisan dense (persamaan 21):",
        f"  ŷ' = W_y × h₂ + b_y = {bt(p['W_y'])} × {a(s2['h'])} + {bt(p['b_y'])} = {a(lstm['maju']['yhat'])}",
    ])

    # 8 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[8]}")
    d.teks(f"Seluruh alur satu time step LSTM (contoh t = 2, memori masuk c₁ = {a(s2['c_prev'])} "
           f"dan h₁ = {a(s2['h_prev'])}):")
    d.tabel(["Urutan", "Gambar", "Perhitungan", "Hasil", "Makna"],
            [["1. Forget", "Gambar 3", "f₂ = σ(a); f₂ × c₁", f"{a(s2['f'])}; {a(s2['f'] * s2['c_prev'])}",
              f"{persen(s2['f'])} memori lama dipertahankan"],
             ["2. Input", "Gambar 4", "i₂ = σ(a); c̃₂ = tanh(a); i₂ × c̃₂",
              f"{a(s2['i'])}; {a(s2['cc'])}; {a(s2['i'] * s2['cc'])}", f"usulan baru masuk {persen(s2['i'])}"],
             ["3. Update", "Gambar 5", "c₂ = f₂c₁ + i₂c̃₂", a(s2["c"]), "memori jangka panjang baru"],
             ["4. Output", "Gambar 6", "o₂ = σ(a); h₂ = o₂ tanh(c₂)", f"{a(s2['o'])}; {a(s2['h'])}",
              f"{persen(s2['o'])} memori yang sudah dimampatkan dikeluarkan"],
             ["5. Dense", "-", "ŷ′ = W_y h₂ + b_y", a(lstm["maju"]["yhat"]), "prediksi (skala ternormalisasi)"]])
    d.teks(f"""
Prediksi ŷ′ = {a(lstm['maju']['yhat'])} masih jauh dari target {Y}. Seberapa salah
prediksi ini dan bagaimana bobot diperbaiki dijelaskan di {tautan(2)} (konsep loss
dan Adam) dan {tautan(3)} (perhitungan lengkapnya).
""")

    # 9 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[9]}")
    d.teks("""
GRU **hanya punya satu garis memori**, yaitu hₜ (garis atas). Tidak ada cₜ
terpisah dan tidak ada output gate. Gambar 7 dibaca dalam enam langkah berikut;
contoh angkanya memakai time step t = 2 dari simulasi GRU mini.
""")
    d.teks(f"""
**Langkah 0 — masukan.**

- hₜ₋₁ masuk dari **kiri atas**, lalu bercabang: tetap di garis atas, turun lewat
  garis vertikal kiri ke jalur bawah, dan bercabang ke tengah menuju lingkaran ×
  milik rₜ.
- xₜ masuk dari **bawah** ke jalur horizontal bawah.
- Jalur bawah membawa hₜ₋₁ dan xₜ ke ketiga kotak kuning: σ (z), σ (r), dan tanh.
- Contoh: h₁ = {a(g2['h_prev'])} (hasil t = 1) dan x₂ = {X[1]}.

**Langkah 1 — update gate (kotak σ kiri) menghasilkan zₜ** (persamaan 23).

- z₂ = σ(W_z × x₂ + U_z × h₁ + b_z(in) + b_z(rec)) = σ({a(g2['a_z'])}) = **{a(g2['z'])}**.
- Panah zₜ naik dan **bercabang dua**: ke **lingkaran × kiri atas**, mengalikan
  hₜ₋₁ menjadi z₂ × h₁ = {a(g2['z'])} × {a(g2['h_prev'])} = **{a(g2['z'] * g2['h_prev'])}**;
  dan ke kanan menuju **lingkaran "1−"**, menghasilkan 1 − z₂ = **{a(1 - g2['z'])}**.

**Langkah 2 — reset gate (kotak σ tengah) menghasilkan rₜ** (persamaan 24).

- r₂ = σ(W_r × x₂ + U_r × h₁ + b_r(in) + b_r(rec)) = σ({a(g2['a_r'])}) = **{a(g2['r'])}**.
- Panah rₜ naik ke **lingkaran × tengah**, tempat ia mengalikan memori lama yang datang
  dari kiri.

**Langkah 3 — kandidat (kotak tanh) menghasilkan h̃ₜ** (persamaan 25).

- Cabang memori lama melewati lingkaran × milik rₜ sebelum masuk tanh. Pada Keras
  (`reset_after=True`), yang dikalikan rₜ adalah bagian rekuren
  qₜ = hₜ₋₁U_h + b_h(rec): q₂ = {bt(q['U_h'])} × {a(g2['h_prev'])} + {bt(q['b_h_rec'])} = {a(g2['q'])},
  sehingga r₂ × q₂ = {a(g2['rq'])}.
- xₜ masuk dari bawah: x₂W_h + b_h(in) = {bt(q['W_h'])} × {X[1]} + {bt(q['b_h_in'])} = {a(g2['masuk'])}.
- h̃₂ = tanh({a(g2['masuk'])} + {a(g2['rq'])}) = tanh({a(g2['a_h'])}) = **{a(g2['hh'])}**.
- Fungsi rₜ: menentukan **seberapa banyak masa lalu dipakai untuk menyusun usulan
  baru**. rₜ mendekati 0 berarti kandidat disusun hampir hanya dari xₜ; rₜ mendekati 1
  berarti masa lalu ikut diperhitungkan penuh. Pada t = 1, q₁ = 0 karena h₀ = 0, jadi
  reset gate belum berpengaruh.

**Langkah 4 — pencampuran** (persamaan 26).

- h̃ₜ naik ke **lingkaran × kanan** dan dikalikan (1 − zₜ):
  {a(1 - g2['z'])} × {a(g2['hh'])} = {a((1 - g2['z']) * g2['hh'])}.
- Hasilnya naik ke **lingkaran +** dan dijumlahkan dengan zₜ × hₜ₋₁ dari kiri:
  h₂ = {a(g2['z'] * g2['h_prev'])} + {a((1 - g2['z']) * g2['hh'])} = **{a(g2['h'])}**.

**Langkah 5 — keluaran.** h₂ keluar ke **kanan** (ke langkah berikutnya) dan ke
**atas** (keluaran waktu t). Karena t = 2 adalah langkah terakhir, h₂ masuk ke dense:
ŷ′ = {bt(q['W_y'])} × {a(g2['h'])} + {bt(q['b_y'])} = **{a(gru['maju']['yhat'])}**.
""")
    d.teks(f"""
**Fungsi utama zₜ:** zₜ bekerja seperti **penggeser (slider) pencampur**. zₜ mendekati 1
berarti hₜ hampir sama dengan hₜ₋₁ (memori lama dipertahankan); zₜ mendekati 0 berarti hₜ
hampir sama dengan h̃ₜ (diganti informasi baru). Porsi lama dan porsi baru **selalu
berjumlah 1**. Pada contoh di atas: {persen(g2['z'])} memori lama + {persen(1 - g2['z'])}
kandidat baru.
""")
    d.tabel(["Langkah", "Bagian Gambar 7", "Hasil (t = 2)"],
            [["1. Update gate", "kotak σ kiri", f"z₂ = {a(g2['z'])}"],
             ["2. Reset gate", "kotak σ tengah", f"r₂ = {a(g2['r'])}"],
             ["3. Kandidat", "lingkaran × tengah, kotak tanh", f"r₂q₂ = {a(g2['rq'])}; h̃₂ = {a(g2['hh'])}"],
             ["4. Pencampuran", "lingkaran × kiri, 1−, × kanan, +",
              f"{a(g2['z'] * g2['h_prev'])} + {a((1 - g2['z']) * g2['hh'])} = {a(g2['h'])}"],
             ["5. Dense", "-", f"ŷ′ = {a(gru['maju']['yhat'])}"]])

    # 10 --------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[10]}")
    d.tabel(["LSTM", "GRU", "Penjelasan"],
            [["fₜ (forget) dan iₜ (input), **independen**", "zₜ dan (1 − zₜ), **terikat**",
              "GRU menggabungkan keduanya menjadi satu update gate: menyimpan lebih banyak "
              "yang lama berarti menerima lebih sedikit yang baru"],
             ["cₜ dan hₜ (dua memori)", "hanya hₜ", "memori langsung disimpan di hidden state"],
             ["oₜ (output gate)", "tidak ada", "seluruh hₜ langsung dikeluarkan"],
             ["tidak ada padanan langsung", "rₜ (reset gate)", "mengatur peran masa lalu saat menyusun kandidat"],
             ["jalur cell state (Gambar 2)", "jalur langsung zₜ ⊙ hₜ₋₁",
              f"\"jalan tol gradien\"; pada simulasi, {persen(porsi_c)} (LSTM) dan {persen(porsi_z)} (GRU) "
              "sinyal kesalahan mengalir lewat jalur ini"],
             ["4 himpunan bobot (i, f, c, o)", "3 himpunan bobot (z, r, h)", "parameter GRU lebih sedikit"]])
    nf = k["n_fitur"]
    p_lstm = lambda n: 4 * (n * (n + nf) + n)
    p_gru = lambda n: 3 * (n * (n + nf) + 2 * n)
    nl, ng = k["lstm"]["neuron"], k["gru"]["neuron"]
    assert p_lstm(nl) + nl + 1 == k["lstm"]["param"] and p_gru(ng) + ng + 1 == k["gru"]["param"]
    d.teks(f"Jumlah parameter lapisan rekuren dengan {nf} fitur (persamaan 22 dan 27, tanpa dense):")
    d.tabel(["Neuron n_u", f"LSTM: 4[n_u(n_u + {nf}) + n_u]", f"GRU: 3[n_u(n_u + {nf}) + 2n_u]"],
            [[n, ribu(p_lstm(n)) + (" ← LSTM terbaik" if n == nl else ""),
              ribu(p_gru(n)) + (" ← GRU terbaik" if n == ng else "")] for n in sorted({ng, nl})])
    d.teks("Pada jumlah neuron yang sama, GRU selalu memerlukan parameter lebih sedikit.")

    # 11 --------------------------------------------------------------------
    d.teks(f"## {JUDUL_1[11]}")
    d.teks("""
1. **Klaim tentang rumus asli Cho et al. (2014) di halaman 17 perlu diperbaiki.**
   Persamaan (7) pada makalah aslinya berbunyi
   hⱼ⟨t⟩ = zⱼ hⱼ⟨t−1⟩ + (1 − zⱼ) h̃ⱼ⟨t⟩, yaitu **sama** dengan persamaan (26) (konvensi
   Keras), sehingga peran zₜ **tidak tertukar**. Bentuk dengan peran zₜ tertukar,
   hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, berasal dari Chung et al. (2014). Perbedaan dengan
   Cho et al. (2014) hanya pada posisi reset gate. Saran kalimat:

   > Rumus GRU pada makalah asli Cho et al. (2014) sedikit berbeda, yaitu gerbang
   > reset diterapkan sebelum perkalian dengan matriks bobot rekuren,
   > h̃ₜ = tanh(xₜW_h + (rₜ ⊙ hₜ₋₁)U_h). Adapun bentuk
   > hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, dengan peran zₜ tertukar, digunakan oleh
   > Chung et al. (2014).

2. **Gambar 7 dan pengaturan `reset_after=True`.** Gambar 7 menggambar rₜ
   mengalikan hₜ₋₁ sebelum kotak tanh (bentuk Cho et al.), sedangkan persamaan (25)
   mengalikan rₜ dengan (hₜ₋₁U_h + b_h(rec)) setelah perkalian matriks. Fungsinya
   sama, jadi gambar tidak perlu diubah. Cukup tambahkan kalimat berikut, atau beri
   label garis dari lingkaran × rₜ ke kotak tanh dengan rₜ ⊙ (hₜ₋₁U_h + b_h(rec)) dan
   garis xₜ ke kotak tanh dengan xₜW_h + b_h(in):

   > Gambar 7 merupakan ilustrasi konseptual; urutan perhitungan yang digunakan
   > mengikuti persamaan (25).

3. **Notasi Cₜ pada Gambar 1-6.** Gambar Olah (2015) memakai huruf besar Cₜ untuk
   cell state, sedangkan persamaan skripsi memakai cₜ. Satu kalimat penjelas dapat
   mencegah pertanyaan penguji.
""")
    penutup(d, 1)


# ------------------------------- Bagian 2 ---------------------------------- #
JUDUL_2 = {
    1: "1. Gambaran Besar: Loss, Gradien, dan Adam",
    2: "2. Kapan Dipakai: Alur Pelatihan di Penelitian",
    3: "3. Fungsi Loss: Mean Squared Error",
    4: "4. Membaca Kurva Loss",
    5: "5. Hubungan MSE dengan RMSE dalam USD",
    6: "6. Adam: Alur Satu Kali Update",
    7: "7. Contoh Angka Adam: 1 Parameter, 2 Iterasi",
    8: "8. Koreksi Bias Seiring Iterasi",
    9: "9. Mengapa Adam Cocok untuk Penelitian Ini",
    10: "10. Ringkasan: Kapan Loss dan Adam Bekerja",
    11: "11. Catatan untuk Naskah Skripsi",
}


def adam_satu_parameter(theta, daftar_gradien):
    """Jalankan Adam pada satu parameter; kembalikan rincian setiap iterasi."""
    m = v = 0.0
    hasil = []
    for k_, g in enumerate(daftar_gradien, 1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        mh, vh = m / (1 - B1 ** k_), v / (1 - B2 ** k_)
        langkah = LR * mh / (math.sqrt(vh) + EPS)
        theta -= langkah
        hasil.append({"g": g, "m": m, "v": v, "mh": mh, "vh": vh, "akar": math.sqrt(vh),
                      "langkah": langkah, "theta": theta})
    return hasil


def seri_2_loss_adam(d, k, lstm):
    kepala(d, 2)
    d.teks(f"""
{tautan(1)} menjelaskan apa yang terjadi di dalam sel saat prediksi dibuat
(*forward pass*). Pada contoh model mini, prediksinya ŷ′ = {a(lstm['maju']['yhat'])}
padahal targetnya {Y}. Bagian ini menjelaskan **bagaimana model belajar dari
kesalahan itu**: fungsi loss mengukur kesalahan, gradien menunjukkan arah perbaikan,
dan Adam menggeser bobot. Perhitungan lengkapnya untuk satu siklus ada di {tautan(3)}.
""")
    daftar_isi(d, JUDUL_2)
    d.teks(CATATAN_ANGKA)

    # 1 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[1]}")
    d.teks("""
Pelatihan model bisa dibayangkan seperti **orang yang menuruni gunung dalam kabut**
untuk mencari titik terendah:
""")
    d.tabel(["Komponen", "Analogi", "Tugasnya"],
            [["**Fungsi loss (MSE)**", "ketinggian posisi saat ini",
              "mengukur **seberapa salah** prediksi; makin kecil makin baik"],
             ["**Gradien gₖ**", "kemiringan tanah di bawah kaki",
              "menunjukkan **arah** perubahan bobot yang menaikkan loss; dihitung dengan "
              "*backpropagation through time* (BPTT)"],
             ["**Adam**", "strategi melangkah",
              "memutuskan **seberapa jauh dan ke mana** setiap bobot digeser agar loss turun"]])
    d.teks("""
Urutannya selalu: **loss dihitung → gradien dihitung dari loss → Adam memakai
gradien untuk memperbarui bobot**. Adam tidak menghitung gradien sendiri; ia hanya
memakai gradien yang sudah ada.
""")

    # 2 ---------------------------------------------------------------------
    total = {e: k["iter_epoch"] * e for e in sorted(k["lstm"]["per_epoch"])}
    d.teks(f"## {JUDUL_2[2]}")
    d.teks(f"""
Di penelitian, data latih berisi **{ribu(k['n_latih'])} sampel** (jendela {k['lookback']} hari ×
{k['n_fitur']} fitur) dengan **batch size {k['batch']}**. Jadi satu epoch terdiri dari
⌈{ribu(k['n_latih'])} / {k['batch']}⌉ = **{k['iter_epoch']} batch**:
{k['iter_epoch'] - 1} batch berisi {k['batch']} sampel dan 1 batch terakhir berisi
{k['batch_akhir']} sampel. Untuk **setiap kombinasi neuron × epoch** pada grid
(misalnya LSTM {k['lstm']['neuron']} neuron, {ribu(k['lstm']['epoch'])} epoch):

**Langkah 0 — persiapan model.**

- Bobot diisi acak (Glorot uniform untuk W, ortogonal untuk U); bias = 0 kecuali
  bias forget LSTM = 1 ({tautan(4)}).
- Memori Adam di-nol-kan: m₀ = 0, v₀ = 0, penghitung k = 0.

**Langkah 1-5 — diulang untuk setiap batch**, berurutan secara kronologis karena
`shuffle=False`:

1. **Forward pass.** {k['batch']} jendela masuk ke LSTM/GRU lalu dense, menghasilkan
   {k['batch']} prediksi ŷ′ (alur {tautan(1)}).
2. **Hitung loss.** MSE dari {k['batch']} prediksi itu terhadap nilai aktual y′
   (subbagian 3).
3. **Hitung gradien.** BPTT menghasilkan gₖ untuk **setiap parameter**
   ({ribu(k['lstm']['param'])} parameter pada LSTM-{k['lstm']['neuron']},
   {ribu(k['gru']['param'])} pada GRU-{k['gru']['neuron']}).
4. **Update Adam.** Setiap parameter digeser memakai persamaan (29)-(31) (subbagian 6).
5. k bertambah 1, lalu lanjut ke batch berikutnya.

**Langkah 6 — akhir setiap epoch.**

- Keras mencatat **loss latih**, yaitu rata-rata loss dari {k['iter_epoch']} batch
  selama epoch itu.
- Keras menghitung **loss validasi** pada {k['n_val']} sampel validasi memakai bobot
  akhir epoch. Ini **hanya diukur; tidak ada update bobot**.
- Kedua angka inilah yang digambar sebagai **kurva loss** (subbagian 4).

**Langkah 7 — setelah semua epoch selesai.**

- Bobot pada **epoch terakhir** yang dipakai (penelitian tidak memakai *early stopping*).
- Prediksi data validasi didenormalisasi ke USD, lalu **RMSE validasi** dipakai untuk
  memilih kombinasi neuron × epoch terbaik: LSTM {k['lstm']['neuron']} neuron
  {ribu(k['lstm']['epoch'])} epoch dan GRU {k['gru']['neuron']} neuron
  {ribu(k['gru']['epoch'])} epoch.
""")
    d.teks("**Jumlah update Adam per model** ({} per epoch):".format(k["iter_epoch"]))
    d.tabel(["Epoch", "Jumlah update"], [[ribu(e), f"{k['iter_epoch']} × {ribu(e)} = {ribu(t)}"]
                                         for e, t in total.items()])
    d.teks("""
**Kapan loss dan Adam TIDAK dipakai:**

- saat **memilih model terbaik** (dipakai RMSE validasi dalam USD);
- saat **memprediksi data uji** (hanya forward pass, bobot sudah beku);
- saat **evaluasi akhir** (RMSE, MAE, MAPE, akurasi arah dalam USD) dan **uji
  Diebold-Mariano**.

Singkatnya, **loss dan Adam hanya bekerja di fase pelatihan**.
""")

    # 3 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[3]}")
    d.teks(r"""
Persamaan (28):

$$\mathcal{L} = \frac{1}{N}\sum_{k=1}^{N}\left(y'_k - \hat{y}'_k\right)^2$$

**Cara kerjanya**, dengan contoh 3 sampel:
""")
    aktual, prediksi = [0.50, 0.52, 0.55], [0.48, 0.53, 0.51]
    selisih = [y_ - p_ for y_, p_ in zip(aktual, prediksi)]
    mse = sum(s_ ** 2 for s_ in selisih) / len(selisih)
    d.tabel(["Sampel", "Aktual y′", "Prediksi ŷ′", "Selisih", "Kuadrat"],
            [[i + 1, f"{y_:.2f}", f"{p_:.2f}", f"{s_:.2f}", a(s_ ** 2, 4)]
             for i, (y_, p_, s_) in enumerate(zip(aktual, prediksi, selisih))])
    d.kode([f"MSE = ({' + '.join(a(s_ ** 2, 4) for s_ in selisih)}) / 3"
            f" = {a(sum(s_ ** 2 for s_ in selisih), 4)} / 3 = {a(mse, 4)}"])
    besar, kecil = max(abs(s_) for s_ in selisih), min(abs(s_) for s_ in selisih)
    contoh_usd = 2500.0
    norm_usd = contoh_usd / k["rentang"]
    d.teks(f"""
**Mengapa dikuadratkan?**

1. Selisih positif dan negatif tidak saling meniadakan.
2. Kesalahan besar dihukum jauh lebih berat: selisih {besar:.2f} menyumbang
   {round((besar / kecil) ** 2)} kali lebih besar daripada selisih {kecil:.2f}, sehingga
   model "dipaksa" menghindari meleset jauh.
3. Fungsi kuadrat **mulus dan bisa diturunkan di semua titik**. Turunannya,
   ∂L/∂ŷ′ = −2(y′ − ŷ′), menjadi titik awal BPTT ({tautan(3)}, subbagian 3.4). MAE
   (nilai mutlak) tidak mulus di titik nol.

**Mengapa dihitung pada skala ternormalisasi, bukan USD?** Harga Bitcoin bernilai
puluhan ribu USD. Selisih {usd(contoh_usd)} USD jika dikuadratkan menjadi
{usd(contoh_usd ** 2)}, sehingga gradien sangat besar dan pelatihan tidak stabil. Pada
skala 0-1 selisih yang sama hanya {a(norm_usd)} dan kuadratnya {a(norm_usd ** 2)}.

**Nilai N dalam praktik:**

- saat pelatihan, N = {k['batch']} (ukuran batch; batch terakhir N = {k['batch_akhir']});
- loss validasi dihitung pada N = {k['n_val']} sampel validasi;
- pada model mini di {tautan(3)}, N = 1, sehingga
  L = ({Y} − {a(lstm['maju']['yhat'])})² = {a(lstm['maju']['L'])}.
""")

    # 4 ---------------------------------------------------------------------
    kl, kg = k["kurva"]["lstm"], k["kurva"]["gru"]
    d.teks(f"## {JUDUL_2[4]}")
    d.teks(f"""
Kurva loss memperlihatkan loss latih dan loss validasi di akhir setiap epoch
(langkah 6 pada subbagian 2). Berikut kurva model terbaik dari notebook:

![Kurva loss LSTM terbaik]({k['gambar_loss_lstm']})

![Kurva loss GRU terbaik]({k['gambar_loss_gru']})
""")
    nama_l = f"LSTM ({k['lstm']['neuron']} neuron, {ribu(k['lstm']['epoch'])} epoch)"
    nama_g = f"GRU ({k['gru']['neuron']} neuron, {ribu(k['gru']['epoch'])} epoch)"
    d.tabel(["Besaran", nama_l, nama_g],
            [["Loss latih epoch 1", f"{kl['latih_awal']:.8f}", f"{kg['latih_awal']:.8f}"],
             ["Loss latih epoch terakhir", f"{kl['latih_akhir']:.8f}", f"{kg['latih_akhir']:.8f}"],
             ["Penurunan loss latih", persen(1 - kl["latih_akhir"] / kl["latih_awal"]),
              persen(1 - kg["latih_akhir"] / kg["latih_awal"])],
             ["Loss validasi epoch 1", f"{kl['val_awal']:.8f}", f"{kg['val_awal']:.8f}"],
             ["Loss validasi epoch terakhir", f"{kl['val_akhir']:.8f}", f"{kg['val_akhir']:.8f}"],
             ["Loss validasi minimum", f"{kl['val_min']:.8f} (epoch {kl['epoch_min']})",
              f"{kg['val_min']:.8f} (epoch {kg['epoch_min']})"]])
    d.teks(f"""
**Cara membacanya:**

- Jika **loss latih dan loss validasi turun bersama**, model sedang mempelajari pola
  yang benar. Itu terlihat pada kedua model di awal pelatihan.
- Jika **loss latih terus turun tetapi loss validasi naik**, itu tanda *overfitting*:
  model mulai menghafal data latih.
- Loss validasi LSTM mencapai minimum di epoch {kl['epoch_min']}, sangat dekat dengan
  epoch terakhir. Loss validasi GRU mencapai minimum lebih awal, di epoch
  {kg['epoch_min']}, lalu sedikit naik sampai epoch terakhir.

**Mengapa jumlah epoch dipilih lewat data validasi?** RMSE validasi pada neuron terbaik
untuk setiap jumlah epoch (Tabel 9 dan Tabel 11):
""")
    epochs = sorted(set(k["lstm"]["per_epoch"]) | set(k["gru"]["per_epoch"]))
    d.tabel(["Epoch", f"LSTM {k['lstm']['neuron']} neuron (USD)", f"GRU {k['gru']['neuron']} neuron (USD)"],
            [[ribu(e),
              usd(k["lstm"]["per_epoch"][e]) + (" ← terbaik" if e == k["lstm"]["epoch"] else ""),
              usd(k["gru"]["per_epoch"][e]) + (" ← terbaik" if e == k["gru"]["epoch"] else "")]
             for e in epochs])
    e_maks = max(epochs)
    for m in ("lstm", "gru"):   # teks di bawah menyatakan epoch maksimum lebih buruk
        assert k[m]["per_epoch"][e_maks] > k[m]["per_epoch"][k[m]["epoch"]]
    d.teks(f"""
Pada neuron terbaik, melatih sampai {ribu(e_maks)} epoch justru memperburuk RMSE
validasi dibandingkan {ribu(k['lstm']['epoch'])} epoch. Pelatihan yang lebih lama tidak
selalu lebih baik; kemungkinan besar model mulai terlalu menyesuaikan diri dengan data
latih. Itulah gunanya data validasi untuk memilih jumlah epoch.
""")

    # 5 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[5]}")
    rmse_l = k["rentang"] * math.sqrt(k["val_loss_lstm"])
    rmse_g = k["rentang"] * math.sqrt(k["val_loss_gru"])
    assert abs(rmse_l / k["lstm"]["rmse_val"] - 1) < 1e-3
    assert abs(rmse_g / k["gru"]["rmse_val"] - 1) < 1e-3
    d.teks(f"""
Karena normalisasi min-max bersifat linear (persamaan 6 dan 8), selisih dalam USD
sama dengan selisih ternormalisasi dikali (x_max − x_min). Akibatnya:

**RMSE (USD) = (x_max − x_min) × √MSE**

Dengan rentang harga data latih {usd(k['x_max'])} − {usd(k['x_min'])} = {usd(k['rentang'])}
dan loss validasi epoch terakhir dari notebook:
""")
    d.tabel(["Model", "Loss validasi (MSE)", "Rentang × √MSE", "RMSE validasi di tabel tuning"],
            [["LSTM", f"{k['val_loss_lstm']:.8f}",
              f"{usd(k['rentang'])} × {math.sqrt(k['val_loss_lstm']):.6f} = {usd(rmse_l)} USD",
              f"{usd(k['lstm']['rmse_val'])} USD"],
             ["GRU", f"{k['val_loss_gru']:.8f}",
              f"{usd(k['rentang'])} × {math.sqrt(k['val_loss_gru']):.6f} = {usd(rmse_g)} USD",
              f"{usd(k['gru']['rmse_val'])} USD"]])
    d.teks("""
Jadi **meminimalkan MSE saat pelatihan sama artinya dengan meminimalkan RMSE dalam
USD** (selisih kecil hanya karena loss validasi dicetak 8 desimal). Kalimat ini
berguna untuk menjawab pertanyaan "mengapa loss-nya MSE tetapi evaluasinya RMSE?".
""")

    # 6 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[6]}")
    d.teks(r"""
Adam dijalankan **untuk setiap parameter secara terpisah**. Setiap bobot dan bias
punya m dan v miliknya sendiri, sehingga setiap parameter punya "kecepatan belajar"
sendiri. Satu kali update terdiri dari empat langkah:

**Langkah A — momen pertama (persamaan 29, kiri)**

$$m_k = \beta_1\, m_{k-1} + (1-\beta_1)\, g_k \qquad (\beta_1 = 0.9)$$

Isinya adalah **rata-rata bergerak dari arah gradien**, kira-kira merangkum ±10
gradien terakhir karena 1/(1 − 0.9) = 10. Fungsinya seperti **momentum bola yang
menggelinding**: gradien dari satu batch bisa "berisik", dan dengan dirata-rata, arah
langkah menjadi lebih stabil.

**Langkah B — momen kedua (persamaan 29, kanan)**

$$v_k = \beta_2\, v_{k-1} + (1-\beta_2)\, g_k^2 \qquad (\beta_2 = 0.999)$$

Isinya adalah **rata-rata bergerak dari besarnya gradien (dikuadratkan)**, kira-kira
merangkum ±1,000 gradien terakhir. Fungsinya **mengukur seberapa besar atau
bergejolak gradien parameter itu**; nilainya dipakai sebagai pembagi di langkah D.

**Langkah C — koreksi bias (persamaan 30)**

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad \hat{v}_k = \frac{v_k}{1-\beta_2^k}$$

m dan v dimulai dari **nol**, sehingga di awal pelatihan nilainya "tertarik" ke nol.
Pembagi (1 − βᵏ) mengoreksinya, dan pengaruhnya hilang setelah banyak iterasi
(subbagian 8).

**Langkah D — update parameter (persamaan 31)**

$$\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}
\qquad (\eta = 0.001,\ \epsilon = 10^{-7})$$

- Pembilang m̂ₖ menentukan **arah** langkah.
- Pembagi √v̂ₖ menyesuaikan **ukuran langkah**: parameter yang gradiennya besar atau
  bergejolak mendapat langkah lebih kecil, sedangkan yang gradiennya kecil mendapat
  langkah relatif lebih besar.
- Tanda minus berarti bergerak **berlawanan** arah gradien, yaitu menuruni loss.
- η = 0.001 adalah batas kasar ukuran langkah. Rasio m̂/√v̂ biasanya sekitar ±1, jadi
  setiap parameter bergeser paling jauh sekitar ±0.001 per iterasi.
- ε hanya pengaman agar tidak terjadi pembagian dengan nol.
""")
    d.teks(f"Penerapan keempat langkah ini pada 14 parameter model mini ada di "
           f"{tautan(3)}, subbagian 3.10 sampai 3.12.")

    # 7 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[7]}")
    theta0, gradien = 0.5, [0.2, 0.1]
    h = adam_satu_parameter(theta0, gradien)
    h100 = adam_satu_parameter(theta0, [100 * g for g in gradien])
    assert all(abs(x["langkah"] - y["langkah"]) < 1e-8 for x, y in zip(h, h100))
    d.teks(f"Misalkan satu bobot bernilai awal θ₀ = {theta0}, dengan gradien iterasi 1 "
           f"g₁ = {gradien[0]} dan iterasi 2 g₂ = {gradien[1]}:")
    i1, i2 = h
    d.tabel(["Tahap", f"Iterasi 1 (g = {gradien[0]})", f"Iterasi 2 (g = {gradien[1]})"],
            [["m = 0.9 m_lama + 0.1 g", f"0.9 × 0 + 0.1 × {gradien[0]} = {a(i1['m'])}",
              f"0.9 × {a(i1['m'])} + 0.1 × {gradien[1]} = {a(i2['m'])}"],
             ["v = 0.999 v_lama + 0.001 g²", f"0.001 × {gradien[0]}² = {a(i1['v'], 8)}",
              f"0.999 × {a(i1['v'], 8)} + 0.001 × {gradien[1]}² = {a(i2['v'], 8)}"],
             ["m̂ = m / (1 − 0.9ᵏ)", f"{a(i1['m'])} / 0.1 = {a(i1['mh'])}",
              f"{a(i2['m'])} / 0.19 = {a(i2['mh'])}"],
             ["v̂ = v / (1 − 0.999ᵏ)", f"{a(i1['v'], 8)} / 0.001 = {a(i1['vh'])}",
              f"{a(i2['v'], 8)} / 0.001999 = {a(i2['vh'])}"],
             ["√v̂", a(i1["akar"]), a(i2["akar"])],
             ["Langkah η × m̂ / (√v̂ + ε)", f"0.001 × {a(i1['mh'])} / {a(i1['akar'])} = {a(i1['langkah'], 7)}",
              f"0.001 × {a(i2['mh'])} / {a(i2['akar'])} = {a(i2['langkah'], 7)}"],
             ["θ baru = θ lama − langkah", f"{theta0} − {a(i1['langkah'], 7)} = {a(i1['theta'])}",
              f"{a(i1['theta'])} − {a(i2['langkah'], 7)} = {a(i2['theta'])}"]])
    d.teks(f"""
**Bukti "tidak dipengaruhi penskalaan gradien".** Jika gradiennya **100 kali lebih
besar** (g₁ = {100 * gradien[0]:g}, g₂ = {100 * gradien[1]:g}), langkahnya tetap
**{a(h100[0]['langkah'], 7)} dan {a(h100[1]['langkah'], 7)}**, sama persis. Rasio m̂/√v̂
menghapus skala gradien. Sebagai pembanding, SGD biasa (langkah = η × g) akan melangkah
100 kali lebih jauh.
""")

    # 8 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[8]}")
    d.teks("""
Pembagi koreksi bias makin lama makin mendekati 1, sehingga pengaruhnya hilang.
Koreksi m hanya penting di epoch pertama, sedangkan koreksi v masih berpengaruh
sampai puluhan epoch (angka epoch memakai 26 iterasi per epoch penelitian):
""")
    titik = ((1, "iterasi pertama"), (2, "iterasi kedua"),
             (k["iter_epoch"], "akhir epoch 1"), (1000, f"± epoch {round(1000 / k['iter_epoch'])}"),
             (k["iter_epoch"] * 100, "akhir epoch 100"), (k["iter_epoch"] * 500, "akhir epoch 500"))
    d.tabel(["Iterasi k", "Keterangan", "1 − 0.9ᵏ (pembagi m)", "1 − 0.999ᵏ (pembagi v)"],
            [[ribu(kk), ket, a(1 - B1 ** kk), a(1 - B2 ** kk)] for kk, ket in titik])

    # 9 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[9]}")
    d.teks(f"""
- Adam menggabungkan dua ide: **momentum** (dari m) dan **langkah adaptif per
  parameter** (dari v, ide RMSProp/AdaGrad; Kingma & Ba, 2015).
- Gradien dari batch data kripto yang fluktuatif cenderung berisik; momentum
  meredamnya.
- Parameter LSTM/GRU sangat beragam (bobot gerbang, bobot kandidat, bias, dense).
  Langkah adaptif membuat semuanya bisa belajar dengan kecepatan wajar tanpa harus
  mengatur learning rate satu per satu. Di {tautan(3)} terlihat gradien terbesar dan
  terkecil berbeda ratusan sampai ribuan kali, tetapi semua parameter tetap bergeser
  ±0.001 pada iterasi pertama.
- Pengaturan Adam **identik** untuk LSTM dan GRU (η = {k['lr']:g}, β₁ = {B1}, β₂ = {B2},
  ε = 10⁻⁷, batch {k['batch']}, seed sama), sehingga perbedaan hasil hanya berasal dari
  arsitektur.
""")

    # 10 --------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[10]}")
    d.tabel(["Tahap penelitian", "Loss MSE", "Adam"],
            [["Inisialisasi model", "-", "m = 0, v = 0, k = 0"],
             [f"Setiap batch latih ({k['iter_epoch']}× per epoch)", "dihitung, menjadi sumber gradien",
              "memperbarui semua bobot dan bias"],
             ["Akhir setiap epoch", "loss latih dan loss validasi dicatat (kurva loss)",
              "tidak ada update dari data validasi"],
             ["Pemilihan neuron dan epoch terbaik", "tidak (memakai RMSE validasi USD, setara √MSE × rentang)",
              "tidak"],
             ["Prediksi data uji dan evaluasi", "tidak", "tidak (bobot sudah beku)"]])

    # 11 --------------------------------------------------------------------
    d.teks(f"## {JUDUL_2[11]}")
    d.teks(f"""
1. **Keterangan N pada persamaan (28).** Saat pelatihan, loss dihitung per
   *mini-batch*, jadi N = {k['batch']} (batch terakhir {k['batch_akhir']}), bukan seluruh
   sampel. Saran kalimat:

   > Saat pelatihan, ℒ dihitung pada setiap mini-batch berukuran N = {k['batch']},
   > sedangkan loss yang dilaporkan per epoch merupakan rata-rata loss seluruh
   > mini-batch.

2. **Arti iterasi ke-k pada Adam.** Satu iterasi adalah satu kali update per batch,
   bukan per epoch. Saran kalimat:

   > Satu iterasi k bersesuaian dengan satu mini-batch, sehingga dengan
   > {ribu(k['n_latih'])} sampel latih dan batch size {k['batch']} terdapat
   > {k['iter_epoch']} iterasi per epoch.

3. **Opsional, hubungan loss dan evaluasi** (subbab 1.5.11 atau 1.5.12):

   > Karena normalisasi min-max bersifat linear, RMSE dalam USD sama dengan
   > (x_max − x_min) × √MSE, sehingga meminimalkan MSE pada skala ternormalisasi
   > setara dengan meminimalkan RMSE dalam USD.
""")
    penutup(d, 2)


def bagian_5(d, k, lstm, gru):
    d.teks(f"## {JUDUL[5]}")
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
pengaturan Adam. Setiap parameter (termasuk setiap elemen matriks) punya gradien,
m, dan v sendiri.
""")

    s_lstm = lstm["maju"]["langkah"][-1]["h"]
    y_lstm = lstm["maju"]["yhat"]
    h2b, y2b = 0.30, 0.65
    yh2 = LSTM_AWAL["W_y"] * h2b + LSTM_AWAL["b_y"]
    d1, d2 = -(2 / 2) * (Y - y_lstm), -(2 / 2) * (y2b - yh2)
    d.teks("### 5.1 Jika Memakai Batch (N > 1)")
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
        f"  ∂L/∂W_y = {kr(d1)} × {a(s_lstm)} + {kr(d2)} × {h2b} = {kr(d1 * s_lstm)} + {kr(d2 * h2b)}"
        f" = {a(d1 * s_lstm + d2 * h2b)}",
        f"  ∂L/∂b_y = {kr(d1)} + {kr(d2)} = {a(d1 + d2)}",
    ])

    d.teks("### 5.2 Versi Vektor pada Lapisan Dense")
    d.teks(f"""
Pada model penelitian, h_T berisi n_u elemen dan W_y juga berisi n_u elemen,
sehingga ŷ′ = Σⱼ h_T,ⱼ·W_y,ⱼ + b_y. Rumus 3.4 berlaku untuk setiap elemen j:
∂L/∂W_y,ⱼ = Σₖ ∂L/∂ŷ′ₖ·h_T,ₖ,ⱼ. Hasilnya n_u gradien bobot + 1 gradien bias,
yaitu **{nl + 1} parameter dense pada LSTM** dan **{ng + 1} pada GRU** (suku
(n_u + 1) pada persamaan 22 dan 27). Sinyal yang masuk ke neuron j adalah
∂L/∂h_T,ⱼ = ∂L/∂ŷ′·W_y,ⱼ.

Hubungan loss MSE dengan RMSE dalam USD dibahas di {tautan(2, JUDUL_2[5])}.
""")

    d.teks("### 5.3 Batasan Simulasi Ini")
    d.teks(f"""
1. Loss simulasi turun hampir ke nol karena hanya ada **1 sampel**, sehingga model
   bisa "menghafal" targetnya. Pada data penelitian dengan {ribu(k['n_latih'])}
   sampel, loss validasi LSTM terbaik berhenti di sekitar {k['val_loss_lstm']:.5f}
   karena model harus menemukan pola umum, bukan menghafal.
2. Angka akhir simulasi **tidak dapat dipakai untuk membandingkan** LSTM dan GRU.
   Bobot awalnya dipilih bulat agar mudah dihitung, sedangkan Keras memakai bobot
   acak. Perbandingan yang sah tetap hasil tahap 9-12.
""")


def bagian_6(d, k, lstm, gru, cek):
    d.teks(f"## {JUDUL[6]}")
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
    d.teks("**Pemeriksaan otomatis yang lolos saat seri ini dibuat:**")
    d.teks("\n".join(f"- {c}" for c in cek))
    penutup(d, 3)


# ------------------------------- Bagian 4 ---------------------------------- #
JUDUL_4 = {
    1: "1. Intinya",
    2: "2. Mengapa Bias Diperlukan",
    3: "3. Cara Menghitung Bias Langkah demi Langkah",
    4: "4. Dua Bias pada GRU: Asal dan Buktinya",
    5: "5. Catatan untuk Naskah Skripsi",
}


def seri_4_bias(d, k, lstm, gru):
    gl, gg = lstm["mundur"]["g"], gru["mundur"]["g"]
    rl = lstm["mundur"]["rincian"]
    kepala(d, 4)
    d.teks(f"""
Bagian ini mengumpulkan semua pembahasan tentang **bias**: mengapa bias perlu ada di
LSTM dan GRU, bagaimana bias dihitung langkah demi langkah, dan mengapa GRU memiliki
dua jenis bias. Angka contohnya diambil dari simulasi di {tautan(3)} dan dari model
terlatih penelitian.
""")
    daftar_isi(d, JUDUL_4)
    d.teks(CATATAN_ANGKA)

    # 1 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_4[1]}")
    d.teks("""
Bias adalah **titik awal (intercept)** setiap gerbang dan neuron, sama seperti
intercept *a* pada regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa
bias, ketika masukannya nol, setiap gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu
setengah terbuka; tanh(0) = 0). Dengan bias, setiap gerbang dapat menentukan **posisi
bawaannya sendiri**.

Bias tidak dihitung dengan satu rumus langsung. Bias **dipelajari** lewat siklus yang
sama dengan bobot: forward → loss → BPTT → Adam (subbagian 3).
""")

    # 2 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_4[2]}")
    d.teks("""
**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Ini sering
terjadi di penelitian: h₀ = 0 di awal setiap jendela 7 hari, dan normalisasi min-max
membuat fitur bernilai dekat 0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman kurva, bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:
""")
    d.tabel(["x", "Dengan bias: σ(5x − 2.5)", "Tanpa bias: σ(5x)"],
            [[f"{x:g}", a(sig(5 * x - 2.5), 3), a(sig(5 * x), 3)] for x in (0, 0.25, 0.5, 0.75, 1)])
    d.teks("""
Tanpa bias, gerbang **tidak pernah turun di bawah 0.5** untuk data ternormalisasi
yang positif, sehingga model tidak bisa menyatakan aturan "tutup gerbang saat nilai
fitur rendah, buka saat tinggi".

**(c) Bias menentukan perilaku bawaan setiap gerbang.** Contoh terpenting: forget gate
LSTM. Jika gerbang hanya ditentukan oleh biasnya, sisa memori setelah 7 hari adalah f⁷:
""")
    bf = k["bias_lstm"]["f"]
    d.tabel(["Bias forget", "f = σ(b)", "Memori tersisa setelah 7 hari (f⁷)"],
            [[lbl, a(sig(b), 3), persen(sig(b) ** 7)]
             for lbl, b in (("0 (tanpa bias)", 0.0), (f"{a(bf)} (neuron ke-1 LSTM terlatih)", bf),
                            ("1 (inisialisasi Keras)", 1.0), ("2", 2.0))])
    d.teks("""
Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*).

Nilai bawaan gerbang pada **model terlatih penelitian** (neuron ke-1, saat kontribusi
xW + hU = 0), dibaca dari `tahap_7_forward_pass_lstm.md` dan
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
Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari sesuai
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di rentang
(−1, 1), jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y cukup menangani
variasinya. Pada model terlatih, b_y LSTM = {a(k['by_lstm'])} (setara
{a(k['by_lstm'])} × {usd(k['rentang'])} ≈ **{usd(k['by_lstm'] * k['rentang'])} USD**) dan
b_y GRU = {a(k['by_gru'])} (≈ **{usd(k['by_gru'] * k['rentang'])} USD**). Tanpa b_y,
prediksi dipaksa jatuh ke harga terendah data latih ({usd(k['x_min'])} USD) setiap kali
h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁
bernilai nol bila masukannya nol (lihat ∂L/∂U di {tautan(3)}, subbagian 3.8: pada t = 1
tidak ada kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.

**(f) Biayanya kecil.** Pada model penelitian, bias hanya
{4 * k['lstm']['neuron'] + 1} dari {ribu(k['lstm']['param'])} parameter LSTM
({persen((4 * k['lstm']['neuron'] + 1) / k['lstm']['param'])}) dan
{6 * k['gru']['neuron'] + 1} dari {ribu(k['gru']['param'])} parameter GRU
({persen((6 * k['gru']['neuron'] + 1) / k['gru']['param'])}).
""")

    # 3 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_4[3]}")
    d.teks(f"""
Bias diperlakukan **persis seperti bobot**. Berikut lima langkahnya, dengan rujukan ke
perhitungan di {tautan(3)}:
""")
    lat = lstm["latih"]
    d.kode([
        "Langkah 1 — Nilai awal (Bagian 3: 3.1 dan 4.1)",
        "  Bias gerbang = 0, kecuali bias forget gate LSTM = 1. Bias dense b_y = 0.",
        "",
        "Langkah 2 — Dipakai di forward pass (Bagian 3: 3.2 dan 4.2)",
        "  Gerbang : a = W × x + U × h + b",
        "  Dense   : ŷ' = W_y × h_T + b_y",
        "",
        "Langkah 3 — Hitung gradiennya (Bagian 3: 3.8 dan 4.8)",
        "  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga",
        "  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ",
        f"  Contoh b_y : ∂L/∂b_y = ∂L/∂ŷ' × 1 = {a(gl['b_y'])}",
        f"  Contoh b_c : ∂L/∂b_c = δc̃₁ + δc̃₂ = {kr(rl[0]['delta_c'])} + {kr(rl[1]['delta_c'])} = {a(gl['b_c'])}",
        "",
        "Langkah 4 — Update dengan Adam (Bagian 3: 3.10 dan 3.12)",
        f"  b_y: 0 → {a(lat['p'][1]['b_y'])} (k = 1) → {a(lat['p'][2]['b_y'])} (k = 2)",
        "",
        "Langkah 5 — Ulangi siklus (Bagian 3: 3.13)",
        f"  Setelah {ITERASI} iterasi b_y = {a(lat['p'][-1]['b_y'])}. Pada model penelitian, b_y LSTM = {a(k['by_lstm'])}",
        f"  adalah hasil akhir proses yang sama setelah {ribu(k['iter_epoch'] * k['lstm']['epoch'])} iterasi.",
    ])
    d.teks("Ringkasan rumus gradien seluruh bias pada simulasi (iterasi k = 1):")
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

    # 4 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_4[4]}")
    d.teks(f"""
**Letaknya pada Gambar 7** ({tautan(1, JUDUL_1[9])}). Bias tidak digambar; bias berada
di dalam setiap kotak kuning (σ, σ, tanh). Setiap kotak menerima dua garis masuk,
yaitu xₜ dari bawah dan hₜ₋₁ dari garis vertikal kiri. Dengan `reset_after=True`, Keras
menghitung kedua garis itu terpisah, masing-masing dengan biasnya sendiri: garis xₜ
membawa **bias masukan** (xₜW + b(in)) dan garis hₜ₋₁ membawa **bias rekuren**
(hₜ₋₁U + b(rec)).
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
    akhir = gru["latih"]["p"][-1]
    d.teks(f"""
**Pada z dan r, dua bias sebenarnya berlebih.** b(in) + b(rec) langsung dijumlahkan,
jadi gradien keduanya selalu sama. Karena nilai awalnya juga sama (0), keduanya akan
selalu kembar:

- simulasi: setelah {ITERASI} iterasi, b_z(in) = b_z(rec) = {a(akhir['b_z_in'])} dan
  b_r(in) = b_r(rec) = {a(akhir['b_r_in'])};
- model GRU terlatih penelitian (neuron ke-1): b_z(in) = {a(bg['z'][0])},
  b_z(rec) = {a(bg['z'][1])}; b_r(in) = {a(bg['r'][0])}, b_r(rec) = {a(bg['r'][1])}.

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikalikan rₜ (jika rₜ
mendekati 0, bias ini ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu
aktif. Gradiennya berbeda (Σ δh̃·r vs Σ δh̃), sehingga nilainya juga berbeda:

- simulasi: gradien {a(gg['b_h_in'])} vs {a(gg['b_h_rec'])}; setelah {ITERASI} iterasi
  b_h(in) = {a(akhir['b_h_in'])} dan b_h(rec) = {a(akhir['b_h_rec'])};
- model terlatih (neuron ke-1): b_h(in) = {a(bg['h'][0])} dan b_h(rec) = {a(bg['h'][1])}.

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak memuat bias sama sekali
(penulisnya menyebut *"to make the equations uncluttered, we omit biases"*). Bentuk
dua bias berasal dari implementasi GRU pada pustaka **NVIDIA cuDNN**, yang memisahkan
bagian masukan dan bagian rekuren setiap gerbang. Keras memakainya sebagai bawaan
(`reset_after=True`, *"cuDNN compatible"*). LSTM di Keras cukup memakai satu bias
karena pada LSTM semua bias hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU {k['gru']['neuron']} neuron memiliki
2 × 3 × {k['gru']['neuron']} = {6 * k['gru']['neuron']} bias gerbang (Tabel 12: bias
berbentuk (2, {3 * k['gru']['neuron']})). Dengan satu bias jumlahnya hanya
{3 * k['gru']['neuron']}, sehingga total parameter menjadi
{ribu(k['gru']['param'] - 3 * k['gru']['neuron'])}, bukan {ribu(k['gru']['param'])}.
""")
    assert akhir["b_z_in"] == akhir["b_z_rec"] and akhir["b_r_in"] == akhir["b_r_rec"]
    assert abs(akhir["b_h_in"] - akhir["b_h_rec"]) > 1e-3
    assert bg["z"][0] == bg["z"][1] and bg["r"][0] == bg["r"][1]

    # 5 ---------------------------------------------------------------------
    d.teks(f"## {JUDUL_4[5]}")
    d.teks("""
1. **Kalimat ringkas tentang fungsi bias** (misalnya setelah persamaan 9 di subbab 1.5.7):

   > Bias berfungsi menggeser fungsi aktivasi sehingga setiap gerbang dan neuron
   > dapat menentukan kondisi bawaannya sendiri dan tidak dipaksa bernilai tetap
   > ketika masukannya bernilai nol, misalnya σ(0) = 0,5 atau tanh(0) = 0. Pada LSTM
   > dan GRU, bias menentukan ambang dan kondisi bawaan setiap gerbang, sedangkan pada
   > lapisan dense bias menentukan tingkat dasar nilai prediksi.

2. **Penjelasan dua bias GRU** (setelah Gambar 7 atau persamaan 26):

   > Pada Gambar 7, bias tidak digambarkan secara eksplisit karena termuat di dalam
   > setiap lapisan (kotak σ dan tanh). Pada implementasi Keras dengan pengaturan
   > reset_after=True, setiap lapisan memisahkan kontribusi masukan xₜW + b⁽ⁱⁿ⁾ dan
   > kontribusi rekuren hₜ₋₁U + b⁽ʳᵉᶜ⁾, masing-masing dengan biasnya sendiri. Pada
   > update gate dan reset gate kedua bias hanya dijumlahkan, sedangkan pada kandidat
   > hidden state bias rekuren b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga
   > keduanya tidak dapat digabung menjadi satu. Pemisahan ini mengikuti implementasi
   > GRU pada pustaka cuDNN yang digunakan Keras (Chollet et al., 2015), sedangkan
   > Cho et al. (2014) sendiri tidak menuliskan suku bias pada persamaannya.
""")
    penutup(d, 4)


# --------------------------------------------------------------------------- #
def tulis(nomor, dokumen):
    path = os.path.join(DIR_MANUAL, SERI[nomor - 1][0])
    with open(path, "w", encoding="utf-8") as f:
        f.write(dokumen.isi())
    print(f"[tersimpan] {os.path.relpath(path, AKAR)}")


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
    lstm, gru = hasil["LSTM"], hasil["GRU"]

    cek = [
        f"Gradien BPTT manual = gradien numerik untuk 14 parameter LSTM "
        f"(selisih maksimum {lstm['selisih']:.1e}) dan 14 parameter GRU "
        f"(selisih maksimum {gru['selisih']:.1e}).",
        "Uji geser pada lapisan dense: perubahan loss sebenarnya = gradien × 0.001 "
        "(toleransi 2e-6).",
        "Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter, dan "
        "langkah Adam tidak berubah bila gradien dikali 100 (Bagian 2).",
        "Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah "
        f"{ITERASI} iterasi, untuk LSTM maupun GRU.",
        "Bias z dan r GRU tetap kembar (bias masukan = bias rekuren), bias kandidat "
        "berbeda, baik pada simulasi maupun model terlatih (Bagian 4).",
        "Rentang harga × √(loss validasi) = RMSE validasi pada tabel tuning "
        "(toleransi 0.1%) untuk LSTM dan GRU (Bagian 2).",
        "Jumlah parameter lapisan rekuren dari rumus (persamaan 22 dan 27) sama dengan "
        "tabel tuning (Bagian 1).",
    ]

    d1 = Dokumen()
    seri_1_alur_sel(d1, k, lstm, gru)
    d2 = Dokumen()
    seri_2_loss_adam(d2, k, lstm)
    d3 = Dokumen()
    bagian_pembuka(d3, k)
    bagian_1(d3, k)
    bagian_2(d3, k)
    bagian_3(d3, k, LSTM_AWAL, lstm["maju"], lstm["mundur"], lstm["num"], lstm["latih"])
    bagian_4(d3, k, GRU_AWAL, gru["maju"], gru["mundur"], gru["num"], gru["latih"])
    bagian_5(d3, k, lstm, gru)
    bagian_6(d3, k, lstm, gru, cek)
    d4 = Dokumen()
    seri_4_bias(d4, k, lstm, gru)

    for nomor, dokumen in enumerate((d1, d2, d3, d4), 1):
        tulis(nomor, dokumen)
    for c in cek:
        print(f"  OK {c}")


if __name__ == "__main__":
    main()
