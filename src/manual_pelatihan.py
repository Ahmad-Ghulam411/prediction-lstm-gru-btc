# -*- coding: utf-8 -*-
"""
Perhitungan manual pelatihan LSTM dan GRU dengan model mini.

Modul ini dipakai notebook pada Tahap 7 (LSTM), Tahap 8 (GRU), dan Tahap 9
(perbandingan). Isinya menghitung satu siklus pelatihan langkah demi langkah
pada model mini (1 neuron, 1 fitur, 2 time step, 1 sampel) dengan rumus yang
sama dengan Keras: forward pass, loss MSE, backpropagation through time (BPTT),
gradien setiap bobot dan bias, lalu update Adam. Hasilnya berupa teks Markdown
yang ditampilkan di notebook dan disimpan ke ``outputs/perhitungan_manual/``.

Modul hanya memakai pustaka standar Python. Setiap klaim dalam teks yang
bergantung pada angka diperiksa dengan ``assert`` atau ditulis bersyarat.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# Pengaturan simulasi model mini
# --------------------------------------------------------------------------- #
X = [0.5, 0.6]   # x_1, x_2 : fitur ternormalisasi hari 1 dan hari 2
Y = 0.7          # y'      : target ternormalisasi hari 3
LR, B1, B2, EPS = 0.001, 0.9, 0.999, 1e-7   # Adam bawaan Keras
ITERASI = 500
TITIK_RIWAYAT = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500]

# Bobot awal dibuat bulat agar mudah dihitung. Bias mengikuti aturan Keras:
# bias forget gate LSTM = 1 (unit_forget_bias), bias lainnya = 0.
AWAL = {
    "LSTM": {
        "W_f": 0.5, "U_f": 0.4, "b_f": 1.0,
        "W_i": 0.6, "U_i": 0.3, "b_i": 0.0,
        "W_c": 0.7, "U_c": 0.2, "b_c": 0.0,
        "W_o": 0.4, "U_o": 0.5, "b_o": 0.0,
        "W_y": 0.8, "b_y": 0.0,
    },
    "GRU": {
        "W_z": 0.5, "U_z": 0.4, "b_z_in": 0.0, "b_z_rec": 0.0,
        "W_r": 0.6, "U_r": 0.3, "b_r_in": 0.0, "b_r_rec": 0.0,
        "W_h": 0.7, "U_h": 0.2, "b_h_in": 0.0, "b_h_rec": 0.0,
        "W_y": 0.8, "b_y": 0.0,
    },
}

# --------------------------------------------------------------------------- #
# Daftar berkas: (nomor subbagian notebook, slug nama berkas, judul)
# Nomor "07_01" berarti Tahap 7, subbagian 7.1 di notebook.
# --------------------------------------------------------------------------- #
BERKAS = {
    ("LSTM", "alur_sel"): ("07_01", "alur_sel_lstm", "Alur Sel LSTM (Gambar 1-6) dengan Model Mini"),
    ("LSTM", "forward_loss"): ("07_02", "simulasi_forward_pass_dan_loss_lstm",
                               "Simulasi Pelatihan LSTM, Langkah 1-2: Forward Pass dan Loss"),
    ("LSTM", "bptt"): ("07_03", "simulasi_bptt_lstm",
                       "Simulasi Pelatihan LSTM, Langkah 3: Backpropagation Through Time"),
    ("LSTM", "adam"): ("07_04", "simulasi_update_adam_lstm", "Simulasi Pelatihan LSTM, Langkah 4: Update Adam"),
    ("LSTM", "jumlah_parameter"): ("07_08", "jumlah_parameter_lstm", "Jumlah Parameter Model LSTM"),
    ("LSTM", "kurva_loss"): ("07_10", "kurva_loss_lstm", "Membaca Kurva Loss Model LSTM"),
    ("LSTM", "forward_pass"): ("07_11", "forward_pass_lstm", "Forward Pass LSTM (Jendela Pertama Data Uji)"),
    ("LSTM", "bias"): ("07_12", "bias_lstm", "Bias pada LSTM: Peran dan Cara Menghitungnya"),
    ("GRU", "alur_sel"): ("08_01", "alur_sel_gru", "Alur Sel GRU (Gambar 7) dengan Model Mini"),
    ("GRU", "forward_loss"): ("08_02", "simulasi_forward_pass_dan_loss_gru",
                              "Simulasi Pelatihan GRU, Langkah 1-2: Forward Pass dan Loss"),
    ("GRU", "bptt"): ("08_03", "simulasi_bptt_gru",
                      "Simulasi Pelatihan GRU, Langkah 3: Backpropagation Through Time"),
    ("GRU", "adam"): ("08_04", "simulasi_update_adam_gru", "Simulasi Pelatihan GRU, Langkah 4: Update Adam"),
    ("GRU", "jumlah_parameter"): ("08_08", "jumlah_parameter_gru", "Jumlah Parameter Model GRU"),
    ("GRU", "kurva_loss"): ("08_10", "kurva_loss_gru", "Membaca Kurva Loss Model GRU"),
    ("GRU", "forward_pass"): ("08_11", "forward_pass_gru", "Forward Pass GRU (Jendela Pertama Data Uji)"),
    ("GRU", "bias"): ("08_12", "bias_gru", "Bias pada GRU: Peran, Cara Menghitung, dan Dua Jenis Bias"),
    ("LSTM-GRU", "perbandingan"): ("09_02", "perbandingan_struktur_lstm_gru",
                                   "Perbandingan Struktur LSTM dan GRU"),
}


def berkas(model: str, bagian: str):
    """Kembalikan ``(nomor, slug, judul)`` untuk ``utils.tulis_manual``."""
    return BERKAS[(model, bagian)]


def nama_berkas(model: str, bagian: str) -> str:
    nomor, slug, _ = BERKAS[(model, bagian)]
    return f"tahap_{nomor}_{slug}.md"


def label_tahap(nomor: str) -> str:
    """"07_03" -> "Tahap 7.3"."""
    tahap, sub = nomor.split("_")
    return f"Tahap {int(tahap)}.{int(sub)}"


def jangkar(judul: str) -> str:
    """Jangkar tautan judul ala GitHub."""
    s = judul.strip().lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch in " -_")
    return s.replace(" ", "-")


def tautan(model: str, bagian: str, sub: str | None = None) -> str:
    """Tautan Markdown ke berkas perhitungan manual lain (opsional ke judulnya)."""
    nomor, _, judul = BERKAS[(model, bagian)]
    if sub is None:
        return f"[{label_tahap(nomor)} ({judul})]({nama_berkas(model, bagian)})"
    nomor_sub = sub.split(".")[0]
    return (f"[{label_tahap(nomor)}, bagian {nomor_sub}]"
            f"({nama_berkas(model, bagian)}#{jangkar(sub)})")


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


def nama(kunci: str) -> str:
    """b_z_in -> b_z(in), W_f -> W_f."""
    for akhiran in ("_in", "_rec"):
        if kunci.endswith(akhiran):
            return f"{kunci[: -len(akhiran)]}({akhiran[1:]})"
    return kunci


# --------------------------------------------------------------------------- #
# Penyusun Markdown
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

    def daftar_isi(self, judul):
        self.teks("**Daftar isi**")
        self.teks("\n".join(f"- [{j}](#{jangkar(j)})" for j in judul.values()))

    def isi(self):
        return "\n".join(self.baris).rstrip() + "\n"


CATATAN_ANGKA = """
**Cara membaca angka.** Angka memakai titik sebagai pemisah desimal dan koma
sebagai pemisah ribuan, sama seperti berkas perhitungan manual lainnya. Angka
ditampilkan 6 desimal, tetapi dihitung dengan presisi penuh, sehingga selisih
pembulatan pada digit ke-6 bisa muncul saat Anda menjumlahkan angka yang tampil.
Nomor persamaan dan gambar mengacu pada draf skripsi (subbab 1.5.7 sampai 1.5.11).
"""


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
    # indeks j = keadaan setelah j kali update (j = 0..iterasi)
    return {"riwayat": riwayat,
            "p": [r["p"] for r in riwayat] + [dict(p)],
            "L": [r["L"] for r in riwayat] + [akhir["L"]],
            "yhat": [r["yhat"] for r in riwayat] + [akhir["yhat"]]}


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


class Simulasi:
    """Satu siklus pelatihan model mini beserta pemeriksaannya."""

    def __init__(self, model: str):
        assert model in ("LSTM", "GRU")
        self.model = model
        self.p0 = dict(AWAL[model])
        self.fungsi_maju = lstm_maju if model == "LSTM" else gru_maju
        self.fungsi_mundur = lstm_mundur if model == "LSTM" else gru_mundur
        self.maju = self.fungsi_maju(self.p0)
        self.mundur = self.fungsi_mundur(self.p0, self.maju)
        self.num = gradien_numerik(self.fungsi_maju, self.p0)
        self.selisih = max(abs(self.mundur["g"][n] - self.num[n]) for n in self.p0)
        assert self.selisih < 1e-8, f"{model}: gradien BPTT tidak cocok dengan gradien numerik"
        self.latih = latih_adam(self.fungsi_maju, self.fungsi_mundur, self.p0, ITERASI)
        assert all(abs(abs(v) - LR) < 1e-6 for v in self.latih["riwayat"][0]["delta"].values()), \
            f"{model}: langkah Adam pertama harus ±η"
        assert self.latih["L"][-1] < 1e-4 < self.latih["L"][0]

    @property
    def porsi_jalan_tol(self) -> float:
        """Porsi sinyal kesalahan yang mengalir lewat 'jalan tol gradien'."""
        if self.model == "LSTM":
            r1 = self.mundur["rincian"][0]
            return r1["dc_lanjut"] / r1["dc"]
        r2 = self.mundur["rincian"][1]
        return r2["suku_h"][0] / r2["dh_prev"]


# --------------------------------------------------------------------------- #
# Konteks penelitian (diisi dari variabel notebook)
# --------------------------------------------------------------------------- #
def konteks_penelitian(n_latih, n_val, lookback, n_fitur, batch, lr, neuron_grid,
                       epoch_grid, x_min, x_max):
    """Angka penelitian yang dipakai untuk mengaitkan model mini dengan model asli."""
    assert abs(lr - LR) < 1e-12, "Learning rate simulasi harus sama dengan penelitian"
    iter_epoch = math.ceil(n_latih / batch)
    return {"n_latih": n_latih, "n_val": n_val, "lookback": lookback, "n_fitur": n_fitur,
            "batch": batch, "lr": lr, "neuron_grid": list(neuron_grid),
            "epoch_grid": list(epoch_grid), "iter_epoch": iter_epoch,
            "batch_akhir": n_latih - (iter_epoch - 1) * batch,
            "x_min": float(x_min), "x_max": float(x_max), "rentang": float(x_max - x_min)}


def jumlah_parameter(model: str, n_u: int, n_f: int, dense: bool = True) -> int:
    """Persamaan 22 (LSTM) dan 27 (GRU, reset_after=True)."""
    rekuren = 4 * (n_u * (n_u + n_f) + n_u) if model == "LSTM" else 3 * (n_u * (n_u + n_f) + 2 * n_u)
    return rekuren + (n_u + 1 if dense else 0)


def hasil_model(model, neuron, epoch, tabel_tuning, riwayat, bobot, gambar_loss=None):
    """
    Ringkas hasil pelatihan model terbaik dari variabel notebook.

    ``tabel_tuning`` adalah DataFrame hasil tuning, ``riwayat`` adalah
    ``History.history`` Keras, dan ``bobot`` adalah ``model.get_weights()``.
    """
    baris = tabel_tuning[(tabel_tuning["Neuron"] == neuron) & (tabel_tuning["Epoch"] == epoch)].iloc[0]
    per_epoch = {int(e): float(r) for e, r in zip(
        tabel_tuning.loc[tabel_tuning["Neuron"] == neuron, "Epoch"],
        tabel_tuning.loc[tabel_tuning["Neuron"] == neuron, "RMSE Validasi (USD)"])}
    loss, val = [float(v) for v in riwayat["loss"]], [float(v) for v in riwayat["val_loss"]]
    epoch_min = min(range(len(val)), key=lambda i: val[i]) + 1
    n = int(neuron)
    if model == "LSTM":       # urutan Keras: i, f, c, o
        b = [float(v) for v in bobot[2]]
        bias = {"i": b[0], "f": b[n], "c": b[2 * n], "o": b[3 * n]}
    else:                     # urutan Keras: z, r, h; baris 0 = masukan, baris 1 = rekuren
        b_in, b_rec = [float(v) for v in bobot[2][0]], [float(v) for v in bobot[2][1]]
        bias = {g: (b_in[j * n], b_rec[j * n]) for j, g in enumerate("zrh")}
    return {"model": model, "neuron": n, "epoch": int(epoch),
            "rmse_val": float(baris["RMSE Validasi (USD)"]),
            "param": int(baris["Jumlah Parameter"]), "per_epoch": per_epoch,
            "kurva": {"latih_awal": loss[0], "latih_akhir": loss[-1], "val_awal": val[0],
                      "val_akhir": val[-1], "val_min": val[epoch_min - 1], "epoch_min": epoch_min},
            "bias": bias, "b_y": float(bobot[4][0]), "gambar_loss": gambar_loss}


# =========================================================================== #
# ISI BERKAS: ALUR SEL (Tahap 7.1 dan 8.1)
# =========================================================================== #
def _tabel_bobot(d, model):
    p = AWAL[model]
    if model == "LSTM":
        d.tabel(["Gerbang", "W (bobot masukan)", "U (bobot rekuren)", "b (bias)"],
                [["forget (f)", bt(p["W_f"]), bt(p["U_f"]), bt(p["b_f"]) + " (unit forget bias)"],
                 ["input (i)", bt(p["W_i"]), bt(p["U_i"]), bt(p["b_i"])],
                 ["kandidat (c̃)", bt(p["W_c"]), bt(p["U_c"]), bt(p["b_c"])],
                 ["output (o)", bt(p["W_o"]), bt(p["U_o"]), bt(p["b_o"])],
                 ["dense", f"W_y = {bt(p['W_y'])}", "-", f"b_y = {bt(p['b_y'])}"]])
    else:
        d.tabel(["Gerbang", "W (bobot masukan)", "U (bobot rekuren)", "b(in) (bias masukan)",
                 "b(rec) (bias rekuren)"],
                [["update (z)", bt(p["W_z"]), bt(p["U_z"]), bt(p["b_z_in"]), bt(p["b_z_rec"])],
                 ["reset (r)", bt(p["W_r"]), bt(p["U_r"]), bt(p["b_r_in"]), bt(p["b_r_rec"])],
                 ["kandidat (h̃)", bt(p["W_h"]), bt(p["U_h"]), bt(p["b_h_in"]), bt(p["b_h_rec"])],
                 ["dense", f"W_y = {bt(p['W_y'])}", "-", f"b_y = {bt(p['b_y'])}", "-"]])


def _model_mini(d, model, kt):
    d.teks(f"""
Supaya setiap langkah punya angka nyata yang bisa diikuti dengan tangan, seluruh
subbagian perhitungan manual pelatihan {model} memakai **model mini** yang sama:
**1 neuron, 1 fitur, 2 time step, dan 1 sampel**. Rumus dan urutan langkahnya
identik dengan model penelitian ({kt['n_fitur']} fitur, {kt['lookback']} time step,
batch {kt['batch']}); yang berbeda hanya jumlah angkanya.

**Data** (harga penutupan ternormalisasi, angka ilustrasi):

- hari 1: x₁ = {X[0]}
- hari 2: x₂ = {X[1]}
- target hari 3: y′ = {Y}

**Keadaan awal:** h₀ = 0{' dan c₀ = 0' if model == 'LSTM' else ''}, sama seperti bawaan Keras.

**Bobot awal** dibuat bulat agar mudah dihitung. Keras sebenarnya mengisi bobot secara
acak (Glorot uniform untuk W, ortogonal untuk U), tetapi aturan biasnya diikuti:
{'**bias forget gate = 1**, bias lain = 0' if model == 'LSTM' else 'semua bias = 0'}.
""")
    _tabel_bobot(d, model)


def alur_sel(sim: Simulasi, kt: dict) -> str:
    return _alur_sel_lstm(sim, kt) if sim.model == "LSTM" else _alur_sel_gru(sim, kt)


JUDUL_SIGMOID_TANH = "3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)"


def _sigmoid_tanh(d, s2):
    """Penjabaran persamaan (11)-(13) dari sigmoid dan tanh, dicek dengan angka."""
    def turunan_sig(z):
        return sig(z) * (1 - sig(z))

    d.teks("""
Kotak kuning σ dan tanh pada Gambar 1-6 memakai dua fungsi aktivasi, yaitu
persamaan (11) dan (12). Persamaan (13) menghubungkan keduanya. Bagian ini
menunjukkan cara **memperoleh persamaan (13) sendiri** dari (11) dan (12), seolah-olah
kita belum pernah melihatnya.

**Penting:** persamaan (12) **bukan** hasil olahan persamaan (11). Keduanya definisi
yang berdiri sendiri. Persamaan (13) adalah **jembatan** yang membuktikan bahwa tanh
sebenarnya sigmoid yang dipercuram, direntangkan, lalu digeser.
""")

    d.teks("### 3.1 Sigmoid, Persamaan (11)")
    d.teks(r"$$\sigma(z) = \frac{1}{1+e^{-z}}$$")
    d.teks("Perilakunya ditentukan oleh e^(−z):")
    d.tabel(["z", "e^(−z)", "σ(z) = 1 / (1 + e^(−z))", "Keterangan"],
            [[str(z), a(math.exp(-z)), a(sig(z)), ket]
             for z, ket in [(10, "z besar → e^(−z) ≈ 0 → σ ≈ 1"), (2, ""), (0, "tepat di tengah"),
                            (-2, ""), (-10, "z sangat negatif → e^(−z) sangat besar → σ ≈ 0")]])
    d.teks("Berapa pun nilai z, hasil sigmoid selalu di antara **0 dan 1**, dan σ(0) = 0.5.")

    d.teks("### 3.2 Tanh, Persamaan (12)")
    d.teks(r"$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$")
    d.tabel(["z", "e^z", "e^(−z)", "tanh(z)"],
            [[str(z), a(math.exp(z)), a(math.exp(-z)), a(math.tanh(z))] for z in (10, 2, 0, -2, -10)])
    d.teks("""
Hasil tanh selalu di antara **−1 dan 1**, tanh(0) = 0, dan simetris: tanh(−z) = −tanh(z).
Bentuk kurvanya sama-sama huruf S seperti sigmoid; yang berbeda hanya rentangnya.
""")

    d.teks("### 3.3 Cara 1: Mulai dari Tanh, Cari Bentuk Sigmoid")
    d.teks("""
Ciri khas sigmoid adalah pola **1 / (1 + e^(−sesuatu))**. Tugasnya: ubah rumus tanh
sedikit demi sedikit sampai pola itu muncul di dalamnya.

**Langkah 1 — tulis rumus tanh (12).**
""")
    d.teks(r"$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$")
    d.teks("""
**Langkah 2 — buat penyebut diawali angka 1.** Penyebut sigmoid berbentuk "1 + …",
sedangkan penyebut tanh diawali e^z. Karena itu **pembilang dan penyebut dibagi e^z**.
Ini boleh, karena membagi atas dan bawah pecahan dengan bilangan yang sama tidak
mengubah nilainya, dan e^z tidak pernah 0. Aturan yang dipakai: e^p / e^q = e^(p−q),
sehingga e^z / e^z = e^0 = 1 dan e^(−z) / e^z = e^(−z−z) = e^(−2z).
""")
    d.teks(r"$$\tanh(z) = \frac{\dfrac{e^{z}}{e^{z}} - \dfrac{e^{-z}}{e^{z}}}"
           r"{\dfrac{e^{z}}{e^{z}} + \dfrac{e^{-z}}{e^{z}}} = \frac{1-e^{-2z}}{1+e^{-2z}}$$")
    d.teks("""
**Langkah 3 — kenali pola sigmoid.** Penyebut 1 + e^(−2z) persis sama dengan penyebut
sigmoid (11), asalkan z di sigmoid diganti **2z**. Jadi yang akan muncul adalah σ(2z),
bukan σ(z). **Inilah asal angka 2 di dalam kurung** pada persamaan (13).
""")
    d.teks(r"$$\sigma(2z) = \frac{1}{1+e^{-2z}}$$")
    d.teks("Agar ringkas, misalkan **u = e^(−2z)**, sehingga:")
    d.teks(r"$$\tanh(z) = \frac{1-u}{1+u}, \qquad \sigma(2z) = \frac{1}{1+u}$$")
    d.teks("""
**Langkah 4 — munculkan (1 + u) di pembilang.** Pembilang tanh adalah 1 − u, sedangkan
pembilang sigmoid adalah 1. Agar bisa dicoret dengan penyebut, (1 + u) harus muncul di
pembilang. Caranya dengan **trik tambah-kurang**: tulis 1 sebagai 2 − 1, sehingga
1 − u = 2 − 1 − u = 2 − (1 + u).
""")
    d.teks(r"$$\tanh(z) = \frac{2-(1+u)}{1+u}$$")
    d.teks("**Langkah 5 — pecah menjadi dua pecahan** (penyebutnya sama), lalu coret (1 + u)/(1 + u) = 1.")
    d.teks(r"$$\tanh(z) = \frac{2}{1+u} - \frac{1+u}{1+u} = 2\cdot\frac{1}{1+u} - 1$$")
    d.teks("**Langkah 6 — ganti 1/(1 + u) dengan σ(2z)** dari Langkah 3. Persamaan (13) muncul:")
    d.teks(r"$$\tanh(z) = 2\sigma(2z) - 1 \qquad (13)$$")
    d.tabel(["Langkah", "Yang dilakukan", "Alasannya"],
            [["1", "Tulis tanh (12)", "titik awal"],
             ["2", "Bagi pembilang dan penyebut dengan e^z", "agar penyebut diawali 1, seperti sigmoid"],
             ["3", "Kenali 1 + e^(−2z)", "itu penyebut σ(2z); asal angka 2"],
             ["4", "1 − u = 2 − (1 + u)", "agar (1 + u) muncul di pembilang"],
             ["5", "Pecah pecahan", "(1 + u)/(1 + u) = 1"],
             ["6", "Ganti 1/(1 + u) dengan σ(2z)", "hasil akhir: persamaan (13)"]])

    d.teks("### 3.4 Cara 2: Mulai dari Sigmoid, Cari Bentuk Tanh")
    d.teks("""
Arahnya dibalik. Pangkat pada tanh berpasangan (+z dan −z), jadi sigmoid dibuat
berpasangan juga.

**Langkah 1 — kalikan pembilang dan penyebut sigmoid dengan e^(z/2).** Penyebutnya
menjadi e^(z/2) + e^(−z) · e^(z/2) = e^(z/2) + e^(−z/2), yang pangkatnya berpasangan
seperti tanh.
""")
    d.teks(r"$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^{z/2}}{e^{z/2}+e^{-z/2}}$$")
    d.teks("""
**Langkah 2 — ubah pembilang menjadi selisih.** Pembilang tanh berupa selisih, sedangkan
pembilang di atas hanya satu suku. Rentang nilainya memberi petunjuk: sigmoid bernilai
0 sampai 1, tanh −1 sampai 1. Untuk memindahkan 0..1 ke −1..1, **kalikan 2** (menjadi
0..2) lalu **kurangi 1** (menjadi −1..1). Jadi hitung 2σ(z) − 1:
""")
    d.teks(r"$$2\sigma(z) - 1 = \frac{2e^{z/2} - \left(e^{z/2}+e^{-z/2}\right)}{e^{z/2}+e^{-z/2}}"
           r" = \frac{e^{z/2}-e^{-z/2}}{e^{z/2}+e^{-z/2}} = \tanh\left(\frac{z}{2}\right)$$")
    d.teks("**Langkah 3 — ganti z dengan 2z** di kedua ruas. Hasilnya sama dengan Cara 1:")
    d.teks(r"$$2\sigma(2z) - 1 = \tanh(z)$$")

    d.teks("### 3.5 Cara 3: Tebak dari Bentuk Grafik, Lalu Buktikan")
    z = 0.5
    tebak_1, tebak_2 = 2 * sig(z) - 1, 2 * sig(2 * z) - 1
    assert abs(tebak_1 - math.tanh(z)) > 0.1 and abs(tebak_2 - math.tanh(z)) < 1e-12
    d.teks("Cara ini kira-kira yang dilakukan seseorang yang belum tahu rumusnya.")
    d.kode([
        "1. Samakan rentang. Sigmoid 0..1, tanh −1..1 → tebakan pertama: 2σ(z) − 1.",
        "",
        f"2. Uji dengan angka, z = {z}:",
        f"     2σ({z}) − 1 = 2 × {a(sig(z))} − 1 = {a(tebak_1)}",
        f"     tanh({z})   = {a(math.tanh(z))}          → tidak sama!",
        "",
        "3. Cari penyebabnya lewat kemiringan (turunan) di z = 0:",
        f"     kemiringan σ(z)        = σ(0)(1 − σ(0)) = 0.5 × 0.5 = {a(turunan_sig(0), 2)}",
        f"     kemiringan 2σ(z) − 1   = 2 × {a(turunan_sig(0), 2)} = {a(2 * turunan_sig(0), 2)}",
        f"     kemiringan tanh(z)     = 1 − tanh²(0) = {a(1 - math.tanh(0) ** 2, 2)}",
        "   Tebakan pertama 2 kali kurang curam → ganti z dengan 2z.",
        "",
        "4. Tebakan kedua: 2σ(2z) − 1",
        f"     2σ({2 * z:g}) − 1 = 2 × {a(sig(2 * z))} − 1 = {a(tebak_2)} = tanh({z}) ✓",
        "",
        "5. Tebakan yang cocok ini lalu dibuktikan secara aljabar dengan Cara 1.",
    ])

    d.teks("### 3.6 Arti Persamaan (13)")
    d.teks("Persamaan (13) mengubah sigmoid menjadi tanh dengan tiga operasi:")
    d.tabel(["Operasi", "Bentuk", "Rentang", "Nilai di z = 0", "Kemiringan di z = 0"],
            [["mulai dari sigmoid", "σ(z)", "0 sampai 1", a(sig(0), 1), a(turunan_sig(0), 2)],
             ["① ganti z dengan 2z", "σ(2z)", "0 sampai 1, 2 kali lebih curam", a(sig(0), 1),
              a(2 * turunan_sig(0), 2)],
             ["② kalikan 2", "2σ(2z)", "0 sampai 2", a(2 * sig(0), 1), a(4 * turunan_sig(0), 2)],
             ["③ kurangi 1", "2σ(2z) − 1", "**−1 sampai 1**", f"**{a(2 * sig(0) - 1, 1)}** = tanh(0)",
              f"**{a(4 * turunan_sig(0), 2)}** = kemiringan tanh"]])
    d.teks("""
Singkatnya, tanh adalah sigmoid yang **dibuat 2 kali lebih curam, direntangkan 2 kali ke
atas, lalu digeser turun 1**.
""")

    d.teks("### 3.7 Cek dengan Angka Model Mini")
    ac = s2["a_c"]
    lewat_12 = (math.exp(ac) - math.exp(-ac)) / (math.exp(ac) + math.exp(-ac))
    u = math.exp(-2 * ac)
    lewat_langkah_2 = (1 - u) / (1 + u)
    lewat_13 = 2 * sig(2 * ac) - 1
    assert max(abs(lewat_12 - s2["cc"]), abs(lewat_langkah_2 - s2["cc"]), abs(lewat_13 - s2["cc"])) < 1e-12
    d.teks(f"""
Kandidat c̃₂ pada Gambar 4 (bagian 7) memakai tanh dengan pra-aktivasi a = {a(ac)}.
Ketiga bentuk rumus memberi hasil yang sama:
""")
    d.kode([
        "Lewat persamaan (12):",
        f"  e^a = e^{a(ac)} = {a(math.exp(ac))}      e^(−a) = {a(math.exp(-ac))}",
        f"  tanh(a) = ({a(math.exp(ac))} − {a(math.exp(-ac))}) / ({a(math.exp(ac))} + {a(math.exp(-ac))})"
        f" = {a(math.exp(ac) - math.exp(-ac))} / {a(math.exp(ac) + math.exp(-ac))} = {a(lewat_12)}",
        "",
        "Lewat hasil Langkah 2 (Cara 1):",
        f"  u = e^(−2a) = e^(−{a(2 * ac)}) = {a(u)}",
        f"  (1 − u) / (1 + u) = {a(1 - u)} / {a(1 + u)} = {a(lewat_langkah_2)}",
        "",
        "Lewat persamaan (13):",
        f"  σ(2a) = σ({a(2 * ac)}) = 1 / (1 + {a(u)}) = {a(sig(2 * ac))}",
        f"  2σ(2a) − 1 = 2 × {a(sig(2 * ac))} − 1 = {a(lewat_13)}",
        "",
        f"Ketiganya = c̃₂ = {a(s2['cc'])} ✓",
    ])

    d.teks("### 3.8 Kaitan dengan Turunan pada BPTT")
    d.teks(f"""
Persamaan (13) juga menghubungkan turunan kedua fungsi. Dengan s = σ(2z):

1 − tanh²(z) = 1 − (2s − 1)² = 1 − (4s² − 4s + 1) = 4s(1 − s)

Jadi turunan tanh pun bisa dihitung dari sigmoid, dan keduanya cukup memakai nilai
keluarannya sendiri: σ′ = σ(1 − σ) dan tanh′ = 1 − tanh². Kedua rumus turunan ini
dipakai di setiap langkah backward pada {tautan('LSTM', 'bptt', '1. Rumus Turunan Dasar')}.
""")


def _alur_sel_lstm(sim, kt):
    p = sim.p0
    s1, s2 = sim.maju["langkah"]
    J = {1: "1. Model Mini yang Dipakai", 2: "2. Cara Membaca Gambar", 3: JUDUL_SIGMOID_TANH,
         4: "4. Gambar 1: Struktur LSTM", 5: "5. Gambar 2: Cell State", 6: "6. Gambar 3: Forget Gate",
         7: "7. Gambar 4: Input Gate", 8: "8. Gambar 5: Pembaruan Cell State",
         9: "9. Gambar 6: Output Gate", 10: "10. Ringkasan Satu Langkah LSTM",
         11: "11. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks("""
Subbagian ini menjelaskan alur **Gambar 1 sampai Gambar 6** pada subbab 1.5.8
langkah demi langkah: apa yang mengalir di setiap garis, apa yang dikerjakan setiap
kotak, dan apa fungsinya. Sebelumnya, bagian 3 menjabarkan fungsi aktivasi sigmoid
dan tanh beserta hubungan keduanya (persamaan 11-13). Setiap gerbang diberi contoh
angka dari model mini pada **time step t = 2**, karena di sana memori dari t = 1
sudah ikut bekerja.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _model_mini(d, "LSTM", kt)
    d.teks(f"""
Hasil time step t = 1 yang dipakai di contoh berikut adalah h₁ = {a(s1['h'])} dan
c₁ = {a(s1['c'])}. Perhitungan lengkap kedua time step ada di {tautan('LSTM', 'forward_loss')}.
""")

    d.teks(f"## {J[2]}")
    d.teks("Gambar 1-6 berasal dari Olah (2015). Simbolnya:")
    d.tabel(["Simbol", "Arti"],
            [["Kotak kuning (σ atau tanh)",
              "lapisan jaringan saraf yang **punya bobot dan bias** (W, U, b) dan dipelajari saat pelatihan"],
             ["Lingkaran merah muda (×, +)",
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

    d.teks(f"## {J[3]}")
    _sigmoid_tanh(d, s2)

    d.teks(f"## {J[4]}")
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
menambahkan garis cₜ dan tiga gerbang yang mengatur apa yang dibuang, disimpan, dan
dikeluarkan.

**Kaitan dengan penelitian:** dengan *window* {kt['lookback']} hari, sel A dijalankan
**{kt['lookback']} kali berturut-turut** (xₜ₋₆ sampai xₜ), dan setiap xₜ berisi
{kt['n_fitur']} fitur blockchain ternormalisasi. Di awal, h₀ = c₀ = 0 (bawaan Keras).
Hanya hidden state **langkah terakhir h_T** yang diteruskan ke lapisan dense
(persamaan 21). Model mini dijalankan 2 kali (t = 1 dan t = 2), lalu h₂ masuk ke dense.
""")

    d.teks(f"## {J[5]}")
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
untuk mengingat (fₜ mendekati 1), gradien hampir tidak mengecil. Buktinya ada di
{tautan('LSTM', 'bptt')}: **{persen(sim.porsi_jalan_tol)}** sinyal kesalahan yang
sampai ke c₁ datang lewat jalur cell state ini.
""")

    d.teks(f"## {J[6]}")
    d.teks("""
**Alurnya:**

1. **hₜ₋₁** (masuk dari kiri) dan **xₜ** (masuk dari bawah) bertemu dan menjadi
   masukan bersama.
2. Keduanya masuk ke **kotak kuning σ pertama** dan dihitung pra-aktivasinya:
   xₜW_f + hₜ₋₁U_f + b_f (persamaan 15).
3. Hasilnya dilewatkan ke sigmoid sehingga menjadi **fₜ**, bernilai 0 sampai 1.
4. Panah fₜ naik ke lingkaran **×** di garis cell state dan **mengalikan cₜ₋₁**.

**Fungsinya:** menentukan **berapa banyak memori lama yang dipertahankan**. fₜ mendekati
0 berarti memori dihapus; fₜ mendekati 1 berarti memori dipertahankan utuh.
""")
    d.kode([
        f"Contoh angka (t = 2), memakai h₁ = {a(s2['h_prev'])} dan c₁ = {a(s2['c_prev'])} dari t = 1:",
        f"  a  = W_f × x₂ + U_f × h₁ + b_f = {bt(p['W_f'])} × {X[1]} + {bt(p['U_f'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_f'])} = {a(s2['a_f'])}",
        f"  f₂ = σ({a(s2['a_f'])}) = {a(s2['f'])}",
        f"  Memori lama yang dipertahankan: f₂ × c₁ = {a(s2['f'])} × {a(s2['c_prev'])} = {a(s2['f'] * s2['c_prev'])}",
        f"  Artinya {persen(s2['f'])} isi memori lama dipertahankan dan {persen(1 - s2['f'])} dibuang.",
    ])
    d.teks(f"""
Bias forget bernilai 1 (*unit forget bias* Keras). Karena itu, walaupun xₜ dan hₜ₋₁
bernilai nol, gerbang ini tetap bernilai σ(1) = {a(sig(1.0), 3)}: secara bawaan model
cenderung **mengingat**. Peran bias dijelaskan lengkap di {tautan('LSTM', 'bias')}.
""")

    d.teks(f"## {J[7]}")
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

    d.teks(f"## {J[8]}")
    d.teks("""
Gambar ini menggabungkan hasil Gambar 3 dan Gambar 4 di garis atas (persamaan 18):

**cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**

**Alurnya:** cₜ₋₁ dikalikan fₜ (memori lama terseleksi), lalu ditambah iₜ × c̃ₜ (memori
baru terseleksi). Hasilnya adalah **cₜ**, memori jangka panjang yang sudah diperbarui,
yang dikirim ke langkah waktu berikutnya.

**Fungsinya:** di sinilah memori benar-benar ditulis ulang. Karena fₜ dan iₜ
**independen**, LSTM bisa sekaligus mempertahankan banyak memori lama dan menambah
banyak memori baru. Ini salah satu pembeda utama dengan GRU (lihat Tahap 9.2).
""")
    d.kode([
        "Contoh angka (t = 2), memakai hasil Gambar 3 dan Gambar 4:",
        f"  c₂ = f₂ × c₁ + i₂ × c̃₂ = {a(s2['f'] * s2['c_prev'])} + {a(s2['i'] * s2['cc'])} = {a(s2['c'])}",
        f"  Memori berubah dari c₁ = {a(s2['c_prev'])} menjadi c₂ = {a(s2['c'])}.",
    ])

    d.teks(f"## {J[9]}")
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

**Fungsinya:** menentukan **bagian memori mana yang ditampilkan** sebagai keluaran saat
ini. Tidak semua isi cₜ relevan untuk keluaran sekarang; sebagian cukup disimpan untuk
dipakai nanti.
""")
    d.kode([
        "Contoh angka (t = 2):",
        f"  o₂ = σ(W_o × x₂ + U_o × h₁ + b_o) = σ({bt(p['W_o'])} × {X[1]} + {bt(p['U_o'])} × {a(s2['h_prev'])}"
        f" + {bt(p['b_o'])}) = σ({a(s2['a_o'])}) = {a(s2['o'])}",
        f"  tanh(c₂) = tanh({a(s2['c'])}) = {a(s2['tc'])}        (oval tanh, tanpa bobot)",
        f"  h₂ = o₂ × tanh(c₂) = {a(s2['o'])} × {a(s2['tc'])} = {a(s2['h'])}",
        "  t = 2 adalah langkah terakhir, jadi h₂ masuk ke lapisan dense (persamaan 21):",
        f"  ŷ' = W_y × h₂ + b_y = {bt(p['W_y'])} × {a(s2['h'])} + {bt(p['b_y'])} = {a(sim.maju['yhat'])}",
    ])

    d.teks(f"## {J[10]}")
    d.teks(f"Seluruh alur satu time step LSTM (t = 2, memori masuk c₁ = {a(s2['c_prev'])} "
           f"dan h₁ = {a(s2['h_prev'])}):")
    d.tabel(["Urutan", "Gambar", "Perhitungan", "Hasil", "Makna"],
            [["1. Forget", "Gambar 3", "f₂ = σ(a); f₂ × c₁", f"{a(s2['f'])}; {a(s2['f'] * s2['c_prev'])}",
              f"{persen(s2['f'])} memori lama dipertahankan"],
             ["2. Input", "Gambar 4", "i₂ = σ(a); c̃₂ = tanh(a); i₂ × c̃₂",
              f"{a(s2['i'])}; {a(s2['cc'])}; {a(s2['i'] * s2['cc'])}", f"usulan baru masuk {persen(s2['i'])}"],
             ["3. Update", "Gambar 5", "c₂ = f₂c₁ + i₂c̃₂", a(s2["c"]), "memori jangka panjang baru"],
             ["4. Output", "Gambar 6", "o₂ = σ(a); h₂ = o₂ tanh(c₂)", f"{a(s2['o'])}; {a(s2['h'])}",
              f"{persen(s2['o'])} memori yang sudah dimampatkan dikeluarkan"],
             ["5. Dense", "-", "ŷ′ = W_y h₂ + b_y", a(sim.maju["yhat"]), "prediksi (skala ternormalisasi)"]])
    d.teks(f"""
Prediksi ŷ′ = {a(sim.maju['yhat'])} masih jauh dari target {Y}. Seberapa salah prediksi
ini dan bagaimana bobot diperbaiki dihitung pada langkah berikutnya, mulai dari
{tautan('LSTM', 'forward_loss')}.
""")

    d.teks(f"## {J[11]}")
    d.teks("""
**Persamaan (11)-(13).**

1. Kalimat "Jika fungsi tanh dihubungkan dengan fungsi sigmoid…" sudah tepat, karena
   persamaan (13) memang **hubungan** kedua fungsi, bukan hasil olahan (11) menjadi (12).
2. Bagian *Keterangan* belum menjelaskan tanh(z), padahal simbol ini muncul di persamaan
   (12) dan (13). Sebaiknya ditambahkan:
   `tanh(z) = Fungsi aktivasi tangen hiperbolik`.
3. Jika penjabaran persamaan (13) ingin dicantumkan, versi ringkas Cara 1 (bagian 3.3)
   cukup ditulis dalam beberapa baris:
""")
    d.teks(r"$$\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$$")
    d.teks(r"$$= \frac{1-e^{-2z}}{1+e^{-2z}} \qquad \text{(pembilang dan penyebut dibagi } e^{z}\text{)}$$")
    d.teks(r"$$= \frac{2-\left(1+e^{-2z}\right)}{1+e^{-2z}}$$")
    d.teks(r"$$= 2\cdot\frac{1}{1+e^{-2z}} - 1$$")
    d.teks(r"$$= 2\sigma(2z) - 1 \qquad \text{(karena } \sigma(2z) = \tfrac{1}{1+e^{-2z}}\text{)}$$")
    d.teks("""
Contoh kalimat pengantarnya:

> Dengan membagi pembilang dan penyebut persamaan (12) dengan e^z, penyebutnya berbentuk
> sama dengan penyebut fungsi sigmoid pada persamaan (11) untuk masukan 2z, sehingga
> diperoleh persamaan (13).

**Notasi Cₜ pada Gambar 1-6.** Gambar Olah (2015) memakai huruf besar Cₜ untuk cell
state, sedangkan persamaan skripsi memakai cₜ. Satu kalimat penjelas dapat mencegah
pertanyaan penguji, misalnya:

> Notasi Cₜ pada Gambar 1-6 sama dengan cₜ pada persamaan (18)-(20).
""")
    return d.isi()


def _alur_sel_gru(sim, kt):
    p = sim.p0
    g1, g2 = sim.maju["langkah"]
    J = {1: "1. Model Mini yang Dipakai", 2: "2. Cara Membaca Gambar 7",
         3: "3. Alur Gambar 7 Langkah demi Langkah", 4: "4. Fungsi Update Gate dan Reset Gate",
         5: "5. Ringkasan Satu Langkah GRU", 6: "6. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks("""
Subbagian ini menjelaskan alur **Gambar 7** (struktur sel GRU) pada subbab 1.5.9
langkah demi langkah: apa yang mengalir di setiap garis, apa yang dikerjakan setiap
kotak dan lingkaran, dan apa fungsinya. Setiap langkah diberi contoh angka dari model
mini GRU pada **time step t = 2**.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _model_mini(d, "GRU", kt)
    d.teks(f"""
Setiap gerbang GRU punya **dua bias**, yaitu bias masukan b(in) dan bias rekuren
b(rec), karena Keras memakai `reset_after=True` (dijelaskan di {tautan('GRU', 'bias')}).
Hasil time step t = 1 yang dipakai di contoh berikut adalah h₁ = {a(g1['h'])}.
Perhitungan lengkap kedua time step ada di {tautan('GRU', 'forward_loss')}.
""")

    d.teks(f"## {J[2]}")
    d.tabel(["Simbol pada Gambar 7", "Arti"],
            [["Kotak kuning σ (kiri dan tengah)",
              "lapisan dengan bobot dan bias yang menghasilkan update gate zₜ dan reset gate rₜ"],
             ["Kotak kuning tanh", "lapisan dengan bobot dan bias yang menghasilkan kandidat h̃ₜ"],
             ["Lingkaran ×", "perkalian elemen demi elemen (⊙), tanpa bobot"],
             ["Lingkaran 1−", "menghitung 1 − zₜ"],
             ["Lingkaran +", "penjumlahan, menghasilkan hₜ"],
             ["Garis atas", "hidden state hₜ, satu-satunya memori GRU"]])
    d.teks(f"""
Seperti pada LSTM, σ (0 sampai 1) berperan sebagai **keran** dan tanh (-1 sampai 1)
sebagai **isi informasi**; penjelasan lengkapnya di {tautan('LSTM', 'alur_sel', '2. Cara Membaca Gambar')}.
Rumus kedua fungsi itu dan penjabaran hubungannya, tanh(z) = 2σ(2z) − 1 (persamaan 11-13),
ada di {tautan('LSTM', 'alur_sel', JUDUL_SIGMOID_TANH)}.
GRU **hanya punya satu garis memori**, yaitu hₜ. Tidak ada cₜ terpisah dan tidak ada
output gate.
""")

    d.teks(f"## {J[3]}")
    d.teks(f"""
**Langkah 0 — masukan.**

- hₜ₋₁ masuk dari **kiri atas**, lalu bercabang: tetap di garis atas, turun lewat garis
  vertikal kiri ke jalur bawah, dan bercabang ke tengah menuju lingkaran × milik rₜ.
- xₜ masuk dari **bawah** ke jalur horizontal bawah.
- Jalur bawah membawa hₜ₋₁ dan xₜ ke ketiga kotak kuning: σ (z), σ (r), dan tanh.
- Contoh: h₁ = {a(g2['h_prev'])} (hasil t = 1) dan x₂ = {X[1]}.

**Langkah 1 — update gate (kotak σ kiri) menghasilkan zₜ** (persamaan 23).

- z₂ = σ(W_z × x₂ + U_z × h₁ + b_z(in) + b_z(rec))
  = σ({bt(p['W_z'])} × {X[1]} + {bt(p['U_z'])} × {a(g2['h_prev'])} + 0 + 0) = σ({a(g2['a_z'])}) = **{a(g2['z'])}**.
- Panah zₜ naik dan **bercabang dua**: ke **lingkaran × kiri atas**, mengalikan hₜ₋₁
  menjadi z₂ × h₁ = {a(g2['z'])} × {a(g2['h_prev'])} = **{a(g2['z'] * g2['h_prev'])}**; dan ke
  kanan menuju **lingkaran "1−"**, menghasilkan 1 − z₂ = **{a(1 - g2['z'])}**.

**Langkah 2 — reset gate (kotak σ tengah) menghasilkan rₜ** (persamaan 24).

- r₂ = σ(W_r × x₂ + U_r × h₁ + b_r(in) + b_r(rec))
  = σ({bt(p['W_r'])} × {X[1]} + {bt(p['U_r'])} × {a(g2['h_prev'])} + 0 + 0) = σ({a(g2['a_r'])}) = **{a(g2['r'])}**.
- Panah rₜ naik ke **lingkaran × tengah**, tempat ia mengalikan memori lama yang datang
  dari kiri.

**Langkah 3 — kandidat (kotak tanh) menghasilkan h̃ₜ** (persamaan 25).

- Cabang memori lama melewati lingkaran × milik rₜ sebelum masuk tanh. Pada Keras
  (`reset_after=True`), yang dikalikan rₜ adalah **bagian rekuren**
  qₜ = hₜ₋₁U_h + b_h(rec): q₂ = {bt(p['U_h'])} × {a(g2['h_prev'])} + {bt(p['b_h_rec'])} = {a(g2['q'])},
  sehingga r₂ × q₂ = {a(g2['r'])} × {a(g2['q'])} = {a(g2['rq'])}.
- xₜ masuk dari bawah sebagai **bagian masukan**:
  x₂W_h + b_h(in) = {bt(p['W_h'])} × {X[1]} + {bt(p['b_h_in'])} = {a(g2['masuk'])}.
- h̃₂ = tanh({a(g2['masuk'])} + {a(g2['rq'])}) = tanh({a(g2['a_h'])}) = **{a(g2['hh'])}**.

**Langkah 4 — pencampuran** (persamaan 26).

- h̃ₜ naik ke **lingkaran × kanan** dan dikalikan (1 − zₜ):
  {a(1 - g2['z'])} × {a(g2['hh'])} = {a((1 - g2['z']) * g2['hh'])}.
- Hasilnya naik ke **lingkaran +** dan dijumlahkan dengan zₜ × hₜ₋₁ dari kiri:
  h₂ = {a(g2['z'] * g2['h_prev'])} + {a((1 - g2['z']) * g2['hh'])} = **{a(g2['h'])}**.

**Langkah 5 — keluaran.** h₂ keluar ke **kanan** (ke langkah berikutnya) dan ke
**atas** (keluaran waktu t). Karena t = 2 adalah langkah terakhir, h₂ masuk ke dense:
ŷ′ = {bt(p['W_y'])} × {a(g2['h'])} + {bt(p['b_y'])} = **{a(sim.maju['yhat'])}**.
""")

    d.teks(f"## {J[4]}")
    d.teks(f"""
**Update gate zₜ bekerja seperti penggeser (slider) pencampur.** zₜ mendekati 1 berarti
hₜ hampir sama dengan hₜ₋₁ (memori lama dipertahankan); zₜ mendekati 0 berarti hₜ hampir
sama dengan h̃ₜ (diganti informasi baru). Porsi lama dan porsi baru **selalu berjumlah 1**.
Pada contoh: {persen(g2['z'])} memori lama + {persen(1 - g2['z'])} kandidat baru.

**Reset gate rₜ mengatur seberapa banyak masa lalu dipakai untuk menyusun usulan
baru.** rₜ mendekati 0 berarti kandidat disusun hampir hanya dari xₜ ("mulai dari nol");
rₜ mendekati 1 berarti masa lalu ikut diperhitungkan penuh. Pada t = 1, q₁ = 0 karena
h₀ = 0, jadi reset gate belum berpengaruh.

**Jalur langsung zₜ ⊙ hₜ₋₁** (lingkaran × kiri atas ke lingkaran +) adalah "jalan tol
gradien" pada GRU, padanan cell state pada LSTM. Di {tautan('GRU', 'bptt')} terlihat
**{persen(sim.porsi_jalan_tol)}** sinyal kesalahan dari t = 2 ke t = 1 mengalir lewat
jalur ini.
""")

    d.teks(f"## {J[5]}")
    d.tabel(["Urutan", "Bagian Gambar 7", "Perhitungan", "Hasil (t = 2)"],
            [["1. Update gate", "kotak σ kiri", "z₂ = σ(a)", a(g2["z"])],
             ["2. Reset gate", "kotak σ tengah", "r₂ = σ(a)", a(g2["r"])],
             ["3. Kandidat", "lingkaran × tengah, kotak tanh", "h̃₂ = tanh(x₂W_h + b_h(in) + r₂q₂)",
              f"r₂q₂ = {a(g2['rq'])}; h̃₂ = {a(g2['hh'])}"],
             ["4. Pencampuran", "lingkaran × kiri, 1−, × kanan, +", "h₂ = z₂h₁ + (1 − z₂)h̃₂",
              f"{a(g2['z'] * g2['h_prev'])} + {a((1 - g2['z']) * g2['hh'])} = {a(g2['h'])}"],
             ["5. Dense", "-", "ŷ′ = W_y h₂ + b_y", a(sim.maju["yhat"])]])
    d.teks(f"""
Prediksi ŷ′ = {a(sim.maju['yhat'])} masih jauh dari target {Y}. Langkah perbaikannya
dihitung mulai dari {tautan('GRU', 'forward_loss')}.
""")

    d.teks(f"## {J[6]}")
    d.teks("""
1. **Klaim tentang rumus asli Cho et al. (2014) perlu diperbaiki** (subbab 1.5.9,
   halaman 17 draf). Persamaan (7) pada makalah aslinya berbunyi
   hⱼ⟨t⟩ = zⱼ hⱼ⟨t−1⟩ + (1 − zⱼ) h̃ⱼ⟨t⟩, yaitu **sama** dengan persamaan (26) (konvensi
   Keras), sehingga peran zₜ **tidak tertukar**. Bentuk dengan peran zₜ tertukar,
   hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, berasal dari Chung et al. (2014). Perbedaan dengan
   Cho et al. (2014) hanya pada posisi reset gate. Saran kalimat:

   > Rumus GRU pada makalah asli Cho et al. (2014) sedikit berbeda, yaitu gerbang
   > reset diterapkan sebelum perkalian dengan matriks bobot rekuren,
   > h̃ₜ = tanh(xₜW_h + (rₜ ⊙ hₜ₋₁)U_h). Adapun bentuk
   > hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ, dengan peran zₜ tertukar, digunakan oleh
   > Chung et al. (2014).

2. **Gambar 7 dan pengaturan `reset_after=True`.** Gambar 7 menggambar rₜ mengalikan
   hₜ₋₁ sebelum kotak tanh (bentuk Cho et al.), sedangkan persamaan (25) mengalikan rₜ
   dengan (hₜ₋₁U_h + b_h(rec)) setelah perkalian matriks. Fungsinya sama, jadi gambar
   tidak perlu diubah. Cukup tambahkan kalimat berikut, atau beri label garis dari
   lingkaran × rₜ ke kotak tanh dengan rₜ ⊙ (hₜ₋₁U_h + b_h(rec)) dan garis xₜ ke kotak
   tanh dengan xₜW_h + b_h(in):

   > Gambar 7 merupakan ilustrasi konseptual; urutan perhitungan yang digunakan
   > mengikuti persamaan (25).
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: LANGKAH 1-2, FORWARD PASS DAN LOSS (Tahap 7.2 dan 8.2)
# =========================================================================== #
def _gambaran_siklus(d, model):
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
Urutannya selalu: **loss dihitung → gradien dihitung dari loss → Adam memakai gradien
untuk memperbarui bobot**. Satu putaran disebut **satu iterasi k**:
""")
    d.kode("""
        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
 Bobot & bias θ ─► (1) FORWARD ─► (2) LOSS ─► (3) BACKWARD (BPTT) ─► (4) ADAM
                    prediksi ŷ'    seberapa     gradien g = ∂L/∂θ      θ baru
                                   salah?       untuk setiap θ
""")
    persamaan = "(15)-(21)" if model == "LSTM" else "(23)-(26), (21)"
    d.tabel(["Langkah", "Pertanyaan yang dijawab", "Persamaan skripsi", "Dihitung di"],
            [["1. Forward pass", "Dengan bobot sekarang, berapa prediksinya?", persamaan,
              f"subbagian ini ({label_tahap(berkas(model, 'forward_loss')[0])})"],
             ["2. Loss", "Seberapa salah prediksinya?", "(28)",
              f"subbagian ini ({label_tahap(berkas(model, 'forward_loss')[0])})"],
             ["3. Backward (BPTT)", "Bobot mana yang menyebabkan salah, ke arah mana?", "aturan rantai",
              tautan(model, "bptt")],
             ["4. Update Adam", "Seberapa jauh setiap bobot digeser?", "(29)-(31)", tautan(model, "adam")]])


def _kapan_dipakai(d, kt):
    contoh_n, contoh_e = kt["neuron_grid"][len(kt["neuron_grid"]) // 2], max(kt["epoch_grid"])
    d.teks(f"""
Di penelitian, data latih berisi **{ribu(kt['n_latih'])} sampel** (jendela {kt['lookback']} hari ×
{kt['n_fitur']} fitur) dengan **batch size {kt['batch']}**. Jadi satu epoch terdiri dari
⌈{ribu(kt['n_latih'])} / {kt['batch']}⌉ = **{kt['iter_epoch']} batch**: {kt['iter_epoch'] - 1} batch
berisi {kt['batch']} sampel dan 1 batch terakhir berisi {kt['batch_akhir']} sampel. Untuk
**setiap kombinasi neuron × epoch** pada grid tuning (misalnya {contoh_n} neuron,
{ribu(contoh_e)} epoch), pelatihan berjalan sebagai berikut:

**Langkah 0 — persiapan model.** Bobot diisi acak (Glorot uniform untuk W, ortogonal
untuk U); bias = 0 kecuali bias forget LSTM = 1. Memori Adam di-nol-kan: m₀ = 0,
v₀ = 0, penghitung k = 0.

**Langkah 1-5 — diulang untuk setiap batch**, berurutan secara kronologis karena
`shuffle=False`:

1. **Forward pass.** {kt['batch']} jendela masuk ke lapisan rekuren lalu dense,
   menghasilkan {kt['batch']} prediksi ŷ′.
2. **Hitung loss.** MSE dari {kt['batch']} prediksi itu terhadap nilai aktual y′.
3. **Hitung gradien.** BPTT menghasilkan gₖ untuk **setiap parameter**.
4. **Update Adam.** Setiap parameter digeser memakai persamaan (29)-(31).
5. k bertambah 1, lalu lanjut ke batch berikutnya.

**Langkah 6 — akhir setiap epoch.** Keras mencatat **loss latih** (rata-rata loss dari
{kt['iter_epoch']} batch selama epoch itu) dan menghitung **loss validasi** pada
{kt['n_val']} sampel validasi memakai bobot akhir epoch. Loss validasi **hanya diukur;
tidak ada update bobot**. Kedua angka ini digambar sebagai kurva loss.

**Langkah 7 — setelah semua epoch selesai.** Bobot pada **epoch terakhir** yang dipakai
(penelitian tidak memakai *early stopping*). Prediksi data validasi didenormalisasi ke
USD, lalu **RMSE validasi** dipakai untuk memilih kombinasi neuron × epoch terbaik.
""")
    d.teks(f"**Jumlah update Adam per model** ({kt['iter_epoch']} per epoch):")
    d.tabel(["Epoch pada grid", "Jumlah update"],
            [[ribu(e), f"{kt['iter_epoch']} × {ribu(e)} = {ribu(kt['iter_epoch'] * e)}"]
             for e in sorted(kt["epoch_grid"])])
    d.teks("""
**Kapan loss dan Adam TIDAK dipakai:** saat memilih model terbaik (dipakai RMSE
validasi dalam USD), saat memprediksi data uji (hanya forward pass, bobot sudah beku),
serta saat evaluasi akhir (RMSE, MAE, MAPE, akurasi arah) dan uji Diebold-Mariano.
Singkatnya, **loss dan Adam hanya bekerja di fase pelatihan**.
""")


def _notasi(d, model):
    baris = [["a", "pra-aktivasi: nilai sebelum σ atau tanh, misalnya a = W·x + U·h + b"],
             ["σ, tanh", "fungsi aktivasi sigmoid (persamaan 11) dan tanh (persamaan 12)"],
             ["∂L/∂θ", "turunan parsial loss terhadap θ; variabel lain dianggap konstan"],
             ["δ", "sinyal kesalahan sebuah gerbang = ∂L/∂a (turunan loss terhadap pra-aktivasinya)"],
             ["g", "gradien sebuah parameter = ∂L/∂θ"],
             ["m, v", "momen pertama dan kedua Adam (persamaan 29)"],
             ["k", "nomor iterasi (satu kali update = satu batch)"]]
    if model == "GRU":
        baris.insert(1, ["q", "bagian rekuren kandidat GRU: qₜ = U_h·hₜ₋₁ + b_h(rec)"])
    d.teks("**Notasi yang dipakai:**")
    d.tabel(["Simbol", "Arti"], baris)


def _blok_gerbang(baris, judul, simbol, kode_bobot, x, h_prev, akt, aktivasi, x_nama, h_nama, bias_suku):
    """Rincian pra-aktivasi satu gerbang pada satu time step."""
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


def _forward_lstm(d, sim):
    p, maju = sim.p0, sim.maju
    s1, s2 = maju["langkah"]
    d.teks(fr"""
Rumus versi 1 neuron (persamaan 15-21). Karena hanya ada satu neuron, semua bobot
berupa angka tunggal (skalar), bukan matriks. Makna setiap gerbang dijelaskan di
{tautan('LSTM', 'alur_sel')}; di sini dihitung angkanya untuk kedua time step.

$$f_t = \sigma(W_f x_t + U_f h_{{t-1}} + b_f) \qquad
i_t = \sigma(W_i x_t + U_i h_{{t-1}} + b_i) \qquad
\tilde{{c}}_t = \tanh(W_c x_t + U_c h_{{t-1}} + b_c)$$

$$c_t = f_t\,c_{{t-1}} + i_t\,\tilde{{c}}_t \qquad
o_t = \sigma(W_o x_t + U_o h_{{t-1}} + b_o) \qquad
h_t = o_t \tanh(c_t) \qquad
\hat{{y}}' = W_y h_T + b_y$$
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
            _blok_gerbang(baris, judul, simbol, (f"W_{gb}", f"U_{gb}"), s["x"], s["h_prev"],
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


def _forward_gru(d, sim):
    p, maju = sim.p0, sim.maju
    s1, s2 = maju["langkah"]
    d.teks(fr"""
Rumus versi 1 neuron (persamaan 23-26, konvensi Keras `reset_after=True`). Untuk
kandidat, bagian rekurennya diberi nama $q_t$ supaya terlihat jelas apa yang dikalikan
reset gate. Alur setiap langkah pada Gambar 7 dijelaskan di {tautan('GRU', 'alur_sel')};
di sini dihitung angkanya untuk kedua time step.

$$z_t = \sigma\bigl(W_z x_t + U_z h_{{t-1}} + b_z^{{(in)}} + b_z^{{(rec)}}\bigr) \qquad
r_t = \sigma\bigl(W_r x_t + U_r h_{{t-1}} + b_r^{{(in)}} + b_r^{{(rec)}}\bigr)$$

$$q_t = U_h h_{{t-1}} + b_h^{{(rec)}} \qquad
\tilde{{h}}_t = \tanh\bigl(W_h x_t + b_h^{{(in)}} + r_t\,q_t\bigr) \qquad
h_t = z_t\,h_{{t-1}} + (1-z_t)\,\tilde{{h}}_t \qquad
\hat{{y}}' = W_y h_T + b_y$$
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
            _blok_gerbang(baris, judul, gb + sub, (f"W_{gb}", f"U_{gb}"), s["x"], s["h_prev"],
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
        baris.append("Hidden state baru (persamaan 26, Gambar 7: lingkaran ×, 1−, dan +)")
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


def _konsep_mse(d, kt, sim):
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
    norm_usd = contoh_usd / kt["rentang"]
    d.teks(f"""
**Mengapa dikuadratkan?**

1. Selisih positif dan negatif tidak saling meniadakan.
2. Kesalahan besar dihukum jauh lebih berat: selisih {besar:.2f} menyumbang
   {round((besar / kecil) ** 2)} kali lebih besar daripada selisih {kecil:.2f}, sehingga model
   "dipaksa" menghindari meleset jauh.
3. Fungsi kuadrat **mulus dan bisa diturunkan di semua titik**. Turunannya,
   ∂L/∂ŷ′ = −2(y′ − ŷ′), menjadi titik awal BPTT ({tautan(sim.model, 'bptt')}). MAE (nilai
   mutlak) tidak mulus di titik nol.

**Mengapa dihitung pada skala ternormalisasi, bukan USD?** Harga Bitcoin bernilai
puluhan ribu USD. Selisih {usd(contoh_usd)} USD jika dikuadratkan menjadi
{usd(contoh_usd ** 2)}, sehingga gradien sangat besar dan pelatihan tidak stabil. Dengan
rentang harga data latih {usd(kt['rentang'])} USD, selisih yang sama pada skala 0-1 hanya
{a(norm_usd)} dan kuadratnya {a(norm_usd ** 2)}.

**Nilai N dalam praktik:** saat pelatihan N = {kt['batch']} (ukuran batch; batch
terakhir N = {kt['batch_akhir']}), loss validasi dihitung pada N = {kt['n_val']} sampel
validasi, dan pada model mini N = 1.
""")


def forward_loss(sim: Simulasi, kt: dict) -> str:
    model = sim.model
    if model == "LSTM":
        J = {1: "1. Gambaran Satu Siklus Pelatihan", 2: "2. Kapan Loss dan Adam Dipakai dalam Penelitian",
             3: "3. Model Mini dan Notasi", 4: "4. Langkah 1: Forward Pass", 5: "5. Langkah 2: Loss MSE"}
    else:
        J = {1: "1. Gambaran Satu Siklus Pelatihan", 3: "2. Model Mini dan Notasi",
             4: "3. Langkah 1: Forward Pass", 5: "4. Langkah 2: Loss MSE"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini memulai **simulasi satu siklus pelatihan {model}** pada model mini. Di sini
dihitung dua langkah pertama: **forward pass** (berapa prediksinya) dan **loss**
(seberapa salah prediksinya). Langkah 3 dan 4 dilanjutkan di {tautan(model, 'bptt')} dan
{tautan(model, 'adam')}.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _gambaran_siklus(d, model)
    if model == "GRU":
        d.teks(f"""
Prosedur pelatihan GRU di penelitian **identik** dengan LSTM: data, batch size,
optimizer Adam, learning rate, dan seed sama persis; yang berbeda hanya jenis lapisan
rekuren. Kapan loss dan Adam dipakai serta berapa kali update terjadi per epoch
dijelaskan di {tautan('LSTM', 'forward_loss', '2. Kapan Loss dan Adam Dipakai dalam Penelitian')}.
""")
    else:
        d.teks(f"## {J[2]}")
        _kapan_dipakai(d, kt)

    d.teks(f"## {J[3]}")
    d.tabel(["Komponen", "Model mini", "Model penelitian"],
            [["Neuron", "1", f"grid {', '.join(str(n) for n in kt['neuron_grid'])}"],
             ["Fitur per hari", "1 (harga penutupan ternormalisasi)", str(kt["n_fitur"])],
             ["Time step (window)", "2", str(kt["lookback"])],
             ["Sampel per iterasi", "1 (N = 1)", f"{kt['batch']} (batch size)"],
             ["Optimizer", f"Adam, η = {LR}", f"Adam, η = {kt['lr']:g}"]])
    d.teks(f"""
Data model mini: x₁ = {X[0]}, x₂ = {X[1]}, target y′ = {Y}; keadaan awal
h₀ = 0{' dan c₀ = 0' if model == 'LSTM' else ''}. Bobot awal (sama dengan {tautan(model, 'alur_sel')}):
""")
    _tabel_bobot(d, model)
    jumlah = jumlah_parameter(model, 1, 1)
    rumus = ("4 × [n_u × (n_u + n_f) + n_u] + (n_u + 1) = 4 × [1 × (1 + 1) + 1] + (1 + 1) = 4 × 3 + 2"
             if model == "LSTM" else
             "3 × [n_u × (n_u + n_f) + 2 × n_u] + (n_u + 1) = 3 × [1 × (1 + 1) + 2] + (1 + 1) = 3 × 4 + 2")
    d.kode([f"Jumlah parameter model mini (persamaan {'22' if model == 'LSTM' else '27'}), n_u = 1, n_f = 1:",
            f"  P = {rumus} = {jumlah} parameter"])
    assert jumlah == len(sim.p0)
    _notasi(d, model)

    d.teks(f"## {J[4]}")
    (_forward_lstm if model == "LSTM" else _forward_gru)(d, sim)

    d.teks(f"## {J[5]}")
    if model == "LSTM":
        _konsep_mse(d, kt, sim)
    else:
        d.teks(f"""
Fungsi loss sama dengan LSTM, yaitu MSE (persamaan 28). Alasan dikuadratkan dan alasan
dihitung pada skala ternormalisasi dijelaskan di
{tautan('LSTM', 'forward_loss', '5. Langkah 2: Loss MSE')}. Pada model mini N = 1:
""")
    d.kode([
        "Persamaan (28) dengan N = 1 sampel:",
        "  L = (y' - ŷ')²",
        f"    = ({Y} - {a(sim.maju['yhat'])})²",
        f"    = {a(Y - sim.maju['yhat'])}²",
        f"    = {a(sim.maju['L'])}",
    ])
    d.teks(f"""
Prediksi masih **terlalu rendah** ({a(sim.maju['yhat'])} padahal seharusnya {Y}). Langkah
berikutnya mencari tahu bobot mana yang perlu diubah agar loss ini turun:
{tautan(model, 'bptt')}.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: LANGKAH 3, BACKPROPAGATION THROUGH TIME (Tahap 7.3 dan 8.3)
# =========================================================================== #
def _rumus_turunan(d, model):
    s = sig(1.25)
    d.teks("""
Seluruh backward pass hanya mengulang beberapa rumus turunan berikut. Polanya selalu
sama: **kalikan sinyal kesalahan dengan turunan lokal, lalu teruskan ke kiri (ke langkah
sebelumnya).**
""")
    perkalian = ("h = o·tanh(c), c = f·cₜ₋₁ + i·c̃" if model == "LSTM"
                 else "h = z·hₜ₋₁ + (1−z)·h̃, r·q pada kandidat")
    sigmoid_di = "gerbang f, i, o" if model == "LSTM" else "gerbang z, r"
    tanh_di = "c̃, tanh(c)" if model == "LSTM" else "h̃"
    jumlah_di = ("h₁ dipakai 4 gerbang; bobot dipakai di setiap t; c₁ punya 2 jalur" if model == "LSTM"
                 else "h₁ dipakai di 4 tempat; bobot dipakai di setiap t")
    d.tabel(["Rumus", "Bentuk", "Contoh", "Dipakai di"],
            [["Aturan pangkat", "d(u²)/du = 2u", "d(x²)/dx = 2x", "∂L/∂ŷ′"],
             ["Fungsi linear", "d(a·u + b)/du = a", "d(3x + 5)/dx = 3",
              "pra-aktivasi dan dense: ∂a/∂W = x, ∂a/∂U = hₜ₋₁, ∂a/∂b = 1"],
             ["Aturan perkalian", "∂(u·v)/∂u = v", "∂(3u)/∂u = 3", perkalian],
             ["Aturan rantai", "∂L/∂a = ∂L/∂b · ∂b/∂a", "-", "semua langkah"],
             ["Aturan penjumlahan", "variabel yang dipakai di beberapa tempat: gradiennya dijumlah", "-",
              jumlah_di],
             ["Turunan sigmoid", "σ′(a) = σ(a)·(1 − σ(a))", f"σ(1.25) = {a(s)} → σ′ = {a(s * (1 - s))}",
              sigmoid_di],
             ["Turunan tanh", "tanh′(a) = 1 − tanh²(a)",
              f"tanh(0.35) = {a(math.tanh(0.35))} → {a(1 - math.tanh(0.35) ** 2)}", tanh_di]])
    if model == "LSTM":
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
        d.teks(f"Persamaan (13) sendiri dijabarkan langkah demi langkah di "
               f"{tautan('LSTM', 'alur_sel', JUDUL_SIGMOID_TANH)}.")
    else:
        d.teks(f"Asal turunan sigmoid dan tanh diuraikan di "
               f"{tautan('LSTM', 'bptt', '1. Rumus Turunan Dasar')}.")


def _dense_mundur(d, sim, rinci=True):
    p, maju, mundur = sim.p0, sim.maju, sim.mundur
    s2 = maju["langkah"][1]
    g, dy = mundur["g"], mundur["dy"]
    if not rinci:
        d.teks(f"Rumusnya identik dengan {tautan('LSTM', 'bptt', '2. Backward di Lapisan Dense')}; "
               "hanya nilai h₂ dan ŷ′ yang berbeda.")
        d.kode([
            f"  ∂L/∂ŷ'  = -2(y' - ŷ') = -2 × ({Y} - {a(maju['yhat'])}) = {a(dy)}",
            f"  ∂L/∂W_y = ∂L/∂ŷ' × h₂ = {kr(dy)} × {a(s2['h'])} = {a(g['W_y'])}",
            f"  ∂L/∂b_y = ∂L/∂ŷ' × 1  = {a(g['b_y'])}",
            f"  ∂L/∂h₂  = ∂L/∂ŷ' × W_y = {kr(dy)} × {bt(p['W_y'])} = {a(mundur['dh_T'])}   (masuk ke sel GRU)",
        ])
        return
    d.teks("""
Backward berjalan dari kanan ke kiri, mulai dari loss. Rantainya:
`h₂ → ŷ' = W_y·h₂ + b_y → L = (y' − ŷ')²`. Setiap turunan memakai rumus dari bagian 1.
""")
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
        "(4) ∂L/∂h₂ — sinyal kesalahan yang dikirim MASUK ke sel (awal BPTT)",
        "    ∂ŷ'/∂h₂ = W_y            (kali ini h₂ yang menjadi variabel)",
        f"    ∂L/∂h₂ = ∂L/∂ŷ' × W_y = {kr(dy)} × {bt(p['W_y'])} = {a(mundur['dh_T'])}",
    ])
    d.teks(f"""
**Makna setiap angka:**

- **∂L/∂ŷ′ = {a(dy)}**: tanda negatif berarti menaikkan prediksi akan menurunkan loss
  (sesuai, karena prediksi masih terlalu rendah). Jika ŷ′ naik 0.001, loss turun sekitar
  {a(abs(dy) * 0.001, 7)}. Nilai ini adalah **sinyal kesalahan keluaran** yang dipakai ulang
  oleh tiga turunan berikutnya.
- **∂L/∂W_y = {a(g['W_y'])}**: pengaruh W_y terhadap prediksi "dikali" h₂. Karena h₂
  hanya {a(s2['h'])}, gradien W_y lebih kecil daripada gradien ŷ′.
- **∂L/∂b_y = {a(g['b_y'])}**: bias langsung ditambahkan ke prediksi, jadi gradiennya
  sama dengan sinyal kesalahan keluaran.
- **∂L/∂h₂ = {a(mundur['dh_T'])}**: h₂ **bukan parameter**, jadi tidak diupdate Adam.
  Nilai ini adalah pintu masuk BPTT ke dalam sel.
""")
    L0, geser = maju["L"], 0.001
    uji = [("ŷ'", dy, (Y - (maju["yhat"] + geser)) ** 2 - L0),
           ("W_y (0.8 → 0.801)", g["W_y"], (Y - ((p["W_y"] + geser) * s2["h"] + p["b_y"])) ** 2 - L0),
           ("b_y (0 → 0.001)", g["b_y"], (Y - (p["W_y"] * s2["h"] + p["b_y"] + geser)) ** 2 - L0),
           ("h₂", mundur["dh_T"], (Y - (p["W_y"] * (s2["h"] + geser) + p["b_y"])) ** 2 - L0)]
    for _, grad, beda in uji:
        assert abs(grad * geser - beda) < 2e-6
    d.teks("""
**Bukti dengan uji geser.** Turunan berarti "perubahan loss bila variabel digeser
sedikit". Setiap besaran digeser +0.001, lalu loss dihitung ulang:
""")
    d.tabel(["Yang digeser +0.001", "Perkiraan dari gradien (gradien × 0.001)", "Perubahan loss sebenarnya"],
            [[n, f"{kr(gr)} × 0.001 = {a(gr * geser, 7)}", a(bd, 7)] for n, gr, bd in uji])
    d.teks("""
Selisih kecil (sekitar 0.000001) muncul karena turunan adalah pendekatan garis lurus,
sedangkan loss berbentuk kuadrat; sisanya sebesar (0.001 × ∂ŷ′/∂θ)².
""")


def _pemeriksaan_numerik(d, sim):
    g = sim.mundur["g"]
    d.teks(r"""
Untuk memastikan seluruh rantai turunan di atas benar, setiap gradien dibandingkan
dengan turunan numerik (beda pusat):
$\dfrac{L(\theta+10^{-6}) - L(\theta-10^{-6})}{2\cdot10^{-6}}$.
""")
    d.tabel(["Parameter", "Gradien BPTT (manual)", "Gradien numerik", "Selisih"],
            [[nama(n), a(g[n], 8), a(sim.num[n], 8), f"{abs(g[n] - sim.num[n]):.1e}"] for n in sim.p0])
    d.teks(f"Selisih terbesar hanya {sim.selisih:.1e}, jadi seluruh gradien manual terbukti benar.")


def bptt(sim: Simulasi, kt: dict) -> str:
    return _bptt_lstm(sim, kt) if sim.model == "LSTM" else _bptt_gru(sim, kt)


def _bptt_lstm(sim, kt):
    p = sim.p0
    s1, s2 = sim.maju["langkah"]
    r1, r2 = sim.mundur["rincian"]
    g = sim.mundur["g"]
    J = {1: "1. Rumus Turunan Dasar", 2: "2. Backward di Lapisan Dense",
         3: "3. Backward di Dalam Sel, t = 2", 4: "4. Mengirim Kesalahan ke t = 1 (Inti Through Time)",
         5: "5. Backward di Dalam Sel, t = 1", 6: "6. Gradien Total Setiap Bobot dan Bias",
         7: "7. Pemeriksaan Gradien dengan Turunan Numerik"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini melanjutkan {tautan('LSTM', 'forward_loss')}. Loss model mini sudah
diketahui, L = {a(sim.maju['L'])}. Sekarang kesalahan itu ditelusuri **mundur** untuk
mencari gradien setiap bobot dan bias: lapisan dense dulu, lalu ke dalam sel pada t = 2,
lalu mundur ke t = 1 (*through time*).
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _rumus_turunan(d, "LSTM")

    d.teks(f"## {J[2]}")
    _dense_mundur(d, sim)

    d.teks(f"## {J[3]}")
    d.teks("""
Sinyal ∂L/∂h₂ masuk ke sel lewat dua rumus forward yang diturunkan:
`h₂ = o₂ × tanh(c₂)` (menuju output gate dan cell state), lalu
`c₂ = f₂ × c₁ + i₂ × c̃₂` (menuju forget gate, input gate, dan kandidat). Setiap gerbang
lalu melewati turunan aktivasinya (σ′ atau tanh′) sehingga diperoleh sinyal kesalahan
gerbang **δ**.
""")
    tc2 = s2["tc"]
    d.kode([
        f"Masukan dari bagian 2: ∂L/∂h₂ = {a(r2['dh'])}",
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

    d.teks(f"## {J[4]}")
    d.teks("""
Kesalahan di t = 2 sebagian disebabkan oleh keadaan hari sebelumnya, jadi sinyalnya
dikirim mundur ke t = 1 lewat **dua jalur**:

1. **Jalur hidden state h₁.** h₁ dipakai oleh keempat gerbang di t = 2 (lewat bobot U),
   jadi keempat kontribusinya dijumlahkan (aturan penjumlahan).
2. **Jalur cell state c₁.** Dari `c₂ = f₂ × c₁ + ...`, turunannya terhadap c₁ adalah f₂.
   Jalur ini hanya berupa perkalian sederhana, itulah sebabnya disebut "jalan tol gradien"
   (Gambar 2).
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

    d.teks(f"## {J[5]}")
    d.teks("""
Di t = 1 rumusnya sama dengan bagian 3. Bedanya, cell state c₁ menerima **dua kiriman**
(dari jalur cell state dan dari h₁), lalu keduanya dijumlahkan.
""")
    d.kode([
        f"Masukan dari bagian 4: ∂L/∂h₁ = {a(r1['dh'])}, kiriman jalur cell state = {a(r1['dc_lanjut'])}",
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
        f"    Porsi jalur cell state = {a(r1['dc_lanjut'])} / {a(r1['dc'])} = {persen(sim.porsi_jalan_tol)}",
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
Sekitar **{persen(sim.porsi_jalan_tol)}** sinyal kesalahan ke c₁ datang lewat jalur cell state.
Inilah wujud nyata alasan LSTM tahan terhadap *vanishing gradient*: di jalur ini gradien
hanya dikalikan f, bukan melewati tanh dan matriks bobot berulang kali seperti pada RNN
biasa.
""")

    d.teks(f"## {J[6]}")
    d.teks(r"""
Bobot yang sama dipakai di t = 1 **dan** t = 2, sehingga kontribusi kedua time step
dijumlahkan (aturan penjumlahan). Dari $a_t = W x_t + U h_{t-1} + b$ diperoleh
$\partial a/\partial W = x_t$, $\partial a/\partial U = h_{t-1}$, dan
$\partial a/\partial b = 1$, sehingga:

$$\frac{\partial L}{\partial W} = \sum_t \delta_t\,x_t \qquad
\frac{\partial L}{\partial U} = \sum_t \delta_t\,h_{t-1} \qquad
\frac{\partial L}{\partial b} = \sum_t \delta_t$$
""")
    baris = []
    for gb, judul in (("f", "Forget gate"), ("i", "Input gate"), ("c", "Kandidat (c̃)"), ("o", "Output gate")):
        d1, d2 = r1["delta_" + gb], r2["delta_" + gb]
        sim_ = "δc̃" if gb == "c" else "δ" + gb
        baris.append(f"{judul}:  {sim_}₁ = {a(d1)},  {sim_}₂ = {a(d2)}")
        baris.append(f"  ∂L/∂W_{gb} = {sim_}₁ × x₁ + {sim_}₂ × x₂ = {kr(d1)} × {X[0]} + {kr(d2)} × {X[1]}"
                     f" = {kr(d1 * X[0])} + {kr(d2 * X[1])} = {a(g['W_' + gb])}")
        baris.append(f"  ∂L/∂U_{gb} = {sim_}₁ × h₀ + {sim_}₂ × h₁ = {kr(d1)} × {a(s1['h_prev'])} + {kr(d2)}"
                     f" × {a(s2['h_prev'])} = {a(g['U_' + gb])}")
        baris.append(f"  ∂L/∂b_{gb} = {sim_}₁ + {sim_}₂ = {kr(d1)} + {kr(d2)} = {a(g['b_' + gb])}")
        baris.append("")
    baris += ["Lapisan dense (dari bagian 2):", f"  ∂L/∂W_y = {a(g['W_y'])}", f"  ∂L/∂b_y = {a(g['b_y'])}"]
    d.kode(baris)
    assert all(g[n] < 0 for n in p), "Teks menyatakan semua gradien LSTM negatif"
    d.teks(f"""
**Semua gradien bernilai negatif.** Prediksi terlalu rendah, dan setiap bobot awal
bernilai positif, sehingga menaikkan bobot mana pun akan menaikkan ŷ′. Karena itu
langkah Adam berikutnya akan menaikkan semua parameter ({tautan('LSTM', 'adam')}).
Perhatikan juga **∂L/∂U = δ₂ × h₁ saja**: pada t = 1 bobot U tidak mendapat gradien karena
h₀ = 0, sedangkan bias tetap mendapat sinyal dari kedua time step
({tautan('LSTM', 'bias')}).
""")

    d.teks(f"## {J[7]}")
    _pemeriksaan_numerik(d, sim)
    return d.isi()


def _bptt_gru(sim, kt):
    p = sim.p0
    s1, s2 = sim.maju["langkah"]
    r1, r2 = sim.mundur["rincian"]
    g = sim.mundur["g"]
    J = {1: "1. Rumus Turunan Dasar", 2: "2. Backward di Lapisan Dense",
         3: "3. Backward di Dalam Sel, t = 2", 4: "4. Mengirim Kesalahan ke t = 1 (Inti Through Time)",
         5: "5. Backward di Dalam Sel, t = 1", 6: "6. Gradien Total Setiap Bobot dan Bias",
         7: "7. Pemeriksaan Gradien dengan Turunan Numerik"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini melanjutkan {tautan('GRU', 'forward_loss')}. Loss model mini GRU sudah
diketahui, L = {a(sim.maju['L'])}. Sekarang kesalahan itu ditelusuri **mundur** untuk
mencari gradien setiap bobot dan bias: lapisan dense dulu, lalu ke dalam sel pada t = 2,
lalu mundur ke t = 1 (*through time*).
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _rumus_turunan(d, "GRU")

    d.teks(f"## {J[2]}")
    _dense_mundur(d, sim, rinci=False)

    d.teks(f"## {J[3]}")
    d.teks("""
Rumus forward yang diturunkan: `h₂ = z₂ × h₁ + (1 − z₂) × h̃₂` (menuju update gate dan
kandidat), lalu `h̃₂ = tanh(W_h x₂ + b_h(in) + r₂ × q₂)` (menuju reset gate dan bagian
rekuren q₂).
""")
    d.kode([
        f"Masukan dari bagian 2: ∂L/∂h₂ = {a(r2['dh'])}",
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
    assert r2["d_z"] > 0 and g["W_z"] > 0, "Teks menyatakan gradien z positif"
    d.teks(f"""
**Makna tanda gradien z.** ∂L/∂z₂ bernilai **positif** karena kandidat baru
(h̃₂ = {a(s2['hh'])}) lebih besar daripada memori lama (h₁ = {a(s2['h_prev'])}), sedangkan
prediksi masih terlalu rendah. Adam nanti akan **menurunkan** z, artinya GRU belajar
untuk lebih mempercayai informasi baru.
""")

    su = r2["suku_h"]
    d.teks(f"## {J[4]}")
    d.teks("""
h₁ dipakai di empat tempat pada t = 2: langsung di `z₂ × h₁`, di update gate (lewat
U_z), di reset gate (lewat U_r), dan di bagian rekuren q₂ (lewat U_h). Keempat
kontribusinya dijumlahkan.
""")
    d.kode([
        "  ∂L/∂h₁ = ∂L/∂h₂ × z₂ + U_z × δz₂ + U_r × δr₂ + U_h × ∂L/∂q₂",
        f"         = {kr(r2['dh'])} × {a(s2['z'])} + {bt(p['U_z'])} × {kr(r2['delta_z'])}"
        f" + {bt(p['U_r'])} × {kr(r2['delta_r'])} + {bt(p['U_h'])} × {kr(r2['d_q'])}",
        f"         = {kr(su[0])} + {kr(su[1])} + {kr(su[2])} + {kr(su[3])}",
        f"         = {a(r2['dh_prev'])}",
        f"  Porsi jalur langsung z₂ × h₁ = {a(su[0])} / {a(r2['dh_prev'])} = {persen(sim.porsi_jalan_tol)}",
    ])
    d.teks(f"""
Sekitar **{persen(sim.porsi_jalan_tol)}** sinyal mengalir lewat jalur langsung `zₜ × hₜ₋₁`. Jalur ini
adalah "jalan tol gradien" pada GRU, padanan cell state pada LSTM.
""")

    d.teks(f"## {J[5]}")
    d.kode([
        f"Masukan dari bagian 4: ∂L/∂h₁ = {a(r1['dh'])}",
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

    d.teks(f"## {J[6]}")
    d.teks(r"""
Kontribusi kedua time step dijumlahkan. Untuk kandidat, bobot $U_h$ dan bias
$b_h^{(rec)}$ berada di dalam $q_t$, sehingga gradiennya memakai
$\partial L/\partial q_t$; bobot $W_h$ dan bias $b_h^{(in)}$ memakai $\delta\tilde{h}_t$.
""")
    baris = []
    for gb, judul in (("z", "Update gate"), ("r", "Reset gate")):
        d1, d2 = r1["delta_" + gb], r2["delta_" + gb]
        baris.append(f"{judul}:  δ{gb}₁ = {a(d1)},  δ{gb}₂ = {a(d2)}")
        baris.append(f"  ∂L/∂W_{gb} = δ{gb}₁ × x₁ + δ{gb}₂ × x₂ = {kr(d1)} × {X[0]} + {kr(d2)} × {X[1]}"
                     f" = {kr(d1 * X[0])} + {kr(d2 * X[1])} = {a(g['W_' + gb])}")
        baris.append(f"  ∂L/∂U_{gb} = δ{gb}₁ × h₀ + δ{gb}₂ × h₁ = {kr(d1)} × {a(s1['h_prev'])} + {kr(d2)}"
                     f" × {a(s2['h_prev'])} = {a(g['U_' + gb])}")
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
        "Lapisan dense (dari bagian 2):",
        f"  ∂L/∂W_y = {a(g['W_y'])}",
        f"  ∂L/∂b_y = {a(g['b_y'])}",
    ]
    d.kode(baris)
    d.teks(f"""
Gradien bias masukan dan bias rekuren pada z dan r **selalu sama**, sedangkan pada
kandidat **berbeda** karena bias rekuren ikut dikalikan rₜ. Penjelasan lengkap tentang dua
bias ini ada di {tautan('GRU', 'bias')}.
""")

    d.teks(f"## {J[7]}")
    _pemeriksaan_numerik(d, sim)
    return d.isi()


# =========================================================================== #
# ISI BERKAS: LANGKAH 4, UPDATE ADAM (Tahap 7.4 dan 8.4)
# =========================================================================== #
def _konsep_adam(d):
    d.teks(r"""
Adam dijalankan **untuk setiap parameter secara terpisah**. Setiap bobot dan bias punya
m dan v miliknya sendiri, sehingga setiap parameter punya "kecepatan belajar" sendiri.
Satu kali update terdiri dari empat langkah:

**Langkah A — momen pertama (persamaan 29, kiri)**

$$m_k = \beta_1\, m_{k-1} + (1-\beta_1)\, g_k \qquad (\beta_1 = 0.9)$$

Isinya adalah **rata-rata bergerak dari arah gradien**, kira-kira merangkum ±10 gradien
terakhir karena 1/(1 − 0.9) = 10. Fungsinya seperti **momentum bola yang
menggelinding**: gradien dari satu batch bisa "berisik", dan dengan dirata-rata, arah
langkah menjadi lebih stabil.

**Langkah B — momen kedua (persamaan 29, kanan)**

$$v_k = \beta_2\, v_{k-1} + (1-\beta_2)\, g_k^2 \qquad (\beta_2 = 0.999)$$

Isinya adalah **rata-rata bergerak dari besarnya gradien (dikuadratkan)**, kira-kira
merangkum ±1,000 gradien terakhir. Fungsinya **mengukur seberapa besar atau bergejolak
gradien parameter itu**; nilainya dipakai sebagai pembagi di langkah D.

**Langkah C — koreksi bias (persamaan 30)**

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad \hat{v}_k = \frac{v_k}{1-\beta_2^k}$$

m dan v dimulai dari **nol**, sehingga di awal pelatihan nilainya "tertarik" ke nol.
Pembagi (1 − βᵏ) mengoreksinya, dan pengaruhnya hilang setelah banyak iterasi
(bagian 6).

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


def _contoh_satu_parameter(d):
    theta0, gradien = 0.5, [0.2, 0.1]
    h = adam_satu_parameter(theta0, gradien)
    h100 = adam_satu_parameter(theta0, [100 * g for g in gradien])
    assert all(abs(x["langkah"] - y["langkah"]) < 1e-8 for x, y in zip(h, h100))
    i1, i2 = h
    d.teks(f"Sebelum dipakai pada model mini, perhatikan satu bobot bernilai awal θ₀ = {theta0}, "
           f"dengan gradien iterasi 1 g₁ = {gradien[0]} dan iterasi 2 g₂ = {gradien[1]}:")
    d.tabel(["Tahap", f"Iterasi 1 (g = {gradien[0]})", f"Iterasi 2 (g = {gradien[1]})"],
            [["m = 0.9 m_lama + 0.1 g", f"0.9 × 0 + 0.1 × {gradien[0]} = {a(i1['m'])}",
              f"0.9 × {a(i1['m'])} + 0.1 × {gradien[1]} = {a(i2['m'])}"],
             ["v = 0.999 v_lama + 0.001 g²", f"0.001 × {gradien[0]}² = {a(i1['v'], 8)}",
              f"0.999 × {a(i1['v'], 8)} + 0.001 × {gradien[1]}² = {a(i2['v'], 8)}"],
             ["m̂ = m / (1 − 0.9ᵏ)", f"{a(i1['m'])} / 0.1 = {a(i1['mh'])}", f"{a(i2['m'])} / 0.19 = {a(i2['mh'])}"],
             ["v̂ = v / (1 − 0.999ᵏ)", f"{a(i1['v'], 8)} / 0.001 = {a(i1['vh'])}",
              f"{a(i2['v'], 8)} / 0.001999 = {a(i2['vh'])}"],
             ["√v̂", a(i1["akar"]), a(i2["akar"])],
             ["Langkah η × m̂ / (√v̂ + ε)", f"0.001 × {a(i1['mh'])} / {a(i1['akar'])} = {a(i1['langkah'], 7)}",
              f"0.001 × {a(i2['mh'])} / {a(i2['akar'])} = {a(i2['langkah'], 7)}"],
             ["θ baru = θ lama − langkah", f"{theta0} − {a(i1['langkah'], 7)} = {a(i1['theta'])}",
              f"{a(i1['theta'])} − {a(i2['langkah'], 7)} = {a(i2['theta'])}"]])
    d.teks(f"""
**Bukti "tidak dipengaruhi penskalaan gradien".** Jika gradiennya **100 kali lebih besar**
(g₁ = {100 * gradien[0]:g}, g₂ = {100 * gradien[1]:g}), langkahnya tetap
**{a(h100[0]['langkah'], 7)} dan {a(h100[1]['langkah'], 7)}**, sama persis. Rasio m̂/√v̂
menghapus skala gradien. Sebagai pembanding, SGD biasa (langkah = η × g) akan melangkah
100 kali lebih jauh.
""")


def _adam_iterasi_1(d, sim, contoh):
    latih = sim.latih
    rw1 = latih["riwayat"][0]
    p_awal = sim.p0
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
**Pola penting pada iterasi pertama.** Karena m dan v diawali 0, koreksi bias membuat
m̂₁ = g₁ dan v̂₁ = g₁², sehingga m̂₁/√v̂₁ = ±1. Akibatnya **setiap parameter bergeser
sebesar 0.001** (= η) berlawanan arah dengan tanda gradiennya, berapa pun besar
gradiennya. Selisih kecil pada digit ke-7 (misalnya pada {nama(kecil)}) berasal dari ε
yang ditambahkan ke penyebut. {arah}

Inilah maksud "Adam tidak dipengaruhi penskalaan gradien". Sebagai pembanding, SGD biasa
(Δ = η × g) akan menggeser {nama(besar)} sebesar {a(LR * abs(g1[besar]), 7)} tetapi
{nama(kecil)} hanya {a(LR * abs(g1[kecil]), 7)}, yaitu
{ribu(round(abs(g1[besar]) / abs(g1[kecil])))} kali lebih kecil. Adam menyamakan kecepatan
belajar semua parameter.
""")
    d.tabel(["Parameter", "Nilai lama", "Gradien g₁", "m₁", "v₁", "Δ = η·m̂₁/(√v̂₁+ε)", "Nilai baru"],
            [[nama(n), a(rw1["p"][n]), a(rw1["g"][n]), a(rw1["m"][n], 7), a(rw1["v"][n], 10),
              a(rw1["delta"][n], 7), a(rw1["p"][n] - rw1["delta"][n])] for n in p_awal])


def _forward_ulang(d, sim):
    latih = sim.latih
    assert latih["L"][1] < latih["L"][0]
    d.kode([
        "Dengan bobot setelah iterasi k = 1, forward pass (langkah 1) diulang:",
        f"  ŷ' = {a(latih['yhat'][1])}     (sebelumnya {a(latih['yhat'][0])})",
        f"  L  = ({Y} - {a(latih['yhat'][1])})² = {a(latih['L'][1])}     (sebelumnya {a(latih['L'][0])})",
        f"  Loss turun {a(latih['L'][0] - latih['L'][1])} hanya dengan satu kali update.",
    ])


def _adam_iterasi_2(d, sim, contoh):
    rw2 = sim.latih["riwayat"][1]
    d.teks("""
Siklus langkah 1-4 diulang dengan bobot baru: forward pass, loss, BPTT menghasilkan
gradien g₂, lalu Adam. Mulai iterasi kedua, m dan v tidak lagi nol, sehingga langkah
Adam merupakan **campuran** gradien sekarang dan gradien sebelumnya. Berikut perhitungan
Adam untuk dua parameter yang sama:
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
    d.teks("""
m₂ menggabungkan arah gradien iterasi 1 dan 2. Jika suatu saat gradien berbalik arah
(misalnya ketika bobot melewati titik minimum), m akan mengecil dan langkahnya otomatis
melambat.
""")


def _koreksi_bias(d, kt):
    d.teks("""
Pembagi koreksi bias makin lama makin mendekati 1, sehingga pengaruhnya hilang. Koreksi
m hanya penting di epoch pertama, sedangkan koreksi v masih berpengaruh sampai puluhan
epoch (angka epoch memakai jumlah iterasi per epoch penelitian):
""")
    ie = kt["iter_epoch"]
    titik = [(1, "iterasi pertama"), (2, "iterasi kedua"), (ie, "akhir epoch 1"),
             (1000, f"± epoch {round(1000 / ie)}")]
    titik += [(ie * e, f"akhir epoch {ribu(e)}") for e in sorted(kt["epoch_grid"])]
    d.tabel(["Iterasi k", "Keterangan", "1 − 0.9ᵏ (pembagi m)", "1 − 0.999ᵏ (pembagi v)"],
            [[ribu(kk), ket, a(1 - B1 ** kk), a(1 - B2 ** kk)] for kk, ket in titik])


def _perjalanan_loss(d, sim):
    latih, p_awal = sim.latih, sim.p0
    kolom_bias = [n for n in p_awal if n.startswith("b_")][:1]
    d.teks(f"""
Langkah 1-4 diulang terus. Tabel berikut mencatat keadaan setelah sejumlah update. Bobot
berubah perlahan karena setiap langkah hanya sekitar 0.001; kolom b_y dan
{nama(kolom_bias[0])} memperlihatkan bias ikut dipelajari seperti bobot.
""")
    d.tabel(["Setelah k update", "Loss L", "Prediksi ŷ′", "b_y"] + [nama(n) for n in kolom_bias],
            [[ribu(j), fl(latih["L"][j]), a(latih["yhat"][j]), a(latih["p"][j]["b_y"])]
             + [a(latih["p"][j][n]) for n in kolom_bias] for j in TITIK_RIWAYAT])
    akhir = latih["p"][-1]
    d.teks(f"Nilai seluruh parameter sebelum dan sesudah {ITERASI} iterasi:")
    d.tabel(["Parameter", "Awal", f"Setelah {ITERASI} iterasi", "Perubahan"],
            [[nama(n), bt(p_awal[n]), a(akhir[n]), f"{akhir[n] - p_awal[n]:+.6f}"] for n in p_awal])


def _mini_ke_penelitian(d, sim, kt):
    model = sim.model
    nf = kt["n_fitur"]
    d.tabel(["Aspek", "Model mini", "Model penelitian"],
            [["Masukan per time step", "1 angka", f"vektor {nf} fitur"],
             ["Time step (BPTT mundur sejauh)", "2", str(kt["lookback"])],
             ["Bobot per gerbang", "angka tunggal", f"matriks W ({nf} × n_u) dan U (n_u × n_u), vektor b (n_u)"],
             ["Loss per iterasi", "1 sampel", f"rata-rata {kt['batch']} sampel (1 batch)"],
             ["Jumlah iterasi", ribu(ITERASI),
              f"{kt['iter_epoch']} per epoch × epoch grid ({', '.join(ribu(e) for e in sorted(kt['epoch_grid']))})"]])
    d.teks(f"Jumlah parameter {model} penelitian untuk setiap jumlah neuron pada grid "
           f"(persamaan {'22' if model == 'LSTM' else '27'}, termasuk dense):")
    d.tabel(["Neuron n_u"] + [str(n) for n in kt["neuron_grid"]],
            [["Parameter"] + [ribu(jumlah_parameter(model, n, nf)) for n in kt["neuron_grid"]]])
    d.teks("""
Yang sama: urutan forward → loss → BPTT → Adam, rumus setiap langkah, dan pengaturan
Adam. Setiap parameter (termasuk setiap elemen matriks) punya gradien, m, dan v sendiri.
""")
    s2 = sim.maju["langkah"][-1]
    yh = sim.maju["yhat"]
    h2b, y2b = 0.30, 0.65
    yh2 = sim.p0["W_y"] * h2b + sim.p0["b_y"]
    d1, d2 = -(2 / 2) * (Y - yh), -(2 / 2) * (y2b - yh2)
    d.teks(r"""
**Jika memakai batch (N > 1).** Persamaan (28) memakai rata-rata
$\mathcal{L} = \frac{1}{N}\sum_k (y'_k - \hat{y}'_k)^2$. Setiap $\hat{y}'_k$ hanya muncul di
suku ke-$k$, sehingga:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}'_k} = -\frac{2}{N}\,(y'_k - \hat{y}'_k) \qquad
\frac{\partial \mathcal{L}}{\partial W_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}\,h_{T,k} \qquad
\frac{\partial \mathcal{L}}{\partial b_y} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{y}'_k}$$

Setiap sampel menerima sinyal $\partial\mathcal{L}/\partial h_{T,k} =
\partial\mathcal{L}/\partial\hat{y}'_k \cdot W_y$ yang menjalankan BPTT-nya sendiri, lalu
gradien semua sampel dijumlahkan sebelum Adam dipanggil sekali.
""")
    d.kode([
        f"Contoh N = 2 (sampel 1 = model mini {model}; sampel 2 = ilustrasi h_T = 0.30, y' = 0.65)",
        f"  ŷ'₁ = {a(yh)},   ŷ'₂ = {bt(sim.p0['W_y'])} × 0.30 + {bt(sim.p0['b_y'])} = {a(yh2)}",
        f"  ∂L/∂ŷ'₁ = -(2/2) × ({Y} - {a(yh)}) = {a(d1)}",
        f"  ∂L/∂ŷ'₂ = -(2/2) × ({y2b} - {a(yh2)}) = {a(d2)}",
        f"  ∂L/∂W_y = {kr(d1)} × {a(s2['h'])} + {kr(d2)} × {h2b} = {kr(d1 * s2['h'])} + {kr(d2 * h2b)}"
        f" = {a(d1 * s2['h'] + d2 * h2b)}",
        f"  ∂L/∂b_y = {kr(d1)} + {kr(d2)} = {a(d1 + d2)}",
    ])
    d.teks("""
**Versi vektor pada lapisan dense.** Pada model penelitian, h_T dan W_y masing-masing
berisi n_u elemen, sehingga ŷ′ = Σⱼ h_T,ⱼ·W_y,ⱼ + b_y. Rumus dense pada BPTT berlaku untuk
setiap elemen j: ∂L/∂W_y,ⱼ = Σₖ ∂L/∂ŷ′ₖ·h_T,ₖ,ⱼ. Hasilnya n_u gradien bobot + 1 gradien
bias, yaitu suku (n_u + 1) pada persamaan 22 dan 27. Sinyal yang masuk ke neuron j adalah
∂L/∂h_T,ⱼ = ∂L/∂ŷ′·W_y,ⱼ.

**Batasan simulasi.** Loss model mini turun hampir ke nol karena hanya ada **1 sampel**,
sehingga model bisa "menghafal" targetnya; pada data penelitian loss berhenti di nilai
yang lebih besar karena model harus menemukan pola umum. Angka akhir simulasi juga
**tidak dapat dipakai untuk membandingkan** LSTM dan GRU, karena bobot awalnya dipilih bulat
agar mudah dihitung, sedangkan Keras memakai bobot acak. Perbandingan yang sah adalah hasil
Tahap 9 sampai 12.
""")


def _ringkasan_siklus(d, sim):
    ll = sim.latih
    jalur = ("cell state" if sim.model == "LSTM" else "z × h_(t-1)")
    d.kode([
        f"SATU SIKLUS PELATIHAN {sim.model}",
        "",
        f" 1. Inisialisasi : bobot acak, bias 0{' (forget = 1)' if sim.model == 'LSTM' else ''}, Adam m = v = 0",
        f" 2. Forward      : x₁ → gerbang → h₁{' (dan c₁)' if sim.model == 'LSTM' else ''} → x₂ → ... → h_T"
        " → ŷ' = W_y h_T + b_y",
        " 3. Loss         : L = (y' - ŷ')²  (rata-rata satu batch)",
        " 4. Dense        : ∂L/∂ŷ' = -2(y' - ŷ') → ∂L/∂W_y, ∂L/∂b_y, ∂L/∂h_T",
        " 5. Sel, t = T   : ∂L/∂h_T → turunan setiap gerbang → δ setiap gerbang",
        f" 6. Through time : δ dikirim ke t-1 lewat U dan lewat jalur {jalur}",
        " 7. Ulangi 5-6 sampai t = 1",
        " 8. Gradien      : ∂L/∂W = Σ δx,  ∂L/∂U = Σ δh_(t-1),  ∂L/∂b = Σ δ",
        " 9. Adam         : m, v → m̂, v̂ → θ baru = θ - η m̂/(√v̂ + ε)",
        "10. Kembali ke langkah 2 dengan batch berikutnya",
        "",
        f"HASIL SIMULASI {sim.model}: loss {a(ll['L'][0])} → {a(ll['L'][1])} (k = 1) → {fl(ll['L'][-1])}"
        f" (k = {ITERASI}); ŷ' {a(ll['yhat'][0])} → {a(ll['yhat'][-1])}",
    ])
    cek = [f"Gradien BPTT manual = gradien numerik untuk {len(sim.p0)} parameter (selisih maksimum "
           f"{sim.selisih:.1e}).",
           "Langkah Adam pada iterasi k = 1 bernilai ±0.001 untuk semua parameter.",
           f"Loss turun setelah satu update dan berakhir di bawah 0.0001 setelah {ITERASI} iterasi."]
    if sim.model == "GRU":
        akhir = sim.latih["p"][-1]
        assert akhir["b_z_in"] == akhir["b_z_rec"] and akhir["b_r_in"] == akhir["b_r_rec"]
        cek.append("Bias masukan dan bias rekuren z serta r tetap kembar selama 500 iterasi.")
    d.teks("**Pemeriksaan otomatis yang lolos saat berkas ini dibuat:**")
    d.teks("\n".join(f"- {c}" for c in cek))


def adam(sim: Simulasi, kt: dict) -> str:
    model = sim.model
    contoh = ("W_c", "b_y") if model == "LSTM" else ("W_h", "b_y")
    if model == "LSTM":
        J = {1: "1. Adam: Alur Satu Kali Update", 2: "2. Contoh Angka: 1 Parameter, 2 Iterasi",
             3: "3. Update Adam pada Model Mini, Iterasi k = 1", 4: "4. Forward Ulang dengan Bobot Baru",
             5: "5. Iterasi k = 2: Momentum Mulai Bekerja", 6: "6. Koreksi Bias Seiring Iterasi",
             7: "7. Siklus Diulang: Perjalanan Loss", 8: "8. Dari Model Mini ke Model Penelitian",
             9: "9. Mengapa Adam Cocok untuk Penelitian Ini", 10: "10. Ringkasan Siklus dan Hasil Verifikasi",
             11: "11. Catatan untuk Naskah Skripsi"}
    else:
        J = {1: "1. Rumus Adam", 3: "2. Update Adam pada Model Mini, Iterasi k = 1",
             4: "3. Forward Ulang dengan Bobot Baru", 5: "4. Iterasi k = 2: Momentum Mulai Bekerja",
             7: "5. Siklus Diulang: Perjalanan Loss", 8: "6. Dari Model Mini ke Model Penelitian",
             10: "7. Ringkasan Siklus dan Hasil Verifikasi"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini melanjutkan {tautan(model, 'bptt')}. Gradien setiap bobot dan bias sudah
diketahui; sekarang **Adam** memakai gradien itu untuk menggeser setiap parameter. Setelah
itu forward pass diulang untuk membuktikan loss turun, lalu siklusnya diulang sampai
{ITERASI} iterasi.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    if model == "LSTM":
        _konsep_adam(d)
        d.teks(f"## {J[2]}")
        _contoh_satu_parameter(d)
    else:
        d.teks(f"""
Rumus Adam untuk GRU **sama persis** dengan LSTM (persamaan 29-31, η = {LR}, β₁ = {B1},
β₂ = {B2}, ε = 10⁻⁷). Makna setiap langkah A-D, contoh satu parameter, dan alasan Adam
cocok untuk penelitian ini dijelaskan di {tautan('LSTM', 'adam')}. Ringkasnya:
""")
        d.teks(r"""
$$m_k = \beta_1 m_{k-1} + (1-\beta_1)\,g_k \qquad
v_k = \beta_2 v_{k-1} + (1-\beta_2)\,g_k^2$$

$$\hat{m}_k = \frac{m_k}{1-\beta_1^k} \qquad
\hat{v}_k = \frac{v_k}{1-\beta_2^k} \qquad
\theta_k = \theta_{k-1} - \eta\,\frac{\hat{m}_k}{\sqrt{\hat{v}_k}+\epsilon}$$
""")

    d.teks(f"## {J[3]}")
    _adam_iterasi_1(d, sim, contoh)
    if model == "GRU":
        d.teks("""
**Makna gerak z.** Adam **menurunkan** bobot dan bias update gate (gradiennya positif),
sehingga zₜ mengecil dan GRU mengambil lebih banyak kandidat baru h̃ₜ. Ini masuk akal:
kandidat baru lebih besar daripada memori lama, sedangkan prediksi masih terlalu rendah.
""")

    d.teks(f"## {J[4]}")
    _forward_ulang(d, sim)

    d.teks(f"## {J[5]}")
    _adam_iterasi_2(d, sim, contoh)
    if model == "LSTM":
        d.teks(f"## {J[6]}")
        _koreksi_bias(d, kt)
    else:
        d.teks(f"Tabel koreksi bias (1 − βᵏ) di {tautan('LSTM', 'adam', '6. Koreksi Bias Seiring Iterasi')} "
               "berlaku sama untuk GRU.")

    d.teks(f"## {J[7]}")
    _perjalanan_loss(d, sim)
    if model == "GRU":
        akhir = sim.latih["p"][-1]
        d.teks(f"""
Perhatikan pasangan bias: **b_z(in) = b_z(rec) = {a(akhir['b_z_in'])}** dan
**b_r(in) = b_r(rec) = {a(akhir['b_r_in'])}** tetap kembar sampai akhir, sedangkan
**b_h(in) = {a(akhir['b_h_in'])}** dan **b_h(rec) = {a(akhir['b_h_rec'])}** berbeda.
{tautan('GRU', 'bias')} menjelaskan sebabnya.
""")

    d.teks(f"## {J[8]}")
    _mini_ke_penelitian(d, sim, kt)

    if model == "LSTM":
        g1 = sim.latih["riwayat"][0]["g"]
        rasio = max(abs(v) for v in g1.values()) / min(abs(v) for v in g1.values())
        d.teks(f"## {J[9]}")
        d.teks(f"""
- Adam menggabungkan dua ide: **momentum** (dari m) dan **langkah adaptif per parameter**
  (dari v, ide RMSProp/AdaGrad; Kingma & Ba, 2015).
- Gradien dari batch data kripto yang fluktuatif cenderung berisik; momentum meredamnya.
- Parameter LSTM sangat beragam (bobot gerbang, bobot kandidat, bias, dense). Langkah
  adaptif membuat semuanya bisa belajar dengan kecepatan wajar tanpa harus mengatur
  learning rate satu per satu: di bagian 3 terlihat gradien terbesar dan terkecil berbeda
  {ribu(round(rasio))} kali, tetapi semua parameter tetap bergeser ±0.001 pada iterasi pertama.
- Pengaturan Adam **identik** untuk LSTM dan GRU (η = {kt['lr']:g}, β₁ = {B1}, β₂ = {B2},
  ε = 10⁻⁷, batch {kt['batch']}, seed sama), sehingga perbedaan hasil hanya berasal dari
  arsitektur.
""")

    d.teks(f"## {J[10]}")
    _ringkasan_siklus(d, sim)

    if model == "LSTM":
        d.teks(f"## {J[11]}")
        d.teks(f"""
1. **Keterangan N pada persamaan (28).** Saat pelatihan, loss dihitung per *mini-batch*,
   jadi N = {kt['batch']} (batch terakhir {kt['batch_akhir']}), bukan seluruh sampel.
   Saran kalimat:

   > Saat pelatihan, ℒ dihitung pada setiap mini-batch berukuran N = {kt['batch']},
   > sedangkan loss yang dilaporkan per epoch merupakan rata-rata loss seluruh mini-batch.

2. **Arti iterasi ke-k pada Adam.** Satu iterasi adalah satu kali update per batch, bukan
   per epoch. Saran kalimat:

   > Satu iterasi k bersesuaian dengan satu mini-batch, sehingga dengan
   > {ribu(kt['n_latih'])} sampel latih dan batch size {kt['batch']} terdapat
   > {kt['iter_epoch']} iterasi per epoch.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: MEMBACA KURVA LOSS (Tahap 7.10 dan 8.10)
# =========================================================================== #
def kurva_loss(model: str, kt: dict, hasil: dict) -> str:
    k = hasil["kurva"]
    J = {1: "1. Apa yang Digambar Kurva Loss", 2: f"2. Kurva Loss Model {model} Terbaik",
         3: "3. Cara Membacanya", 4: "4. Mengapa Jumlah Epoch Dipilih Lewat Data Validasi",
         5: "5. Hubungan MSE dengan RMSE dalam USD", 6: "6. Ringkasan: Kapan Loss dan Adam Bekerja"}
    if model == "LSTM":
        J[7] = "7. Catatan untuk Naskah Skripsi"
    d = Dokumen()
    d.teks(f"""
Simulasi pada {tautan(model, 'forward_loss')} sampai {tautan(model, 'adam')} memperlihatkan
satu siklus pelatihan pada model mini. Subbagian ini membaca hasil siklus yang sama pada
**model {model} penelitian yang sebenarnya** ({hasil['neuron']} neuron, {ribu(hasil['epoch'])} epoch),
yaitu kurva loss latih dan loss validasi.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks(f"""
Di akhir setiap epoch, Keras mencatat dua angka:

- **loss latih**: rata-rata MSE dari {kt['iter_epoch']} batch selama epoch itu
  ({kt['iter_epoch'] - 1} batch × {kt['batch']} sampel + 1 batch × {kt['batch_akhir']} sampel);
- **loss validasi**: MSE pada {kt['n_val']} sampel validasi memakai bobot akhir epoch.
  Angka ini **hanya diukur; tidak ada update bobot** dari data validasi.

Keduanya dihitung pada skala ternormalisasi (persamaan 28). Selama {ribu(hasil['epoch'])} epoch,
Adam melakukan {kt['iter_epoch']} × {ribu(hasil['epoch'])} = **{ribu(kt['iter_epoch'] * hasil['epoch'])}
kali update**.
""")

    d.teks(f"## {J[2]}")
    if hasil.get("gambar_loss"):
        d.teks(f"![Kurva loss {model} terbaik]({hasil['gambar_loss']})")
    turun = 1 - k["latih_akhir"] / k["latih_awal"]
    d.tabel(["Besaran", "Nilai"],
            [["Loss latih epoch 1", f"{k['latih_awal']:.8f}"],
             [f"Loss latih epoch terakhir ({hasil['epoch']})", f"{k['latih_akhir']:.8f}"],
             ["Penurunan loss latih", persen(turun)],
             ["Loss validasi epoch 1", f"{k['val_awal']:.8f}"],
             [f"Loss validasi epoch terakhir ({hasil['epoch']})", f"{k['val_akhir']:.8f}"],
             ["Loss validasi minimum", f"{k['val_min']:.8f} (epoch {k['epoch_min']})"]])

    d.teks(f"## {J[3]}")
    if k["epoch_min"] == hasil["epoch"]:
        posisi = "tepat di epoch terakhir, jadi bobot yang dipakai adalah bobot dengan loss validasi terkecil"
    elif hasil["epoch"] - k["epoch_min"] <= 0.05 * hasil["epoch"]:
        posisi = (f"di epoch {k['epoch_min']}, sangat dekat dengan epoch terakhir; loss validasi akhir "
                  f"({k['val_akhir']:.8f}) hampir sama dengan minimumnya")
    else:
        posisi = (f"lebih awal, di epoch {k['epoch_min']}, lalu naik sedikit sampai epoch terakhir "
                  f"({k['val_min']:.8f} menjadi {k['val_akhir']:.8f})")
    d.teks(f"""
- Jika **loss latih dan loss validasi turun bersama**, model sedang mempelajari pola yang
  benar.
- Jika **loss latih terus turun tetapi loss validasi naik**, itu tanda *overfitting*: model
  mulai menghafal data latih.
- Pada model ini, loss latih turun {persen(turun)} dari epoch pertama. Loss validasi mencapai
  minimum {posisi}.
""")

    d.teks(f"## {J[4]}")
    per = hasil["per_epoch"]
    d.teks(f"RMSE validasi model {hasil['neuron']} neuron untuk setiap jumlah epoch pada grid "
           f"(tabel tuning {model}):")
    d.tabel(["Epoch", "RMSE validasi (USD)"],
            [[ribu(e), usd(per[e]) + (" ← terbaik" if e == hasil["epoch"] else "")] for e in sorted(per)])
    e_maks = max(per)
    if e_maks == hasil["epoch"]:
        kalimat = (f"Pada neuron ini, epoch terbanyak ({ribu(e_maks)}) memberi RMSE validasi terkecil, "
                   "sehingga pelatihan yang lebih lama masih membantu.")
    elif per[e_maks] > per[hasil["epoch"]]:
        kalimat = (f"Pada neuron ini, melatih sampai {ribu(e_maks)} epoch justru memperburuk RMSE validasi "
                   f"dibandingkan {ribu(hasil['epoch'])} epoch. Pelatihan yang lebih lama tidak selalu lebih "
                   "baik; kemungkinan besar model mulai terlalu menyesuaikan diri dengan data latih.")
    else:
        kalimat = "RMSE validasi berbeda-beda untuk setiap jumlah epoch."
    d.teks(f"{kalimat} Itulah gunanya data validasi untuk memilih jumlah epoch, sedangkan data uji "
           "tetap netral.")

    d.teks(f"## {J[5]}")
    rmse = kt["rentang"] * math.sqrt(k["val_akhir"])
    assert abs(rmse / hasil["rmse_val"] - 1) < 1e-3, "Rentang × √MSE harus sama dengan RMSE validasi"
    d.teks(f"""
Karena normalisasi min-max bersifat linear (persamaan 6 dan 8), selisih dalam USD sama
dengan selisih ternormalisasi dikali (x_max − x_min). Akibatnya:

**RMSE (USD) = (x_max − x_min) × √MSE**

Dengan rentang harga data latih {usd(kt['x_max'])} − {usd(kt['x_min'])} = {usd(kt['rentang'])}:
""")
    d.kode([f"RMSE = {usd(kt['rentang'])} × √{k['val_akhir']:.8f}",
            f"     = {usd(kt['rentang'])} × {math.sqrt(k['val_akhir']):.6f}",
            f"     = {usd(rmse)} USD",
            f"RMSE validasi pada tabel tuning = {usd(hasil['rmse_val'])} USD"])
    d.teks("""
Jadi **meminimalkan MSE saat pelatihan sama artinya dengan meminimalkan RMSE dalam USD**.
Kalimat ini berguna untuk menjawab pertanyaan "mengapa loss-nya MSE tetapi evaluasinya
RMSE?".
""")

    d.teks(f"## {J[6]}")
    d.tabel(["Tahap penelitian", "Loss MSE", "Adam"],
            [["Inisialisasi model", "-", "m = 0, v = 0, k = 0"],
             [f"Setiap batch latih ({kt['iter_epoch']}× per epoch)", "dihitung, menjadi sumber gradien",
              "memperbarui semua bobot dan bias"],
             ["Akhir setiap epoch", "loss latih dan loss validasi dicatat (kurva loss)",
              "tidak ada update dari data validasi"],
             ["Pemilihan neuron dan epoch terbaik", "tidak (memakai RMSE validasi USD, setara √MSE × rentang)",
              "tidak"],
             ["Prediksi data uji dan evaluasi", "tidak", "tidak (bobot sudah beku)"]])

    if model == "LSTM":
        d.teks(f"## {J[7]}")
        d.teks("""
**Opsional, hubungan loss dan evaluasi** (subbab 1.5.11 atau 1.5.12). Saran kalimat:

> Karena normalisasi min-max bersifat linear, RMSE dalam USD sama dengan
> (x_max − x_min) × √MSE, sehingga meminimalkan MSE pada skala ternormalisasi setara
> dengan meminimalkan RMSE dalam USD.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: BIAS (Tahap 7.12 dan 8.12)
# =========================================================================== #
def _bias_mengapa(d, sim, kt, hasil):
    model = sim.model
    d.teks("""
**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Ini sering
terjadi di penelitian: h₀ = 0 di awal setiap jendela, dan normalisasi min-max membuat
fitur bernilai dekat 0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman kurva, bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:
""")
    d.tabel(["x", "Dengan bias: σ(5x − 2.5)", "Tanpa bias: σ(5x)"],
            [[f"{x:g}", a(sig(5 * x - 2.5), 3), a(sig(5 * x), 3)] for x in (0, 0.25, 0.5, 0.75, 1)])
    d.teks("""
Tanpa bias, gerbang **tidak pernah turun di bawah 0.5** untuk data ternormalisasi yang
positif, sehingga model tidak bisa menyatakan aturan "tutup gerbang saat nilai fitur
rendah, buka saat tinggi".
""")
    b = hasil["bias"]
    if model == "LSTM":
        bf = b["f"]
        d.teks(f"""
**(c) Bias menentukan perilaku bawaan setiap gerbang.** Contoh terpenting: forget gate.
Jika gerbang hanya ditentukan oleh biasnya, sisa memori setelah {kt['lookback']} hari adalah
f^{kt['lookback']}:
""")
        d.tabel(["Bias forget", "f = σ(b)", f"Memori tersisa setelah {kt['lookback']} hari"],
                [[lbl, a(sig(v), 3), persen(sig(v) ** kt["lookback"])]
                 for lbl, v in (("0 (tanpa bias)", 0.0), (f"{a(bf)} (neuron ke-1 model terlatih)", bf),
                                ("1 (inisialisasi Keras)", 1.0), ("2", 2.0))])
        d.teks("""
Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*).

Nilai bawaan setiap gerbang pada **model LSTM terlatih** (neuron ke-1, saat kontribusi
xW + hU = 0; nilai bias yang sama tampil di Tahap 7.11):
""")
        d.tabel(["Gerbang", "Bias", "Nilai bawaan", "Arti"],
                [["input i", a(b["i"]), f"σ = {a(sig(b['i']), 3)}",
                  "cenderung hemat menerima informasi baru" if sig(b["i"]) < 0.5 else "cenderung menerima informasi baru"],
                 ["forget f", a(b["f"]), f"σ = {a(sig(b['f']), 3)}",
                  "cenderung mempertahankan memori" if sig(b["f"]) > 0.5 else "cenderung membuang memori"],
                 ["kandidat c̃", a(b["c"]), f"tanh = {a(math.tanh(b['c']), 3)}",
                  "isi bawaan hampir netral" if abs(b["c"]) < 0.1 else "isi bawaan tidak netral"],
                 ["output o", a(b["o"]), f"σ = {a(sig(b['o']), 3)}",
                  "cenderung menahan sebagian keluaran" if sig(b["o"]) < 0.5 else "cenderung mengeluarkan memori"]])
    else:
        z, r = sum(b["z"]), sum(b["r"])
        d.teks("""
**(c) Bias menentukan perilaku bawaan setiap gerbang.** Pada GRU, bias update gate
menentukan campuran bawaan memori lama vs baru, dan bias reset gate menentukan seberapa
banyak masa lalu dipakai untuk kandidat. Nilai bawaan pada **model GRU terlatih** (neuron
ke-1, saat kontribusi xW + hU = 0; nilai bias yang sama tampil di Tahap 8.11):
""")
        d.tabel(["Gerbang", "Bias masukan + bias rekuren", "Nilai bawaan", "Arti"],
                [["update z", f"{a(b['z'][0])} + {kr(b['z'][1])} = {a(z)}", f"σ = {a(sig(z), 3)}",
                  f"cenderung mempertahankan {persen(sig(z), 0)} memori lama"],
                 ["reset r", f"{a(b['r'][0])} + {kr(b['r'][1])} = {a(r)}", f"σ = {a(sig(r), 3)}",
                  f"memakai sekitar {persen(sig(r), 0)} masa lalu untuk kandidat"],
                 ["kandidat h̃", f"b_h(in) = {a(b['h'][0])}, b_h(rec) = {a(b['h'][1])}", "-",
                  "b_h(rec) ikut dikalikan rₜ (bagian 4)"]])
    d.teks(f"""
Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari sesuai
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menggeser tingkat dasar prediksi.** h_T selalu berada di rentang (−1, 1),
jadi b_y yang menentukan "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada
model {model} terlatih, b_y = {a(hasil['b_y'])}, setara {a(hasil['b_y'])} × {usd(kt['rentang'])}
≈ **{usd(hasil['b_y'] * kt['rentang'])} USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih ({usd(kt['x_min'])} USD) setiap kali h_T = 0.

**(e) Bias selalu menerima sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai
nol bila masukannya nol (lihat ∂L/∂U di
{tautan(model, 'bptt', '6. Gradien Total Setiap Bobot dan Bias')}: pada t = 1 tidak ada
kontribusi karena h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa pun.
""")
    n = hasil["neuron"]
    jumlah_bias = (4 * n if model == "LSTM" else 6 * n) + 1
    d.teks(f"**(f) Biayanya kecil.** Model {model} terlatih memiliki {ribu(jumlah_bias)} bias "
           f"(termasuk b_y) dari {ribu(hasil['param'])} parameter, atau {persen(jumlah_bias / hasil['param'])}.")


def _bias_cara(d, sim):
    model = sim.model
    g, rinci, lat = sim.mundur["g"], sim.mundur["rincian"], sim.latih
    d.teks("""
Bias tidak dihitung dengan satu rumus langsung; bias **dipelajari** dengan cara yang
persis sama seperti bobot. Berikut lima langkahnya pada model mini, dengan rujukan ke
subbagian yang sudah dihitung:
""")
    if model == "LSTM":
        contoh = (f"  Contoh b_c : ∂L/∂b_c = δc̃₁ + δc̃₂ = {kr(rinci[0]['delta_c'])} + {kr(rinci[1]['delta_c'])}"
                  f" = {a(g['b_c'])}")
        nilai_awal = "Bias gerbang = 0, kecuali bias forget gate = 1. Bias dense b_y = 0."
    else:
        contoh = (f"  Contoh b_h(in)  : δh̃₁ + δh̃₂ = {kr(rinci[0]['delta_h'])} + {kr(rinci[1]['delta_h'])}"
                  f" = {a(g['b_h_in'])}\n"
                  f"  Contoh b_h(rec) : ∂L/∂q₁ + ∂L/∂q₂ = {kr(rinci[0]['d_q'])} + {kr(rinci[1]['d_q'])}"
                  f" = {a(g['b_h_rec'])}")
        nilai_awal = "Semua bias gerbang (masukan dan rekuren) = 0. Bias dense b_y = 0."
    nf, nb, na = (label_tahap(berkas(model, x)[0]) for x in ("forward_loss", "bptt", "adam"))
    d.kode([
        f"Langkah 1 — Nilai awal ({nf})",
        f"  {nilai_awal}",
        "",
        f"Langkah 2 — Dipakai di forward pass ({nf})",
        "  Gerbang : a = W × x + U × h + b",
        "  Dense   : ŷ' = W_y × h_T + b_y",
        "",
        f"Langkah 3 — Hitung gradiennya ({nb}, bagian 6)",
        "  Karena a = W × x + U × h + b, maka ∂a/∂b = 1, sehingga",
        "  ∂L/∂b = δ × 1 = δ, dijumlahkan untuk semua time step:  ∂L/∂b = Σ δ",
        f"  Contoh b_y : ∂L/∂b_y = ∂L/∂ŷ' × 1 = {a(g['b_y'])}",
        contoh,
        "",
        f"Langkah 4 — Update dengan Adam ({na})",
        f"  b_y: 0 → {a(lat['p'][1]['b_y'])} (k = 1) → {a(lat['p'][2]['b_y'])} (k = 2)",
        "",
        f"Langkah 5 — Ulangi siklus ({na}, perjalanan loss)",
        f"  Setelah {ITERASI} iterasi b_y = {a(lat['p'][-1]['b_y'])}.",
    ])
    d.teks("Ringkasan rumus gradien seluruh bias pada model mini (iterasi k = 1):")
    if model == "LSTM":
        isi = [["Dense b_y", "∂L/∂ŷ′", a(g["b_y"])], ["b_f", "δf₁ + δf₂", a(g["b_f"])],
               ["b_i", "δi₁ + δi₂", a(g["b_i"])], ["b_c", "δc̃₁ + δc̃₂", a(g["b_c"])],
               ["b_o", "δo₁ + δo₂", a(g["b_o"])]]
    else:
        isi = [["Dense b_y", "∂L/∂ŷ′", a(g["b_y"])],
               ["b_z(in) dan b_z(rec)", "keduanya δz₁ + δz₂", f"{a(g['b_z_in'])} dan {a(g['b_z_rec'])}"],
               ["b_r(in) dan b_r(rec)", "keduanya δr₁ + δr₂", f"{a(g['b_r_in'])} dan {a(g['b_r_rec'])}"],
               ["b_h(in)", "δh̃₁ + δh̃₂", a(g["b_h_in"])],
               ["b_h(rec)", "δh̃₁·r₁ + δh̃₂·r₂", a(g["b_h_rec"])]]
    d.tabel(["Bias", "Rumus gradien", "Nilai"], isi)


def bias(sim: Simulasi, kt: dict, hasil: dict) -> str:
    model = sim.model
    if model == "LSTM":
        J = {1: "1. Intinya", 2: "2. Mengapa Bias Diperlukan", 3: "3. Cara Menghitung Bias Langkah demi Langkah",
             4: "4. Catatan untuk Naskah Skripsi"}
    else:
        J = {1: "1. Intinya", 2: "2. Mengapa Bias Diperlukan", 3: "3. Cara Menghitung Bias Langkah demi Langkah",
             4: "4. Dua Bias pada GRU: Asal dan Buktinya", 5: "5. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini mengumpulkan pembahasan tentang **bias pada {model}**: mengapa bias perlu ada,
dan bagaimana bias dihitung langkah demi langkah{', serta mengapa GRU memiliki dua jenis bias' if model == 'GRU' else ''}.
Angka contohnya diambil dari simulasi model mini ({tautan(model, 'forward_loss')} sampai
{tautan(model, 'adam')}) dan dari model {model} terlatih penelitian.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks("""
Bias adalah **titik awal (intercept)** setiap gerbang dan neuron, sama seperti intercept
*a* pada regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa bias, ketika
masukannya nol, setiap gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu setengah
terbuka; tanh(0) = 0). Dengan bias, setiap gerbang dapat menentukan **posisi bawaannya
sendiri**. Bias tidak dihitung dengan satu rumus langsung; bias **dipelajari** lewat siklus
yang sama dengan bobot (bagian 3).
""")

    d.teks(f"## {J[2]}")
    _bias_mengapa(d, sim, kt, hasil)

    d.teks(f"## {J[3]}")
    _bias_cara(d, sim)

    if model == "GRU":
        b = hasil["bias"]
        akhir = sim.latih["p"][-1]
        gg = sim.mundur["g"]
        kembar = abs(b["z"][0] - b["z"][1]) < 5e-7 and abs(b["r"][0] - b["r"][1]) < 5e-7
        d.teks(f"## {J[4]}")
        d.teks(f"""
**Letaknya pada Gambar 7** ({tautan('GRU', 'alur_sel')}). Bias tidak digambar; bias berada di
dalam setiap kotak kuning (σ, σ, tanh). Setiap kotak menerima dua garis masuk, yaitu xₜ dari
bawah dan hₜ₋₁ dari garis vertikal kiri. Dengan `reset_after=True`, Keras menghitung kedua
garis itu terpisah, masing-masing dengan biasnya sendiri: garis xₜ membawa **bias masukan**
(xₜW + b(in)) dan garis hₜ₋₁ membawa **bias rekuren** (hₜ₋₁U + b(rec)).
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
        bukti_terlatih = (f"model GRU terlatih (neuron ke-1): b_z(in) = {a(b['z'][0])}, b_z(rec) = {a(b['z'][1])}; "
                          f"b_r(in) = {a(b['r'][0])}, b_r(rec) = {a(b['r'][1])}"
                          + (" (kembar)." if kembar else "."))
        d.teks(f"""
**Pada z dan r, dua bias sebenarnya berlebih.** b(in) + b(rec) langsung dijumlahkan, jadi
gradien keduanya selalu sama. Karena nilai awalnya juga sama (0), keduanya akan selalu
kembar:

- model mini: setelah {ITERASI} iterasi, b_z(in) = b_z(rec) = {a(akhir['b_z_in'])} dan
  b_r(in) = b_r(rec) = {a(akhir['b_r_in'])};
- {bukti_terlatih}

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikalikan rₜ (jika rₜ mendekati 0,
bias ini ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu aktif. Gradiennya
berbeda (Σ δh̃·r vs Σ δh̃), sehingga nilainya juga berbeda:

- model mini: gradien {a(gg['b_h_in'])} vs {a(gg['b_h_rec'])}; setelah {ITERASI} iterasi
  b_h(in) = {a(akhir['b_h_in'])} dan b_h(rec) = {a(akhir['b_h_rec'])};
- model terlatih (neuron ke-1): b_h(in) = {a(b['h'][0])} dan b_h(rec) = {a(b['h'][1])}.

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak memuat bias sama sekali
(penulisnya menyebut *"to make the equations uncluttered, we omit biases"*). Bentuk dua
bias berasal dari implementasi GRU pada pustaka **NVIDIA cuDNN**, yang memisahkan bagian
masukan dan bagian rekuren setiap gerbang. Keras memakainya sebagai bawaan
(`reset_after=True`, *"cuDNN compatible"*). LSTM di Keras cukup memakai satu bias karena
pada LSTM semua bias hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU {hasil['neuron']} neuron memiliki
2 × 3 × {hasil['neuron']} = {6 * hasil['neuron']} bias gerbang (bias Keras berbentuk
(2, {3 * hasil['neuron']})). Dengan satu bias jumlahnya hanya {3 * hasil['neuron']}, sehingga total
parameter menjadi {ribu(hasil['param'] - 3 * hasil['neuron'])}, bukan {ribu(hasil['param'])}.
""")
        assert akhir["b_z_in"] == akhir["b_z_rec"] and akhir["b_r_in"] == akhir["b_r_rec"]
        assert abs(akhir["b_h_in"] - akhir["b_h_rec"]) > 1e-3

    d.teks(f"## {J[len(J)]}")
    if model == "LSTM":
        d.teks("""
**Kalimat ringkas tentang fungsi bias** (misalnya setelah persamaan 9 di subbab 1.5.7):

> Bias berfungsi menggeser fungsi aktivasi sehingga setiap gerbang dan neuron dapat
> menentukan kondisi bawaannya sendiri dan tidak dipaksa bernilai tetap ketika masukannya
> bernilai nol, misalnya σ(0) = 0,5 atau tanh(0) = 0. Pada LSTM dan GRU, bias menentukan
> ambang dan kondisi bawaan setiap gerbang, sedangkan pada lapisan dense bias menentukan
> tingkat dasar nilai prediksi.
""")
    else:
        d.teks("""
**Penjelasan dua bias GRU** (setelah Gambar 7 atau persamaan 26):

> Pada Gambar 7, bias tidak digambarkan secara eksplisit karena termuat di dalam setiap
> lapisan (kotak σ dan tanh). Pada implementasi Keras dengan pengaturan reset_after=True,
> setiap lapisan memisahkan kontribusi masukan xₜW + b⁽ⁱⁿ⁾ dan kontribusi rekuren
> hₜ₋₁U + b⁽ʳᵉᶜ⁾, masing-masing dengan biasnya sendiri. Pada update gate dan reset gate
> kedua bias hanya dijumlahkan, sedangkan pada kandidat hidden state bias rekuren
> b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga keduanya tidak dapat digabung
> menjadi satu. Pemisahan ini mengikuti implementasi GRU pada pustaka cuDNN yang
> digunakan Keras (Chollet et al., 2015), sedangkan Cho et al. (2014) sendiri tidak
> menuliskan suku bias pada persamaannya.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: PERBANDINGAN STRUKTUR LSTM DAN GRU (Tahap 9.2)
# =========================================================================== #
def perbandingan(sim_lstm: Simulasi, sim_gru: Simulasi, kt: dict, hasil_lstm: dict, hasil_gru: dict) -> str:
    J = {1: "1. Perbedaan Struktur Sel", 2: "2. Jalan Tol Gradien pada Kedua Model",
         3: "3. Jumlah Parameter pada Jumlah Neuron yang Sama", 4: "4. Model Mini: Siklus Sama, Rumus Sel Berbeda"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini merangkum perbedaan struktur LSTM (Tahap 7) dan GRU (Tahap 8) yang sudah
dihitung terpisah, sebagai pelengkap Tabel 13 dan subbab 1.5.10 (Tabel 2). Alur sel
masing-masing dijelaskan di {tautan('LSTM', 'alur_sel')} dan {tautan('GRU', 'alur_sel')}.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.tabel(["Aspek", "LSTM", "GRU", "Penjelasan"],
            [["Gerbang", "3 (forget, input, output)", "2 (update, reset)", "GRU tidak punya output gate"],
             ["Simpan vs terima", "fₜ dan iₜ **independen**", "zₜ dan (1 − zₜ) **terikat**",
              "pada GRU, menyimpan lebih banyak yang lama berarti menerima lebih sedikit yang baru"],
             ["Memori internal", "cₜ dan hₜ", "hanya hₜ", "GRU menyimpan memori langsung di hidden state"],
             ["Peran masa lalu pada kandidat", "lewat U_c", "diatur rₜ (reset gate)",
              "tidak ada padanan langsung reset gate pada LSTM"],
             ["Himpunan bobot", "4 (i, f, c, o)", "3 (z, r, h)", "parameter GRU lebih sedikit"],
             ["Bias per himpunan (Keras)", "n_u", "2n_u", "GRU memakai bias masukan dan bias rekuren"]])

    d.teks(f"## {J[2]}")
    d.teks(f"""
Kedua model punya jalur sederhana yang membuat gradien tidak cepat mengecil saat dikirim
mundur ke time step sebelumnya:

- **LSTM**: jalur cell state (Gambar 2), gradien hanya dikalikan fₜ. Pada model mini,
  **{persen(sim_lstm.porsi_jalan_tol)}** sinyal kesalahan ke c₁ lewat jalur ini
  ({tautan('LSTM', 'bptt')}).
- **GRU**: jalur langsung zₜ ⊙ hₜ₋₁ (Gambar 7), gradien hanya dikalikan zₜ. Pada model
  mini, **{persen(sim_gru.porsi_jalan_tol)}** sinyal kesalahan ke h₁ lewat jalur ini
  ({tautan('GRU', 'bptt')}).
""")

    d.teks(f"## {J[3]}")
    nf = kt["n_fitur"]
    for hasil in (hasil_lstm, hasil_gru):
        assert jumlah_parameter(hasil["model"], hasil["neuron"], nf) == hasil["param"]
    d.teks(f"Jumlah parameter total (lapisan rekuren + dense) dengan {nf} fitur, "
           "persamaan 22 dan 27:")
    d.tabel(["Neuron n_u", "LSTM", "GRU", "GRU / LSTM"],
            [[n,
              ribu(jumlah_parameter("LSTM", n, nf)) + (" ← terbaik" if n == hasil_lstm["neuron"] else ""),
              ribu(jumlah_parameter("GRU", n, nf)) + (" ← terbaik" if n == hasil_gru["neuron"] else ""),
              persen(jumlah_parameter("GRU", n, nf) / jumlah_parameter("LSTM", n, nf))]
             for n in kt["neuron_grid"]])
    d.teks(f"""
Pada jumlah neuron yang sama, GRU selalu memerlukan parameter lebih sedikit. Model
terbaik penelitian adalah LSTM {hasil_lstm['neuron']} neuron ({ribu(hasil_lstm['param'])} parameter)
dan GRU {hasil_gru['neuron']} neuron ({ribu(hasil_gru['param'])} parameter).
""")

    d.teks(f"## {J[4]}")
    ll, lg = sim_lstm.latih, sim_gru.latih
    d.tabel(["Besaran", "LSTM mini", "GRU mini"],
            [["Jumlah parameter (n_u = 1, n_f = 1)", len(sim_lstm.p0), len(sim_gru.p0)],
             ["Prediksi awal ŷ′", a(ll["yhat"][0]), a(lg["yhat"][0])],
             ["Loss awal", a(ll["L"][0]), a(lg["L"][0])],
             ["Loss setelah 1 update", a(ll["L"][1]), a(lg["L"][1])],
             [f"Loss setelah {ITERASI} update", fl(ll["L"][-1]), fl(lg["L"][-1])]])
    d.teks("""
Urutan siklus pelatihan (forward → loss → BPTT → Adam) dan pengaturan Adam sama persis;
yang berbeda hanya rumus di dalam sel. Pada model mini dengan n_u = 1, kebetulan kedua model
sama-sama punya 14 parameter. Angka loss model mini **tidak dapat dipakai untuk
menyimpulkan** model mana yang lebih baik, karena bobot awalnya dipilih bulat dan hanya ada
satu sampel. Perbandingan yang sah adalah hasil evaluasi dan uji Diebold-Mariano pada
Tahap 11 dan 12.
""")
    return d.isi()
