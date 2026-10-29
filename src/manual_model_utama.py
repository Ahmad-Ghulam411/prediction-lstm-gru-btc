# -*- coding: utf-8 -*-
"""
Perhitungan manual pelatihan LSTM dan GRU pada MODEL UTAMA penelitian.

Modul ini dipakai notebook ``notebooks/btc_lstm_vs_gru_model_utama.ipynb``.
Berbeda dengan ``manual_pelatihan.py`` (model mini 1 neuron), semua angka di sini
berasal dari model terbaik hasil tuning: masukan 7 hari x 10 fitur, jumlah neuron
hasil tuning, batch 32, dan bobot awal asli Keras (seed 42). Satu siklus
pelatihan dihitung ulang dengan NumPy: forward pass, loss MSE, backpropagation
through time (BPTT), lalu update Adam, dan setiap hasilnya dicocokkan dengan
Keras.

Bagian atas modul berisi perhitungan numerik; bagian bawah menyusun teks
Markdown yang ditampilkan di notebook dan disimpan ke folder perhitungan manual.
"""

from __future__ import annotations

import math

import numpy as np

from manual_pelatihan import Dokumen, a, kr, persen, ribu, usd, jangkar, jumlah_parameter

B1, B2, EPS = 0.9, 0.999, 1e-7   # Adam bawaan Keras


# =========================================================================== #
# BAGIAN NUMERIK
# =========================================================================== #
def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def pecah_bobot(model: str, bobot):
    """
    Ubah ``model.get_weights()`` Keras menjadi kamus bobot per gerbang (float64).

    LSTM : kernel (F, 4u) urutan i, f, c, o; recurrent (u, 4u); bias (4u,)
    GRU  : kernel (F, 3u) urutan z, r, h; recurrent (u, 3u); bias (2, 3u)
           baris 0 = bias masukan, baris 1 = bias rekuren
    """
    W, U, b, Wy, by = [np.asarray(w, dtype=np.float64) for w in bobot]
    p = {"W_y": Wy.copy(), "b_y": by.copy()}
    if model == "LSTM":
        for j, g in enumerate("ifco"):
            u = W.shape[1] // 4
            p["W_" + g] = W[:, j * u:(j + 1) * u].copy()
            p["U_" + g] = U[:, j * u:(j + 1) * u].copy()
            p["b_" + g] = b[j * u:(j + 1) * u].copy()
    else:
        u = W.shape[1] // 3
        for j, g in enumerate("zrh"):
            p["W_" + g] = W[:, j * u:(j + 1) * u].copy()
            p["U_" + g] = U[:, j * u:(j + 1) * u].copy()
            p["b_" + g + "_in"] = b[0, j * u:(j + 1) * u].copy()
            p["b_" + g + "_rec"] = b[1, j * u:(j + 1) * u].copy()
    return p


def gabung_bobot(model: str, p):
    """Kebalikan ``pecah_bobot``: susun lagi menjadi daftar bobot ala Keras."""
    if model == "LSTM":
        urut = "ifco"
        W = np.concatenate([p["W_" + g] for g in urut], axis=1)
        U = np.concatenate([p["U_" + g] for g in urut], axis=1)
        b = np.concatenate([p["b_" + g] for g in urut])
    else:
        urut = "zrh"
        W = np.concatenate([p["W_" + g] for g in urut], axis=1)
        U = np.concatenate([p["U_" + g] for g in urut], axis=1)
        b = np.stack([np.concatenate([p["b_" + g + "_in"] for g in urut]),
                      np.concatenate([p["b_" + g + "_rec"] for g in urut])])
    return [W, U, b, p["W_y"], p["b_y"]]


def kunci_bobot(model: str):
    if model == "LSTM":
        return ([f"W_{g}" for g in "ifco"] + [f"U_{g}" for g in "ifco"]
                + [f"b_{g}" for g in "ifco"] + ["W_y", "b_y"])
    return ([f"W_{g}" for g in "zrh"] + [f"U_{g}" for g in "zrh"]
            + [f"b_{g}_in" for g in "zrh"] + [f"b_{g}_rec" for g in "zrh"] + ["W_y", "b_y"])


def lstm_maju(p, X, y=None):
    """Forward pass LSTM untuk satu batch X berukuran (B, T, F)."""
    X = np.asarray(X, dtype=np.float64)
    B, T, _ = X.shape
    u = p["U_f"].shape[0]
    h, c = np.zeros((B, u)), np.zeros((B, u))
    langkah = []
    for t in range(T):
        x = X[:, t, :]
        s = {"x": x, "h_prev": h, "c_prev": c}
        for g in "ifco":
            s["mx_" + g] = x @ p["W_" + g]           # bagian masukan x_t W
            s["mh_" + g] = h @ p["U_" + g]           # bagian rekuren h_(t-1) U
            s["a_" + g] = s["mx_" + g] + s["mh_" + g] + p["b_" + g]
        s["i"], s["f"], s["o"] = sig(s["a_i"]), sig(s["a_f"]), sig(s["a_o"])
        s["cc"] = np.tanh(s["a_c"])
        s["c"] = s["f"] * c + s["i"] * s["cc"]
        s["tc"] = np.tanh(s["c"])
        s["h"] = s["o"] * s["tc"]
        langkah.append(s)
        h, c = s["h"], s["c"]
    yhat = h @ p["W_y"] + p["b_y"]
    hasil = {"langkah": langkah, "yhat": yhat}
    if y is not None:
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        hasil["y"] = y
        hasil["L"] = float(np.mean((y - yhat) ** 2))
    return hasil


def lstm_mundur(p, maju):
    """BPTT LSTM. Mengembalikan gradien setiap bobot dan rincian sinyal galat."""
    S, y, yhat = maju["langkah"], maju["y"], maju["yhat"]
    B = y.shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dy = 2.0 * (yhat - y) / B                         # dL/dy_hat, (B, 1)
    g["W_y"] = S[-1]["h"].T @ dy
    g["b_y"] = dy.sum(axis=0)
    dh = dy @ p["W_y"].T
    dc_lanjut = np.zeros_like(dh)
    rincian = [None] * len(S)
    for t in reversed(range(len(S))):
        s = S[t]
        jalur_h = dh * s["o"] * (1 - s["tc"] ** 2)
        dc = dc_lanjut + jalur_h
        r = {"dh": dh, "dc_lanjut": dc_lanjut, "jalur_h": jalur_h, "dc": dc}
        r["d_o"] = dh * s["tc"]
        r["d_f"] = dc * s["c_prev"]
        r["d_i"] = dc * s["cc"]
        r["d_cc"] = dc * s["i"]
        r["delta_o"] = r["d_o"] * s["o"] * (1 - s["o"])
        r["delta_f"] = r["d_f"] * s["f"] * (1 - s["f"])
        r["delta_i"] = r["d_i"] * s["i"] * (1 - s["i"])
        r["delta_c"] = r["d_cc"] * (1 - s["cc"] ** 2)
        dh_prev = np.zeros_like(dh)
        for gb in "ifco":
            d = r["delta_" + gb]
            g["W_" + gb] += s["x"].T @ d
            g["U_" + gb] += s["h_prev"].T @ d
            g["b_" + gb] += d.sum(axis=0)
            r["dh_dari_" + gb] = d @ p["U_" + gb].T
            dh_prev += r["dh_dari_" + gb]
        r["dh_prev"] = dh_prev
        r["dc_prev"] = dc * s["f"]
        rincian[t] = r
        dh, dc_lanjut = dh_prev, r["dc_prev"]
    return {"dy": dy, "g": g, "rincian": rincian}


def gru_maju(p, X, y=None):
    """Forward pass GRU (Keras, reset_after=True) untuk satu batch X (B, T, F)."""
    X = np.asarray(X, dtype=np.float64)
    B, T, _ = X.shape
    u = p["U_z"].shape[0]
    h = np.zeros((B, u))
    langkah = []
    for t in range(T):
        x = X[:, t, :]
        s = {"x": x, "h_prev": h}
        for g in "zrh":
            s["mx_" + g] = x @ p["W_" + g] + p["b_" + g + "_in"]      # jalur masukan
            s["mh_" + g] = h @ p["U_" + g] + p["b_" + g + "_rec"]     # jalur rekuren
        s["a_z"] = s["mx_z"] + s["mh_z"]
        s["a_r"] = s["mx_r"] + s["mh_r"]
        s["z"], s["r"] = sig(s["a_z"]), sig(s["a_r"])
        s["rq"] = s["r"] * s["mh_h"]
        s["a_h"] = s["mx_h"] + s["rq"]
        s["hh"] = np.tanh(s["a_h"])
        s["h"] = s["z"] * h + (1 - s["z"]) * s["hh"]
        langkah.append(s)
        h = s["h"]
    yhat = h @ p["W_y"] + p["b_y"]
    hasil = {"langkah": langkah, "yhat": yhat}
    if y is not None:
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        hasil["y"] = y
        hasil["L"] = float(np.mean((y - yhat) ** 2))
    return hasil


def gru_mundur(p, maju):
    """BPTT GRU (Keras, reset_after=True)."""
    S, y, yhat = maju["langkah"], maju["y"], maju["yhat"]
    B = y.shape[0]
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dy = 2.0 * (yhat - y) / B
    g["W_y"] = S[-1]["h"].T @ dy
    g["b_y"] = dy.sum(axis=0)
    dh = dy @ p["W_y"].T
    rincian = [None] * len(S)
    for t in reversed(range(len(S))):
        s = S[t]
        r = {"dh": dh}
        r["d_z"] = dh * (s["h_prev"] - s["hh"])
        r["delta_z"] = r["d_z"] * s["z"] * (1 - s["z"])
        r["d_hh"] = dh * (1 - s["z"])
        r["delta_h"] = r["d_hh"] * (1 - s["hh"] ** 2)
        r["d_r"] = r["delta_h"] * s["mh_h"]
        r["delta_r"] = r["d_r"] * s["r"] * (1 - s["r"])
        r["d_q"] = r["delta_h"] * s["r"]               # dL/d(h_(t-1) U_h + b_h(rec))
        dmx = {"z": r["delta_z"], "r": r["delta_r"], "h": r["delta_h"]}
        dmh = {"z": r["delta_z"], "r": r["delta_r"], "h": r["d_q"]}
        r["dh_langsung"] = dh * s["z"]                 # jalur langsung z_t * h_(t-1)
        dh_prev = r["dh_langsung"].copy()
        for gb in "zrh":
            g["W_" + gb] += s["x"].T @ dmx[gb]
            g["b_" + gb + "_in"] += dmx[gb].sum(axis=0)
            g["U_" + gb] += s["h_prev"].T @ dmh[gb]
            g["b_" + gb + "_rec"] += dmh[gb].sum(axis=0)
            r["dh_dari_" + gb] = dmh[gb] @ p["U_" + gb].T
            dh_prev += r["dh_dari_" + gb]
        r["dh_prev"] = dh_prev
        rincian[t] = r
        dh = dh_prev
    return {"dy": dy, "g": g, "rincian": rincian}


FUNGSI = {"LSTM": (lstm_maju, lstm_mundur), "GRU": (gru_maju, gru_mundur)}


def adam_langkah(p, g, m, v, k, lr):
    """
    Satu update Adam persis seperti Keras 3 (iterasi ke-k, k mulai 1).

        m = m + (g - m)(1 - beta1)        (sama dengan beta1 m + (1 - beta1) g)
        v = v + (g^2 - v)(1 - beta2)
        alpha = lr sqrt(1 - beta2^k) / (1 - beta1^k)
        theta = theta - alpha m / (sqrt(v) + eps)
    """
    alpha = lr * math.sqrt(1 - B2 ** k) / (1 - B1 ** k)
    p_baru, m_baru, v_baru, delta = {}, {}, {}, {}
    for n in p:
        m_baru[n] = m[n] + (g[n] - m[n]) * (1 - B1)
        v_baru[n] = v[n] + (g[n] ** 2 - v[n]) * (1 - B2)
        delta[n] = alpha * m_baru[n] / (np.sqrt(v_baru[n]) + EPS)
        p_baru[n] = p[n] - delta[n]
    return p_baru, m_baru, v_baru, delta, alpha


class SiklusModelUtama:
    """
    Satu siklus pelatihan model utama yang dihitung ulang dengan NumPy.

    Parameter
    ---------
    model        : "LSTM" atau "GRU"
    bobot_awal   : ``model.get_weights()`` dari model yang baru dibangun dengan
                   seed penelitian (sebelum dilatih)
    X, y         : data latih (sliding window) dan targetnya
    batch, lr    : batch size dan learning rate penelitian
    """

    def __init__(self, model, bobot_awal, X, y, batch, lr, tanggal_target=None):
        assert model in ("LSTM", "GRU")
        self.model, self.batch, self.lr = model, int(batch), float(lr)
        self.maju_f, self.mundur_f = FUNGSI[model]
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.tanggal = tanggal_target
        self.p0 = pecah_bobot(model, bobot_awal)
        self.n_unit = self.p0["W_y"].shape[0]
        self.n_fitur = self.X.shape[2]
        self.lookback = self.X.shape[1]
        self.n_param = int(sum(v.size for v in self.p0.values()))

        # Iterasi k = 1: batch pertama dengan bobot awal
        Xb1, yb1 = self.ambil_batch(0)
        self.maju1 = self.maju_f(self.p0, Xb1, yb1)
        self.mundur1 = self.mundur_f(self.p0, self.maju1)
        nol = {n: np.zeros_like(v) for n, v in self.p0.items()}
        self.p1, self.m1, self.v1, self.delta1, self.alpha1 = adam_langkah(
            self.p0, self.mundur1["g"], nol, nol, 1, self.lr)
        self.L1_sesudah = self.maju_f(self.p1, Xb1, yb1)["L"]

        # Iterasi k = 2: batch kedua dengan bobot hasil iterasi 1
        Xb2, yb2 = self.ambil_batch(1)
        self.maju2 = self.maju_f(self.p1, Xb2, yb2)
        self.mundur2 = self.mundur_f(self.p1, self.maju2)
        self.p2, self.m2, self.v2, self.delta2, self.alpha2 = adam_langkah(
            self.p1, self.mundur2["g"], self.m1, self.v1, 2, self.lr)

        # Satu sampel (jendela pertama) untuk alur sel
        self.maju_satu = self.maju_f(self.p0, self.X[:1], self.y[:1])

    def ambil_batch(self, nomor):
        a_, b_ = nomor * self.batch, (nomor + 1) * self.batch
        return self.X[a_:b_], self.y[a_:b_]

    @property
    def iter_per_epoch(self):
        return math.ceil(len(self.X) / self.batch)

    # ------------------------------------------------------------------ #
    # Pemeriksaan terhadap Keras / TensorFlow
    # ------------------------------------------------------------------ #
    def periksa_forward(self, prediksi_keras_satu, prediksi_keras_batch, loss_keras):
        self.selisih_satu = float(abs(self.maju_satu["yhat"][0, 0] - float(np.ravel(prediksi_keras_satu)[0])))
        self.selisih_batch = float(np.max(np.abs(self.maju1["yhat"].ravel()
                                                 - np.ravel(prediksi_keras_batch))))
        self.selisih_loss = float(abs(self.maju1["L"] - float(loss_keras)))
        self.loss_keras = float(loss_keras)
        assert self.selisih_satu < 1e-5 and self.selisih_batch < 1e-5, \
            f"{self.model}: forward pass manual tidak cocok dengan Keras"
        assert self.selisih_loss < 1e-6, f"{self.model}: loss manual tidak cocok dengan Keras"

    def periksa_gradien(self, gradien_tf):
        """``gradien_tf``: daftar gradien TensorFlow dengan urutan ``model.trainable_weights``."""
        manual = gabung_bobot(self.model, self.mundur1["g"])
        self.gradien_tf = [np.asarray(gt, dtype=np.float64) for gt in gradien_tf]
        self.selisih_gradien = float(max(np.max(np.abs(m - t)) for m, t in zip(manual, self.gradien_tf)))
        skala = float(max(np.max(np.abs(t)) for t in self.gradien_tf))
        self.skala_gradien = skala
        assert self.selisih_gradien < 1e-5 * max(1.0, skala), \
            f"{self.model}: gradien BPTT manual tidak cocok dengan TensorFlow"

    def gradien_numerik(self, daftar, langkah=1e-6):
        """Turunan numerik (selisih pusat) untuk beberapa parameter terpilih."""
        Xb1, yb1 = self.ambil_batch(0)
        hasil = []
        for nama_p, indeks in daftar:
            naik = {k: v.copy() for k, v in self.p0.items()}
            turun = {k: v.copy() for k, v in self.p0.items()}
            naik[nama_p][indeks] += langkah
            turun[nama_p][indeks] -= langkah
            num = (self.maju_f(naik, Xb1, yb1)["L"] - self.maju_f(turun, Xb1, yb1)["L"]) / (2 * langkah)
            hasil.append((nama_p, indeks, float(self.mundur1["g"][nama_p][indeks]), float(num)))
        self.cek_numerik = hasil
        for _, _, g_, n_ in hasil:
            assert abs(g_ - n_) < 1e-6 + 1e-4 * abs(g_), f"{self.model}: gradien numerik tidak cocok"
        return hasil

    def periksa_adam(self, bobot_keras_1, bobot_keras_2):
        """Bobot Keras setelah batch ke-1 dan ke-2 dari ``model.fit`` sungguhan."""
        def selisih(p, keras):
            return float(max(np.max(np.abs(m - np.asarray(k, dtype=np.float64)))
                             for m, k in zip(gabung_bobot(self.model, p), keras)))
        self.selisih_adam1 = selisih(self.p1, bobot_keras_1)
        self.selisih_adam2 = selisih(self.p2, bobot_keras_2)
        assert self.selisih_adam1 < 1e-5 and self.selisih_adam2 < 1e-5, \
            f"{self.model}: update Adam manual tidak cocok dengan model.fit Keras"


# =========================================================================== #
# DAFTAR BERKAS PERHITUNGAN MANUAL (penomoran notebook versi model utama)
# =========================================================================== #
BERKAS = {
    ("LSTM", "jumlah_parameter"): ("07_04", "jumlah_parameter_lstm", "Jumlah Parameter Model LSTM"),
    ("LSTM", "alur_sel"): ("07_06", "alur_sel_lstm_model_utama",
                           "Fungsi Aktivasi dan Alur Sel LSTM pada Model Utama (Gambar 1-6)"),
    ("LSTM", "forward_loss"): ("07_07", "pelatihan_lstm_forward_pass_dan_loss",
                               "Pelatihan LSTM Model Utama, Langkah 1-2: Forward Pass dan Loss"),
    ("LSTM", "bptt"): ("07_08", "pelatihan_lstm_bptt",
                       "Pelatihan LSTM Model Utama, Langkah 3: Backpropagation Through Time"),
    ("LSTM", "adam"): ("07_09", "pelatihan_lstm_update_adam",
                       "Pelatihan LSTM Model Utama, Langkah 4: Update Adam"),
    ("LSTM", "kurva_loss"): ("07_10", "kurva_loss_lstm", "Membaca Kurva Loss Model LSTM"),
    ("LSTM", "forward_pass"): ("07_11", "forward_pass_lstm", "Forward Pass LSTM (Jendela Pertama Data Uji)"),
    ("LSTM", "bias"): ("07_12", "bias_lstm", "Bias pada LSTM: Peran dan Cara Menghitungnya"),
    ("GRU", "jumlah_parameter"): ("08_04", "jumlah_parameter_gru", "Jumlah Parameter Model GRU"),
    ("GRU", "alur_sel"): ("08_06", "alur_sel_gru_model_utama", "Alur Sel GRU pada Model Utama (Gambar 7)"),
    ("GRU", "forward_loss"): ("08_07", "pelatihan_gru_forward_pass_dan_loss",
                              "Pelatihan GRU Model Utama, Langkah 1-2: Forward Pass dan Loss"),
    ("GRU", "bptt"): ("08_08", "pelatihan_gru_bptt",
                      "Pelatihan GRU Model Utama, Langkah 3: Backpropagation Through Time"),
    ("GRU", "adam"): ("08_09", "pelatihan_gru_update_adam",
                      "Pelatihan GRU Model Utama, Langkah 4: Update Adam"),
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
    tahap, sub = nomor.split("_")
    return f"Tahap {int(tahap)}.{int(sub)}"


def label(model: str, bagian: str) -> str:
    """("LSTM", "bptt") -> "Tahap 7.8"."""
    return label_tahap(BERKAS[(model, bagian)][0])


def tautan(model: str, bagian: str, sub: str | None = None) -> str:
    nomor, _, judul = BERKAS[(model, bagian)]
    if sub is None:
        return f"[{label_tahap(nomor)} ({judul})]({nama_berkas(model, bagian)})"
    return (f"[{label_tahap(nomor)}, bagian {sub.split('.')[0]}]"
            f"({nama_berkas(model, bagian)}#{jangkar(sub)})")


# --------------------------------------------------------------------------- #
# Fungsi yang dipakai ulang dari modul lama (tidak bergantung pada model mini)
# --------------------------------------------------------------------------- #
from manual_pelatihan import konteks_penelitian, hasil_model  # noqa: E402,F401


# =========================================================================== #
# FORMAT ANGKA
# =========================================================================== #
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sb(n) -> str:
    """Angka subskrip: 7 -> ₇."""
    return str(n).translate(SUB)


def e(x, d=6) -> str:
    """Angka yang bisa sangat kecil (gradien): pakai notasi ilmiah bila |x| < 1e-4."""
    x = float(x)
    if x == 0.0:
        return "0"
    if abs(x) < 1e-4:
        return f"{x:.4e}"
    return a(x, d)


def ke(x, d=6) -> str:
    """Seperti ``e`` tetapi diberi kurung bila negatif (untuk perkalian/penjumlahan)."""
    s = e(x, d)
    return f"({s})" if float(x) < 0 and s != "0" else s


def tgl(t) -> str:
    return str(t)[:10]


CATATAN_ANGKA = """
**Cara membaca angka.** Titik dipakai sebagai pemisah desimal dan koma sebagai pemisah
ribuan. Angka biasanya ditampilkan 6 desimal; angka yang sangat kecil (misalnya gradien)
ditulis dalam notasi ilmiah, contohnya 2.3416e-07 = 0.00000023416. Semua hitungan
dilakukan dengan presisi penuh, jadi kalau Anda menjumlahkan angka yang tampil, selisih
kecil di digit terakhir itu wajar. Nomor persamaan dan gambar mengacu pada draf skripsi
(subbab 1.5.7 sampai 1.5.11).
"""


def _pengantar_model_utama(d, sk, kt, info):
    m = sk.model
    d.teks(f"""
**Semua angka di sini berasal dari model {m} utama penelitian**, yaitu model terbaik hasil
tuning: **{sk.n_unit} neuron**, masukan **{sk.lookback} hari × {sk.n_fitur} fitur**, batch
**{sk.batch}**, dan dilatih {ribu(info['epoch'])} epoch. Tidak ada model mini atau angka
karangan. Karena angkanya banyak ({ribu(sk.n_param)} parameter), triknya begini: kita ikuti
**satu neuron (neuron ke-1)** dan **satu sampel** secara rinci sampai bisa dicek dengan
kalkulator, sedangkan neuron dan sampel lainnya dihitung komputer dengan rumus yang sama
lalu dicocokkan dengan Keras.
""")


# =========================================================================== #
# ISI BERKAS: ALUR SEL (Tahap 7.6 dan 8.6)
# =========================================================================== #
def _tabel_bobot_awal(d, sk):
    p, F, u = sk.p0, sk.n_fitur, sk.n_unit
    if sk.model == "LSTM":
        W = np.concatenate([p["W_" + g] for g in "ifco"], axis=1)
        U = np.concatenate([p["U_" + g] for g in "ifco"], axis=1)
        batas_W = math.sqrt(6 / (F + 4 * u))
        bias_baris = [["bias b", f"({4 * u},)", "nol, kecuali forget gate = 1 (*unit forget bias*)",
                       f"b_f = {a(p['b_f'].min(), 0)} untuk semua neuron; b_i = b_c = b_o = 0"]]
        nama_W, nama_U = f"({F}, {4 * u}) = {F} fitur × 4 gerbang × {u} neuron", f"({u}, {4 * u})"
    else:
        W = np.concatenate([p["W_" + g] for g in "zrh"], axis=1)
        U = np.concatenate([p["U_" + g] for g in "zrh"], axis=1)
        batas_W = math.sqrt(6 / (F + 3 * u))
        semua_b = np.concatenate([p[f"b_{g}_{j}"] for g in "zrh" for j in ("in", "rec")])
        bias_baris = [["bias b", f"(2, {3 * u})", "semua nol (baris 1 = bias masukan, baris 2 = bias rekuren)",
                       f"semuanya {a(np.abs(semua_b).max(), 0)}"]]
        nama_W, nama_U = f"({F}, {3 * u}) = {F} fitur × 3 gerbang × {u} neuron", f"({u}, {3 * u})"
    orto = float(np.max(np.abs(U.T @ U - np.eye(U.shape[1])))) if U.shape[1] <= U.shape[0] else \
        float(np.max(np.abs(U @ U.T - np.eye(U.shape[0]))))
    batas_Wy = math.sqrt(6 / (u + 1))
    assert np.abs(W).max() <= batas_W + 1e-6 and np.abs(p["W_y"]).max() <= batas_Wy + 1e-6
    d.tabel(["Bobot", "Bentuk", "Cara Keras mengisinya", "Nilai pada model ini"],
            [["kernel W (masukan)", nama_W, f"acak Glorot uniform, rentang ±√(6/(fan_in+fan_out)) = ±{a(batas_W, 4)}",
              f"{a(W.min(), 4)} s.d. {a(W.max(), 4)}"],
             ["recurrent kernel U", nama_U, "acak ortogonal (baris-barisnya saling tegak lurus)",
              f"{a(U.min(), 4)} s.d. {a(U.max(), 4)}; cek ortogonal: selisih {orto:.1e}"],
             *bias_baris,
             ["Dense W_y", f"({u}, 1)", f"Glorot uniform, ±{a(batas_Wy, 4)}",
              f"{a(p['W_y'].min(), 4)} s.d. {a(p['W_y'].max(), 4)}"],
             ["Dense b_y", "(1,)", "nol", a(p["b_y"][0], 0)]])


def _jendela_pertama(d, sk, kt, info):
    X0, y0 = sk.X[0], float(sk.y[0, 0])
    kode = info["kode"]
    d.tabel(["Hari", "Tanggal"] + kode,
            [[f"t = {t + 1}", tgl(info["tanggal_jendela"][t])] + [a(X0[t, k], 4) for k in range(sk.n_fitur)]
             for t in range(sk.lookback)])
    d.teks(f"""
Targetnya adalah Close Price ternormalisasi tanggal **{tgl(info['tanggal_target'][0])}**:
y′ = **{a(y0)}**, setara {a(y0)} × {usd(kt['rentang'])} + {usd(kt['x_min'])} =
**{usd(y0 * kt['rentang'] + kt['x_min'])} USD**. Keadaan awal memori mengikuti Keras: h₀ = 0
{'dan c₀ = 0 ' if sk.model == 'LSTM' else ''}untuk semua neuron.
""")


def _tabel_kali_masukan(d, sk, info, t, gerbang, simbol):
    """Tabel x_t,k × W[k, neuron 1] untuk satu gerbang."""
    x = sk.maju_satu["langkah"][t]["x"][0]
    Wg = sk.p0["W_" + gerbang][:, 0]
    baris = [[info["kode"][k], info["nama"][k], a(x[k]), kr(Wg[k]), kr(x[k] * Wg[k])] for k in range(sk.n_fitur)]
    baris.append(["", "**jumlah**", "", "", f"**{a(float(x @ Wg))}**"])
    d.tabel(["Kode", "Fitur", f"x{sb(t + 1)}", f"W_{simbol}[k, 1]", f"x{sb(t + 1)} × W_{simbol}"], baris)
    return float(x @ Wg)


def _rekuren_neuron1(sk, t, kunci_U):
    """Rincian h_(t-1) U[:, 1] untuk neuron ke-1: (beberapa suku pertama, total)."""
    h_prev = sk.maju_satu["langkah"][t]["h_prev"][0]
    U1 = sk.p0[kunci_U][:, 0]
    suku = h_prev * U1
    return h_prev, U1, suku, float(suku.sum())


def _baris_rekuren(nama_h, h_prev, U1, suku, simbol, banyak=3):
    teks = " + ".join(f"{kr(h_prev[m])}×{kr(U1[m])}" for m in range(min(banyak, len(suku))))
    sisa = len(suku) - banyak
    return (f"  {nama_h}·U_{simbol}[:,1] = {teks}"
            + (f" + … ({sisa} suku lagi)" if sisa > 0 else "")
            + f" = {a(suku.sum())}")


def _alur_lstm(sk, kt, info):
    import manual_pelatihan as mp
    s = sk.maju_satu["langkah"]
    p = sk.p0
    u, T = sk.n_unit, sk.lookback
    s1, s2 = s[0], s[1]
    J = {1: "1. Model dan Data yang Dipakai", 2: "2. Cara Membaca Gambar",
         3: "3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)",
         4: "4. Gambar 1: Satu Sel LSTM Menerima Apa Saja", 5: "5. Time Step t = 1, Neuron ke-1",
         6: "6. Time Step t = 2: Memori Mulai Bekerja", 7: f"7. Tujuh Time Step Neuron ke-1",
         8: "8. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)", 9: "9. Ringkasan Satu Langkah LSTM",
         10: "10. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Bagian ini menjelaskan alur **Gambar 1 sampai Gambar 6** (subbab 1.5.8) memakai model LSTM
yang benar-benar dipakai penelitian. Kita jalankan satu jendela data asli melewati sel LSTM,
hari demi hari, lalu melihat apa yang dikerjakan setiap gerbang.
""")
    _pengantar_model_utama(d, sk, kt, info)
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks(f"""
**Arsitektur:** Input ({T} × {sk.n_fitur}) → LSTM ({u} neuron) → Dense (1), total
**{ribu(sk.n_param)} parameter** (rinciannya di {tautan('LSTM', 'jumlah_parameter')}).

**Bobot yang dipakai adalah bobot awal**, yaitu bobot tepat sebelum epoch pertama dimulai.
Bobot ini dibuat ulang dengan `bangun_model("LSTM", {u}, seed=42)`. Seed dikunci sebelum
bobot diundi, jadi hasilnya sama persis dengan bobot awal model terbaik waktu dilatih di
Tahap 7.1 (notebook memeriksanya dengan membangun model dua kali dan membandingkan semua
bobotnya). Kenapa bobot awal, bukan bobot hasil latihan? Karena kita ingin melihat apa yang
terjadi **saat pelatihan dimulai**. Bobot hasil latihan dipakai di {tautan('LSTM', 'forward_pass')}.
""")
    _tabel_bobot_awal(d, sk)
    d.teks(f"""
Jadi bobot awal memang acak, tetapi tidak sembarangan: rentangnya dibatasi supaya nilai
gerbang tidak langsung jenuh di 0 atau 1, dan bias forget gate dibuat 1 supaya di awal
pelatihan sel cenderung **menyimpan** memori, bukan membuangnya.

**Data:** jendela pertama data latih (`X_latih[0]`, sama dengan Tahap 6.2), nilai
ternormalisasi:
""")
    _jendela_pertama(d, sk, kt, info)

    d.teks(f"## {J[2]}")
    d.teks("Gambar 1-6 berasal dari Olah (2015). Simbolnya:")
    d.tabel(["Simbol", "Arti"],
            [["Kotak kuning (σ atau tanh)", "lapisan yang **punya bobot dan bias** (W, U, b) dan ikut dilatih"],
             ["Lingkaran merah muda (×, +)", "operasi elemen demi elemen **tanpa bobot** (× = perkalian Hadamard ⊙)"],
             ["Oval \"tanh\" (Gambar 1 dan 6)", "fungsi tanh biasa tanpa bobot, hanya memampatkan nilai ke −1 sampai 1"],
             ["Dua garis menyatu", "hₜ₋₁ dan xₜ masuk bersama ke satu gerbang (di skripsi: xₜW + hₜ₋₁U + b)"],
             ["Garis paling atas", "cell state cₜ (memori jangka panjang)"]])
    d.teks("""
Gampangnya: **σ (0 sampai 1) itu keran**, menentukan berapa persen sesuatu boleh lewat.
**tanh (−1 sampai 1) itu isi informasinya**, bisa positif atau negatif, dan tidak bisa
meledak karena dibatasi.
""")

    d.teks(f"## {J[3]}")
    sementara = Dokumen()
    mp._sigmoid_tanh(sementara, {"a_c": float(s1["a_c"][0, 0]), "cc": float(s1["cc"][0, 0])})
    isi = "\n".join(sementara.baris)
    isi = isi.split("### 3.7")[0].rstrip()
    d.baris.extend(isi.split("\n"))
    d.baris.append("")
    d.teks("### 3.7 Cek dengan Angka Model Utama")
    ac = float(s1["a_c"][0, 0])
    lewat_12 = (math.exp(ac) - math.exp(-ac)) / (math.exp(ac) + math.exp(-ac))
    uu = math.exp(-2 * ac)
    lewat_13 = 2 * (1 / (1 + uu)) - 1
    assert abs(lewat_12 - float(s1["cc"][0, 0])) < 1e-12 and abs(lewat_13 - lewat_12) < 1e-12
    d.teks(f"""
Kandidat cell state neuron ke-1 pada t = 1 (bagian 5) memakai tanh dengan pra-aktivasi
a = {a(ac)}. Kedua rumus memberi hasil yang sama:
""")
    d.kode([f"Persamaan (12): (e^a − e^(−a)) / (e^a + e^(−a))",
            f"  = ({a(math.exp(ac))} − {a(math.exp(-ac))}) / ({a(math.exp(ac))} + {a(math.exp(-ac))}) = {a(lewat_12)}",
            f"Persamaan (13): 2σ(2a) − 1 = 2 × σ({a(2 * ac)}) − 1 = 2 × {a(1 / (1 + uu))} − 1 = {a(lewat_13)}",
            f"Nilai c̃₁ dari model = {a(s1['cc'][0, 0])} ✓"])
    d.teks("### 3.8 Kaitan dengan Turunan pada BPTT")
    d.teks(f"""
Dengan s = σ(2z), persamaan (13) memberi 1 − tanh²(z) = 4s(1 − s). Artinya turunan kedua
fungsi cukup dihitung dari nilai keluarannya sendiri: σ′ = σ(1 − σ) dan tanh′ = 1 − tanh².
Dua rumus ini dipakai terus di {tautan('LSTM', 'bptt')}.
""")

    d.teks(f"## {J[4]}")
    d.teks(f"""
Pada setiap hari t, setiap neuron menerima dua hal: **xₜ** (10 nilai fitur hari itu) dan
**hₜ₋₁** ({u} nilai hidden state dari hari sebelumnya, satu dari setiap neuron). Keduanya
masuk ke empat kotak kuning: forget gate, input gate, kandidat, dan output gate. Untuk
neuron ke-j, isi kotaknya selalu berpola sama:

$$a_j = \\underbrace{{\\sum_{{k=1}}^{{{sk.n_fitur}}} x_{{t,k}}\\,W[k,j]}}_{{\\text{{bagian masukan}}}}
+ \\underbrace{{\\sum_{{m=1}}^{{{u}}} h_{{t-1,m}}\\,U[m,j]}}_{{\\text{{bagian memori}}}} + b_j$$

lalu a dimasukkan ke σ (untuk gerbang) atau tanh (untuk kandidat). Satu neuron per gerbang
butuh {sk.n_fitur} + {u} = {sk.n_fitur + u} perkalian, jadi satu hari untuk seluruh sel butuh
4 × {u} × {sk.n_fitur + u} = **{ribu(4 * u * (sk.n_fitur + u))} perkalian**. Inilah sebabnya
kita merinci satu neuron saja; neuron lain caranya persis sama.
""")

    d.teks(f"## {J[5]}")
    d.teks(f"""
Di hari pertama belum ada memori (h₀ = 0, c₀ = 0), jadi bagian memori ΣhU bernilai 0.
Contoh lengkap untuk **forget gate** (Gambar 3) neuron ke-1: kalikan setiap fitur hari
pertama ({tgl(info['tanggal_jendela'][0])}) dengan bobotnya, lalu jumlahkan.
""")
    _tabel_kali_masukan(d, sk, info, 0, "f", "f")
    d.teks("Tiga gerbang lainnya dihitung dengan cara yang sama persis, hanya bobotnya yang beda:")
    baris_g = []
    for g, nm, akt in (("f", "forget f₁", "σ"), ("i", "input i₁", "σ"), ("c", "kandidat c̃₁", "tanh"),
                       ("o", "output o₁", "σ")):
        nilai = s1["cc"][0, 0] if g == "c" else s1[g][0, 0]
        baris_g.append([nm, a(s1["mx_" + g][0, 0]), "0", a(p["b_" + g][0], 0), a(s1["a_" + g][0, 0]),
                        f"{akt}({a(s1['a_' + g][0, 0])}) = **{a(nilai)}**"])
    d.tabel(["Gerbang", "Σ x₁·W", "Σ h₀·U", "b", "a = jumlah", "Hasil"], baris_g)
    f1, i1, c1t, o1 = (float(s1[k][0, 0]) for k in ("f", "i", "cc", "o"))
    d.kode([
        "Gambar 3 (forget): karena c₀ = 0, belum ada memori yang bisa disimpan atau dibuang.",
        f"  f₁ × c₀ = {a(f1)} × 0 = 0",
        "Gambar 4 (input): usulan isi baru c̃₁ hanya masuk sebagian, sebesar i₁.",
        f"  i₁ × c̃₁ = {a(i1)} × {kr(c1t)} = {a(i1 * c1t)}",
        "Gambar 5 (update cell state):",
        f"  c₁ = f₁ × c₀ + i₁ × c̃₁ = 0 + {kr(i1 * c1t)} = {a(s1['c'][0, 0])}",
        "Gambar 6 (output):",
        f"  tanh(c₁) = tanh({a(s1['c'][0, 0])}) = {a(s1['tc'][0, 0])}",
        f"  h₁ = o₁ × tanh(c₁) = {a(o1)} × {kr(s1['tc'][0, 0])} = {a(s1['h'][0, 0])}",
    ])
    d.teks(f"""
Perhatikan forget gate: biasnya 1, jadi meskipun masukannya kecil, f₁ = {a(f1)} sudah lebih
dari 0.5. Sel "disetel" untuk menyimpan memori sejak awal. Semua {u} neuron mengerjakan hal
yang sama secara bersamaan, sehingga hasil t = 1 adalah vektor h₁ dan c₁ yang masing-masing
berisi {u} angka.
""")

    d.teks(f"## {J[6]}")
    h1 = s1["h"][0]
    d.teks(f"""
Mulai hari kedua, bagian memori tidak nol lagi. Neuron ke-1 sekarang menerima h₁ dari
**seluruh {u} neuron**: h₁ = [{', '.join(a(v) for v in h1[:4])}, …]. Contoh bagian memori untuk
forget gate:
""")
    hp, U1, suku, tot = _rekuren_neuron1(sk, 1, "U_f")
    d.kode([_baris_rekuren("h₁", hp, U1, suku, "f")])
    baris_g = []
    for g, nm, akt in (("f", "forget f₂", "σ"), ("i", "input i₂", "σ"), ("c", "kandidat c̃₂", "tanh"),
                       ("o", "output o₂", "σ")):
        nilai = s2["cc"][0, 0] if g == "c" else s2[g][0, 0]
        baris_g.append([nm, a(s2["mx_" + g][0, 0]), a(s2["mh_" + g][0, 0]), a(p["b_" + g][0], 0),
                        a(s2["a_" + g][0, 0]), f"{akt}({a(s2['a_' + g][0, 0])}) = **{a(nilai)}**"])
    d.tabel(["Gerbang", "Σ x₂·W (10 suku)", f"Σ h₁·U ({u} suku)", "b", "a = jumlah", "Hasil"], baris_g)
    f2, i2, cc2, c1 = (float(s2[k][0, 0]) for k in ("f", "i", "cc", "c_prev"))
    d.kode([
        f"Memori lama yang disimpan (Gambar 3): f₂ × c₁ = {a(f2)} × {kr(c1)} = {a(f2 * c1)}",
        f"Informasi baru yang ditulis (Gambar 4): i₂ × c̃₂ = {a(i2)} × {kr(cc2)} = {a(i2 * cc2)}",
        f"Cell state baru (Gambar 5): c₂ = {kr(f2 * c1)} + {kr(i2 * cc2)} = {a(s2['c'][0, 0])}",
        f"Hidden state (Gambar 6): h₂ = o₂ × tanh(c₂) = {a(s2['o'][0, 0])} × {kr(s2['tc'][0, 0])} = {a(s2['h'][0, 0])}",
    ])
    d.teks(f"""
Di sini terlihat peran forget gate: {persen(f2)} isi memori kemarin dibawa ke hari ini, lalu
ditambah informasi baru yang lolos lewat input gate sebesar {persen(i2)}.
""")

    d.teks(f"## {J[7]}")
    d.teks(f"Hari ketiga sampai ketujuh dihitung dengan cara yang sama. Hasil neuron ke-1:")
    d.tabel(["t", "Tanggal", "fₜ", "iₜ", "c̃ₜ", "cₜ", "oₜ", "hₜ"],
            [[str(t + 1), tgl(info["tanggal_jendela"][t])] + [a(s[t][k][0, 0]) for k in ("f", "i", "cc", "c", "o", "h")]
             for t in range(T)])
    f_semua = np.array([s[t]["f"][0, 0] for t in range(1, T)])
    d.teks(f"""
Forget gate neuron ke-1 berada di kisaran {a(f_semua.min(), 3)} sampai {a(f_semua.max(), 3)}.
Kalau dikalikan terus dari hari ke-2 sampai ke-{T}, porsi isi memori hari pertama yang masih
tersisa di hari terakhir kira-kira {persen(float(np.prod(f_semua)))}. Itulah "memori jangka
panjang" pada LSTM: informasi lama tidak hilang mendadak, tetapi meluruh pelan-pelan sesuai
nilai forget gate. Setelah dilatih, nilai-nilai gerbang ini akan berubah karena bobotnya
berubah.
""")

    d.teks(f"## {J[8]}")
    hT = s[-1]["h"][0]
    Wy = p["W_y"][:, 0]
    suku = hT * Wy
    yhat = float(sk.maju_satu["yhat"][0, 0])
    y0 = float(sk.y[0, 0])
    baris = [f"  ŷ′ = Σ h{sb(T)}[j] × W_y[j] + b_y   (j = 1 … {u})"]
    for j in range(min(4, u)):
        baris.append(f"     h{sb(T)}[{j + 1}] × W_y[{j + 1}] = {kr(hT[j])} × {kr(Wy[j])} = {kr(suku[j])}")
    if u > 4:
        baris.append(f"     … ({u - 4} suku lagi)")
    baris += [f"  Jumlah {u} suku = {a(suku.sum())}",
              f"  ŷ′ = {a(suku.sum())} + {kr(p['b_y'][0])} = {a(yhat)}",
              "",
              f"  Dalam USD: {a(yhat)} × {usd(kt['rentang'])} + {usd(kt['x_min'])} = "
              f"{usd(yhat * kt['rentang'] + kt['x_min'])} USD",
              f"  Target   : {a(y0)}  ({usd(y0 * kt['rentang'] + kt['x_min'])} USD)",
              f"  Selisih dengan model.predict() Keras: {sk.selisih_satu:.2e}"]
    d.teks(f"""
Setelah hari ke-{T}, hidden state h{sb(T)} dari **{u} neuron** dikirim ke lapisan Dense
(persamaan 21). Dense hanya menjumlahkan h{sb(T)} yang sudah diberi bobot:
""")
    d.kode(baris)
    d.teks(f"""
Hasil hitungan manual sama dengan `model.predict()` Keras (selisih {sk.selisih_satu:.1e}).
Prediksinya meleset {usd(abs(yhat - y0) * kt['rentang'])} USD dari target. Besar kecilnya meleset di
sampel ini belum berarti apa-apa, karena bobotnya masih acak. Justru di sinilah pelatihan
mulai bekerja: selisih antara ŷ′ dan y′ akan diukur dengan loss
({tautan('LSTM', 'forward_loss')}), lalu bobot diperbaiki sedikit demi sedikit.
""")

    d.teks(f"## {J[9]}")
    d.tabel(["Urutan", "Gambar", "Rumus (persamaan)", "Fungsinya"],
            [["1. Forget", "Gambar 3", "fₜ = σ(xₜW_f + hₜ₋₁U_f + b_f) (15)", "berapa persen memori lama disimpan"],
             ["2. Input dan kandidat", "Gambar 4", "iₜ = σ(…), c̃ₜ = tanh(…) (16, 17)", "informasi baru apa dan seberapa banyak"],
             ["3. Update memori", "Gambar 5", "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ (18)", "memori jangka panjang yang baru"],
             ["4. Output", "Gambar 6", "oₜ = σ(…), hₜ = oₜ ⊙ tanh(cₜ) (19, 20)", "bagian memori yang dikeluarkan hari ini"],
             ["5. Prediksi", "-", f"ŷ′ = h{sb(T)}W_y + b_y (21)", "dipakai sekali, setelah hari terakhir"]])

    d.teks(f"## {J[10]}")
    d.teks(f"""
Kalimat yang bisa dipakai di subbab 1.5.8 setelah Gambar 6:

> Pada penelitian ini setiap sel LSTM terdiri atas {u} neuron. Pada setiap hari t, setiap
> neuron menerima {sk.n_fitur} nilai fitur xₜ dan {u} nilai hidden state hₜ₋₁, menghitung
> keempat gerbang menurut persamaan (15)-(20), lalu setelah hari ke-{T} hidden state h{sb(T)}
> diteruskan ke lapisan dense untuk menghasilkan prediksi harga penutupan hari berikutnya.
""")
    return d.isi()


def _alur_gru(sk, kt, info):
    s = sk.maju_satu["langkah"]
    p = sk.p0
    u, T = sk.n_unit, sk.lookback
    s1, s2 = s[0], s[1]
    J = {1: "1. Model dan Data yang Dipakai", 2: "2. Cara Membaca Gambar 7",
         3: "3. Time Step t = 1, Neuron ke-1", 4: "4. Time Step t = 2: Memori Mulai Bekerja",
         5: "5. Tujuh Time Step Neuron ke-1", 6: "6. Dari Seluruh Neuron ke Prediksi (Lapisan Dense)",
         7: "7. Ringkasan Satu Langkah GRU", 8: "8. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks("""
Bagian ini menjelaskan alur **Gambar 7** (struktur sel GRU, subbab 1.5.9) memakai model GRU
yang benar-benar dipakai penelitian. Polanya sama dengan LSTM di Tahap 7.6, tetapi selnya
lebih ramping: dua gerbang dan tanpa cell state terpisah.
""")
    _pengantar_model_utama(d, sk, kt, info)
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks(f"""
**Arsitektur:** Input ({T} × {sk.n_fitur}) → GRU ({u} neuron) → Dense (1), total
**{ribu(sk.n_param)} parameter** (rinciannya di {tautan('GRU', 'jumlah_parameter')}).

Sama seperti LSTM, yang dipakai adalah **bobot awal** model terbaik, dibuat ulang dengan
`bangun_model("GRU", {u}, seed=42)` sehingga sama persis dengan bobot awal saat model ini
dilatih di Tahap 8.1:
""")
    _tabel_bobot_awal(d, sk)
    d.teks("""
Berbeda dengan LSTM, GRU tidak punya trik bias awal: semua biasnya mulai dari nol.

**Data:** jendela pertama data latih (`X_latih[0]`), sama dengan yang dipakai LSTM:
""")
    _jendela_pertama(d, sk, kt, info)

    d.teks(f"## {J[2]}")
    d.tabel(["Simbol pada Gambar 7", "Arti"],
            [["Kotak kuning σ (dua buah)", "lapisan berbobot yang menghasilkan update gate zₜ dan reset gate rₜ"],
             ["Kotak kuning tanh", "lapisan berbobot yang menghasilkan kandidat h̃ₜ"],
             ["Lingkaran ×", "perkalian elemen demi elemen (⊙), tanpa bobot"],
             ["Lingkaran 1−", "menghitung 1 − zₜ"],
             ["Lingkaran +", "penjumlahan, menghasilkan hₜ"],
             ["Garis atas", "hidden state hₜ, satu-satunya memori GRU"]])
    d.teks(f"""
Seperti di LSTM, σ berperan sebagai keran dan tanh sebagai isi informasi. Rumus dan
penjabaran keduanya (persamaan 11-13) ada di
{tautan('LSTM', 'alur_sel', '3. Sigmoid dan Tanh: Penjabaran Persamaan (11)-(13)')}.

Rumus yang dihitung mengikuti Keras (`reset_after=True`, Tahap 8.A), dengan dua bias per
gerbang:

- zₜ = σ(xₜW_z + b_z(in) + hₜ₋₁U_z + b_z(rec))  (persamaan 23)
- rₜ = σ(xₜW_r + b_r(in) + hₜ₋₁U_r + b_r(rec))  (persamaan 24)
- h̃ₜ = tanh(xₜW_h + b_h(in) + rₜ ⊙ (hₜ₋₁U_h + b_h(rec)))  (persamaan 25)
- hₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ h̃ₜ  (persamaan 26)
""")

    d.teks(f"## {J[3]}")
    d.teks(f"""
Di hari pertama h₀ = 0, jadi semua bagian memori nol dan reset gate belum berpengaruh
apa-apa. Contoh lengkap **update gate** neuron ke-1 untuk hari {tgl(info['tanggal_jendela'][0])}:
""")
    _tabel_kali_masukan(d, sk, info, 0, "z", "z")
    xw = {g: float(s1["x"][0] @ p["W_" + g][:, 0]) for g in "zrh"}
    d.teks("Reset gate dan kandidat dihitung dengan cara yang sama:")
    d.tabel(["Bagian", "Σ x₁·W", "b(in)", "Σ h₀·U + b(rec)", "Hasil"],
            [["update z₁", a(xw["z"]), a(p["b_z_in"][0], 0), e(s1["mh_z"][0, 0]),
              f"σ({a(s1['a_z'][0, 0])}) = **{a(s1['z'][0, 0])}**"],
             ["reset r₁", a(xw["r"]), a(p["b_r_in"][0], 0), e(s1["mh_r"][0, 0]),
              f"σ({a(s1['a_r'][0, 0])}) = **{a(s1['r'][0, 0])}**"],
             ["kandidat h̃₁", a(xw["h"]), a(p["b_h_in"][0], 0), f"r₁ × {e(s1['mh_h'][0, 0])} = {e(s1['rq'][0, 0])}",
              f"tanh({a(s1['a_h'][0, 0])}) = **{a(s1['hh'][0, 0])}**"]])
    z1, hh1 = float(s1["z"][0, 0]), float(s1["hh"][0, 0])
    d.kode([
        "Campuran memori lama dan baru (persamaan 26):",
        f"  h₁ = z₁ × h₀ + (1 − z₁) × h̃₁ = {a(z1)} × 0 + {a(1 - z1)} × {kr(hh1)} = {a(s1['h'][0, 0])}",
    ])
    d.teks(f"""
Perhatikan bedanya dengan LSTM: di GRU, **z dan (1 − z) saling terikat**. Kalau z = {a(z1, 3)},
berarti {persen(z1)} memori lama dipertahankan dan otomatis hanya {persen(1 - z1)} kandidat
baru yang masuk. Di LSTM, forget gate dan input gate bisa bergerak sendiri-sendiri.
""")

    d.teks(f"## {J[4]}")
    h1 = s1["h"][0]
    d.teks(f"""
Mulai hari kedua, neuron ke-1 menerima h₁ dari **seluruh {u} neuron**:
h₁ = [{', '.join(a(v) for v in h1[:4])}, …]. Contoh bagian memori untuk update gate:
""")
    hp, U1, suku, tot = _rekuren_neuron1(sk, 1, "U_z")
    d.kode([_baris_rekuren("h₁", hp, U1, suku, "z")])
    xw2 = {g: float(s2["x"][0] @ p["W_" + g][:, 0]) for g in "zrh"}
    d.tabel(["Bagian", "Σ x₂·W + b(in)", f"Σ h₁·U + b(rec) ({u} suku)", "Hasil"],
            [["update z₂", a(s2["mx_z"][0, 0]), a(s2["mh_z"][0, 0]), f"σ({a(s2['a_z'][0, 0])}) = **{a(s2['z'][0, 0])}**"],
             ["reset r₂", a(s2["mx_r"][0, 0]), a(s2["mh_r"][0, 0]), f"σ({a(s2['a_r'][0, 0])}) = **{a(s2['r'][0, 0])}**"],
             ["kandidat h̃₂", a(s2["mx_h"][0, 0]),
              f"r₂ × {a(s2['mh_h'][0, 0])} = {a(s2['r'][0, 0])} × {kr(s2['mh_h'][0, 0])} = {a(s2['rq'][0, 0])}",
              f"tanh({a(s2['a_h'][0, 0])}) = **{a(s2['hh'][0, 0])}**"]])
    assert abs(xw2["z"] + p["b_z_in"][0] - s2["mx_z"][0, 0]) < 1e-12
    z2, hh2, h1_1 = float(s2["z"][0, 0]), float(s2["hh"][0, 0]), float(s2["h_prev"][0, 0])
    d.kode([
        f"Reset gate: dari isi memori h₁U_h + b_h(rec) = {a(s2['mh_h'][0, 0])}, "
        f"hanya {persen(float(s2['r'][0, 0]))} yang dipakai menyusun kandidat.",
        f"h₂ = z₂ × h₁ + (1 − z₂) × h̃₂ = {a(z2)} × {kr(h1_1)} + {a(1 - z2)} × {kr(hh2)}",
        f"   = {kr(z2 * h1_1)} + {kr((1 - z2) * hh2)} = {a(s2['h'][0, 0])}",
    ])

    d.teks(f"## {J[5]}")
    d.tabel(["t", "Tanggal", "zₜ", "rₜ", "h̃ₜ", "hₜ"],
            [[str(t + 1), tgl(info["tanggal_jendela"][t])] + [a(s[t][k][0, 0]) for k in ("z", "r", "hh", "h")]
             for t in range(T)])
    z_semua = np.array([s[t]["z"][0, 0] for t in range(1, T)])
    d.teks(f"""
Update gate neuron ke-1 berada di kisaran {a(z_semua.min(), 3)} sampai {a(z_semua.max(), 3)}, jadi
setiap hari sekitar {persen(float(z_semua.mean()), 0)} memori lama dibawa ke hari berikutnya. Porsi isi memori hari
pertama yang masih tersisa di hari ke-{T} kira-kira {persen(float(np.prod(z_semua)))}. Nilai-nilai ini
akan berubah setelah bobot dilatih.
""")

    d.teks(f"## {J[6]}")
    hT, Wy = s[-1]["h"][0], p["W_y"][:, 0]
    suku = hT * Wy
    yhat, y0 = float(sk.maju_satu["yhat"][0, 0]), float(sk.y[0, 0])
    baris = [f"  ŷ′ = Σ h{sb(T)}[j] × W_y[j] + b_y   (j = 1 … {u})"]
    for j in range(min(4, u)):
        baris.append(f"     h{sb(T)}[{j + 1}] × W_y[{j + 1}] = {kr(hT[j])} × {kr(Wy[j])} = {kr(suku[j])}")
    if u > 4:
        baris.append(f"     … ({u - 4} suku lagi)")
    baris += [f"  Jumlah {u} suku = {a(suku.sum())}",
              f"  ŷ′ = {a(suku.sum())} + {kr(p['b_y'][0])} = {a(yhat)}",
              "",
              f"  Dalam USD: {usd(yhat * kt['rentang'] + kt['x_min'])} USD   "
              f"(target {usd(y0 * kt['rentang'] + kt['x_min'])} USD)",
              f"  Selisih dengan model.predict() Keras: {sk.selisih_satu:.2e}"]
    d.kode(baris)
    d.teks(f"""
Hasilnya cocok dengan Keras. Prediksinya meleset {usd(abs(yhat - y0) * kt['rentang'])} USD dari
target; wajar karena bobotnya masih acak. Bobot inilah yang akan diperbaiki lewat siklus pelatihan di {tautan('GRU', 'forward_loss')} sampai
{tautan('GRU', 'adam')}.
""")

    d.teks(f"## {J[7]}")
    d.tabel(["Urutan", "Bagian Gambar 7", "Rumus", "Fungsinya"],
            [["1. Update gate", "σ kiri", "zₜ (23)", "berapa persen memori lama dipertahankan"],
             ["2. Reset gate", "σ tengah", "rₜ (24)", "berapa banyak memori lama dipakai menyusun kandidat"],
             ["3. Kandidat", "tanh", "h̃ₜ (25)", "usulan isi memori yang baru"],
             ["4. Campuran", "1−, ×, +", "hₜ = zₜ ⊙ hₜ₋₁ + (1 − zₜ) ⊙ h̃ₜ (26)", "memori baru sekaligus keluaran hari ini"],
             ["5. Prediksi", "-", f"ŷ′ = h{sb(T)}W_y + b_y", "dipakai sekali, setelah hari terakhir"]])

    d.teks(f"## {J[8]}")
    d.teks(f"""
1. **Satu memori saja.** Pada GRU {u} neuron ini, memori setiap hari hanyalah vektor hₜ
   berisi {u} angka; tidak ada cₜ seperti pada LSTM.
2. **Gambar 7 dan `reset_after=True`.** Gambar 7 menggambar rₜ mengalikan hₜ₋₁ sebelum
   masuk tanh, sedangkan Keras mengalikan rₜ setelah hₜ₋₁ dikali U_h. Saran kalimat:

   > Gambar 7 merupakan ilustrasi konseptual; pada implementasi Keras yang digunakan dalam
   > penelitian ini (reset_after=True), reset gate diterapkan setelah perkalian hₜ₋₁ dengan
   > matriks bobot rekuren, sebagaimana persamaan (25).
""")
    return d.isi()


def alur_sel(sk, kt, info) -> str:
    return _alur_lstm(sk, kt, info) if sk.model == "LSTM" else _alur_gru(sk, kt, info)


# =========================================================================== #
# ISI BERKAS: LANGKAH 1-2, FORWARD PASS SATU BATCH DAN LOSS (Tahap 7.7 dan 8.7)
# =========================================================================== #
def forward_loss(sk, kt, info) -> str:
    m = sk.model
    J = {1: "1. Satu Siklus Pelatihan dan Letaknya di Notebook", 2: "2. Dari Satu Sampel ke Satu Batch",
         3: "3. Langkah 1: Forward Pass 32 Sampel", 4: "4. Langkah 2: Loss MSE",
         5: "5. Apa Artinya Angka Ini?", 6: "6. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Di {tautan(m, 'alur_sel')} kita sudah menghitung satu sampel melewati sel {m}. Pelatihan
yang sesungguhnya tidak berhenti di situ. Keras mengulang siklus empat langkah berikut
ribuan kali, dan bagian ini membahas **langkah 1 dan 2** pada iterasi pertama model {m}
utama.
""")
    _pengantar_model_utama(d, sk, kt, info)
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.tabel(["Langkah", "Isi", "Dibahas di"],
            [["1. Forward pass", "hitung prediksi ŷ′ untuk satu batch", label(m, "forward_loss")],
             ["2. Loss", "ukur seberapa jauh ŷ′ dari y′ (MSE, persamaan 28)", label(m, "forward_loss")],
             ["3. BPTT", "hitung gradien loss terhadap setiap bobot dan bias", label(m, "bptt")],
             ["4. Update Adam", "geser setiap parameter sedikit ke arah yang menurunkan loss", label(m, "adam")]])
    total = kt["iter_epoch"] * info["epoch"]
    d.teks(f"""
Satu putaran empat langkah ini disebut **satu iterasi**. Model {m} terbaik dilatih
{ribu(info['epoch'])} epoch × {kt['iter_epoch']} iterasi per epoch = **{ribu(total)} iterasi**. Yang kita
hitung di sini adalah iterasi pertama, persis seperti yang dikerjakan `model.fit()` saat
model ini dilatih.
""")

    d.teks(f"## {J[2]}")
    tb = info["tanggal_target"]
    d.teks(f"""
Keras tidak memperbarui bobot setelah setiap sampel, tetapi setelah setiap **batch** berisi
{sk.batch} sampel. Karena data runtun waktu tidak diacak (`shuffle=False`), batch pertama
berisi 32 jendela pertama data latih, yaitu target tanggal **{tgl(tb[0])} s.d.
{tgl(tb[sk.batch - 1])}**. Dengan {ribu(kt['n_latih'])} sampel latih, satu epoch terdiri dari
{kt['iter_epoch'] - 1} batch × {sk.batch} sampel + 1 batch × {kt['batch_akhir']} sampel =
{kt['iter_epoch']} batch.

Ke-32 sampel itu dihitung **bersamaan** dengan bobot yang sama, masing-masing persis seperti
sampel pertama di {label(m, 'alur_sel')}. Sampel ke-2 adalah jendela yang digeser satu hari,
dan seterusnya.
""")

    d.teks(f"## {J[3]}")
    yb, yhb = sk.maju1["y"].ravel(), sk.maju1["yhat"].ravel()
    err = yb - yhb
    d.teks(f"Hasil forward pass seluruh batch pertama dengan bobot awal:")
    d.tabel(["No", "Tanggal target", "y′ (target)", "ŷ′ (prediksi)", "e = y′ − ŷ′", "e²"],
            [[str(n + 1), tgl(tb[n]), a(yb[n]), a(yhb[n]), kr(err[n]), e(err[n] ** 2)] for n in range(len(yb))])
    d.teks(f"""
Baris pertama sama dengan hasil {label(m, 'alur_sel')} (ŷ′ = {a(yhb[0])}). Prediksi ke-32
sampel cocok dengan `model.predict()` Keras, selisih terbesarnya {sk.selisih_batch:.1e}.
""")

    d.teks(f"## {J[4]}")
    d.teks(r"$$L = \frac{1}{B}\sum_{n=1}^{B}\left(y'_n - \hat{y}'_n\right)^2 \qquad (28)$$")
    jum = float(np.sum(err ** 2))
    d.kode([f"Σ e² = {e(err[0] ** 2)} + {e(err[1] ** 2)} + … + {e(err[-1] ** 2)}   ({len(err)} suku)",
            f"     = {e(jum)}",
            f"L    = {e(jum)} / {len(err)} = {e(sk.maju1['L'])}",
            "",
            f"Loss dari Keras untuk batch yang sama = {e(sk.loss_keras)}   (selisih {sk.selisih_loss:.1e})"])
    rmse_usd = kt["rentang"] * math.sqrt(sk.maju1["L"])
    d.teks(f"""
Supaya terasa maknanya, ubah ke dolar. Karena normalisasi min-max linear, RMSE dalam USD =
(x_max − x_min) × √L = {usd(kt['rentang'])} × √{e(sk.maju1['L'])} = **{usd(rmse_usd)} USD**.
Jadi pada awal pelatihan, prediksi model untuk batch ini rata-rata meleset sekitar
{usd(rmse_usd)} USD.
""")

    d.teks(f"## {J[5]}")
    rendah, nb_ = int(np.sum(err > 0)), len(yb)
    if rendah == nb_:
        arah = f"Semua {nb_} sampel diprediksi terlalu rendah (e > 0)."
    elif rendah == 0:
        arah = f"Semua {nb_} sampel diprediksi terlalu tinggi (e < 0)."
    else:
        arah = (f"Sebanyak {rendah} dari {nb_} sampel diprediksi terlalu rendah (e > 0) dan "
                f"{nb_ - rendah} sampel terlalu tinggi (e < 0).")
    if yhb.mean() < 0:
        arah += (" Rata-rata prediksinya bahkan negatif, artinya di bawah harga terendah data latih "
                 f"({usd(kt['x_min'])} USD).")
    kecil = math.sqrt(sk.maju1["L"]) < 0.05
    d.teks(f"""
- Rata-rata prediksi batch ini {a(yhb.mean())}, sedangkan rata-rata targetnya {a(yb.mean())}.
  {arah} Tanda galat inilah yang nanti menentukan arah perbaikan bobot di langkah 3.
- Bobot masih acak, jadi model belum "membaca" pola apa pun dari data.{' Loss yang tampak kecil di batch ini lebih karena kebetulan: harga di awal data latih dekat dengan harga terendah (nilai ternormalisasinya dekat 0), dan model yang belum dilatih juga mengeluarkan angka di sekitar 0. Batch lain yang harganya lebih tinggi bisa memberi loss jauh lebih besar.' if kecil else ''}
- Angka L ini **bukan** titik pertama kurva loss di {label(m, 'kurva_loss')}. Loss latih
  epoch 1 pada kurva adalah rata-rata loss dari {kt['iter_epoch']} batch selama epoch pertama,
  ketika bobotnya sudah ikut berubah di setiap batch.
""")

    d.teks(f"## {J[6]}")
    d.teks(f"""
Saran kalimat (subbab 1.5.11, setelah persamaan 28):

> Pelatihan dilakukan per mini-batch berukuran {sk.batch} sampel tanpa pengacakan urutan,
> sehingga satu epoch terdiri atas {kt['iter_epoch']} iterasi. Pada setiap iterasi, nilai loss
> dihitung sebagai rata-rata kuadrat galat {sk.batch} sampel dalam batch tersebut.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: LANGKAH 3, BPTT (Tahap 7.8 dan 8.8)
# =========================================================================== #
def _ringkas_gradien(d, sk):
    g = sk.mundur1["g"]
    baris = []
    for k in kunci_bobot(sk.model):
        v = g[k]
        contoh = v.flat[0]
        baris.append([k.replace("_in", "(in)").replace("_rec", "(rec)"), str(v.shape), ribu(v.size),
                      e(contoh), e(np.mean(np.abs(v))), e(np.max(np.abs(v)))])
    d.tabel(["Parameter", "Bentuk", "Elemen", "Contoh: elemen pertama", "Rata-rata \\|g\\|", "\\|g\\| terbesar"], baris)


def _awal_bptt(d, sk, kt, info):
    m = sk.model
    dy = sk.mundur1["dy"].ravel()
    yb, yhb = sk.maju1["y"].ravel(), sk.maju1["yhat"].ravel()
    hT = sk.maju1["langkah"][-1]["h"]
    T = sk.lookback
    d.teks(f"""
Gradien menjawab pertanyaan sederhana: **kalau satu parameter dinaikkan sedikit, loss naik
atau turun, dan seberapa cepat?** Tandanya memberi arah (positif berarti parameter harus
diturunkan), besarnya memberi seberapa sensitif loss terhadap parameter itu.

Karena loss bergantung pada parameter lewat rantai yang panjang (bobot → gerbang → memori →
hidden state → ... → prediksi → loss), gradien dihitung dengan **aturan rantai**, dimulai
dari belakang (loss) dan berjalan mundur ke hari pertama. Itulah arti *backpropagation
through time*. Dua turunan dasar yang dipakai berulang kali:

- σ′(a) = σ(a)(1 − σ(a))
- tanh′(a) = 1 − tanh²(a)
""")
    d.teks("### Mulai dari loss")
    d.teks(f"""
Dari L = (1/{sk.batch}) Σ (y′ₙ − ŷ′ₙ)², turunannya terhadap prediksi sampel ke-n adalah
∂L/∂ŷ′ₙ = 2(ŷ′ₙ − y′ₙ)/{sk.batch}. Untuk tiga sampel pertama:
""")
    d.kode([f"  sampel {n + 1}: 2 × ({a(yhb[n])} − {a(yb[n])}) / {sk.batch} = {e(dy[n])}" for n in range(3)]
           + ["  … (sampel lainnya dengan cara yang sama)"])
    d.teks("### Lapisan Dense")
    d.kode([f"∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = {e(dy[0])} + {ke(dy[1])} + … ({sk.batch} suku) = {e(sk.mundur1['g']['b_y'][0])}",
            f"∂L/∂W_y[1] = Σₙ ∂L/∂ŷ′ₙ × h{sb(T)},ₙ[1]",
            f"           = {ke(dy[0])}×{kr(hT[0, 0])} + {ke(dy[1])}×{kr(hT[1, 0])} + … = {e(sk.mundur1['g']['W_y'][0, 0])}"])
    d.teks(f"""
Bias b_y menerima gradien paling "polos": jumlah semua ∂L/∂ŷ′. Bobot W_y[j] mendapat gradien
yang juga ditimbang dengan nilai h{sb(T)}[j]: neuron yang keluarannya besar ikut
bertanggung jawab lebih besar atas galat.
""")


def _cek_bptt(d, sk):
    d.teks(f"""
Seluruh {ribu(sk.n_param)} gradien dihitung dengan cara di atas lalu dibandingkan dengan dua
cara lain:

1. **TensorFlow `GradientTape`** (diferensiasi otomatis yang dipakai `model.fit`): selisih
   terbesar di antara semua parameter = **{sk.selisih_gradien:.1e}** (gradien terbesar
   {e(sk.skala_gradien)}).
2. **Turunan numerik**: naikkan dan turunkan satu parameter sebesar 10⁻⁶, hitung ulang loss,
   lalu (L₊ − L₋)/(2 × 10⁻⁶). Cara ini tidak memakai rumus turunan sama sekali.
""")
    d.tabel(["Parameter", "BPTT manual", "Turunan numerik"],
            [[f"{nm.replace('_in', '(in)').replace('_rec', '(rec)')}{list(int(i) + 1 for i in idx)}",
              e(g_), e(n_)] for nm, idx, g_, n_ in sk.cek_numerik])
    d.teks("Ketiga cara memberi angka yang sama, jadi rumus BPTT di atas memang rumus yang dipakai Keras.")


def _bptt_lstm(sk, kt, info):
    S, R = sk.maju1["langkah"], sk.mundur1["rincian"]
    p, T, u = sk.p0, sk.lookback, sk.n_unit
    J = {1: "1. Ide Dasarnya", 2: f"2. Masuk ke Sel pada t = {T} (Sampel 1, Neuron ke-1)",
         3: f"3. Mundur ke t = {T - 1}: Dua Jalan Sinyal Kesalahan", 4: "4. Seberapa Jauh Sinyal Mengalir Mundur",
         5: "5. Gradien Total Setiap Bobot dan Bias", 6: "6. Pemeriksaan", 7: "7. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Langkah 3 dari siklus pelatihan: menghitung gradien loss batch pertama
({tautan('LSTM', 'forward_loss')}) terhadap **seluruh {ribu(sk.n_param)} parameter** model LSTM
utama. Rinciannya kita ikuti untuk sampel 1 dan neuron ke-1, sisanya dikerjakan komputer
dengan rumus yang sama.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _awal_bptt(d, sk, kt, info)

    d.teks(f"## {J[2]}")
    s, r = S[-1], R[-1]
    n, j = 0, 0
    v = lambda arr: float(arr[n, j])   # noqa: E731
    d.teks(f"""
Sinyal kesalahan yang masuk ke hidden state neuron ke-1 pada hari terakhir adalah
δh{sb(T)} = ∂L/∂ŷ′ × W_y[1]. Dari sana sinyal dibagi ke gerbang-gerbang (nilai fₜ, iₜ, c̃ₜ, oₜ,
cₜ diambil dari forward pass):
""")
    d.kode([
        f"δh{sb(T)}  = ∂L/∂ŷ′₁ × W_y[1] = {ke(sk.mundur1['dy'][n, 0])} × {kr(p['W_y'][j, 0])} = {e(v(r['dh']))}",
        "",
        "Output gate (Gambar 6), dari h = o × tanh(c):",
        f"  ∂L/∂o{sb(T)} = δh × tanh(c{sb(T)}) = {ke(v(r['dh']))} × {kr(v(s['tc']))} = {e(v(r['d_o']))}",
        f"  δa_o    = ∂L/∂o × o(1 − o) = {ke(v(r['d_o']))} × {a(v(s['o']) * (1 - v(s['o'])))} = {e(v(r['delta_o']))}",
        "",
        "Cell state, dari h = o × tanh(c) (belum ada sinyal dari hari sesudahnya):",
        f"  δc{sb(T)} = δh × o × (1 − tanh²c) = {ke(v(r['dh']))} × {a(v(s['o']))} × {a(1 - v(s['tc']) ** 2)}"
        f" = {e(v(r['dc']))}",
        "",
        f"Forget gate (Gambar 3), dari c = f × c_lama + i × c̃:",
        f"  ∂L/∂f{sb(T)} = δc × c{sb(T - 1)} = {ke(v(r['dc']))} × {kr(v(s['c_prev']))} = {e(v(r['d_f']))}",
        f"  δa_f    = ∂L/∂f × f(1 − f) = {ke(v(r['d_f']))} × {a(v(s['f']) * (1 - v(s['f'])))} = {e(v(r['delta_f']))}",
        "Input gate (Gambar 4):",
        f"  ∂L/∂i{sb(T)} = δc × c̃ = {ke(v(r['dc']))} × {kr(v(s['cc']))} = {e(v(r['d_i']))}",
        f"  δa_i    = ∂L/∂i × i(1 − i) = {e(v(r['delta_i']))}",
        "Kandidat:",
        f"  ∂L/∂c̃{sb(T)} = δc × i = {ke(v(r['dc']))} × {a(v(s['i']))} = {e(v(r['d_cc']))}",
        f"  δa_c    = ∂L/∂c̃ × (1 − c̃²) = {ke(v(r['d_cc']))} × {a(1 - v(s['cc']) ** 2)} = {e(v(r['delta_c']))}",
    ])
    x1 = float(s["x"][n, 0])
    d.teks(f"""
Angka δa inilah "jatah kesalahan" setiap gerbang. Dari δa, gradien bobotnya tinggal dikali
masukannya. Misalnya, sumbangan sampel 1 hari ke-{T} untuk bobot **Close Price → forget
gate neuron ke-1**:

∂L/∂W_f[1, 1] (sumbangan ini) = δa_f × x{sb(T)},₁ = {ke(v(r['delta_f']))} × {a(x1)} = {e(v(r['delta_f']) * x1)}

dan untuk biasnya cukup δa_f itu sendiri = {e(v(r['delta_f']))}, karena ∂a/∂b = 1.
""")

    d.teks(f"## {J[3]}")
    r6 = R[-2]
    tol, jalur = float(r6["dc_lanjut"][n, j]), float(r6["jalur_h"][n, j])
    d.teks(f"""
Ke hari ke-{T - 1}, sinyal kesalahan untuk cell state neuron ke-1 datang lewat **dua jalan**:
""")
    d.kode([
        f"Jalan 1, lewat cell state (\"jalan tol\", Gambar 2): cukup dikali forget gate",
        f"  δc{sb(T)} × f{sb(T)} = {ke(v(r['dc']))} × {a(v(s['f']))} = {e(tol)}",
        f"Jalan 2, lewat hidden state: δh{sb(T - 1)} dikumpulkan dari 4 gerbang × {u} neuron",
        f"  δh{sb(T - 1)}[1] = Σ (δa_f U_f + δa_i U_i + δa_c U_c + δa_o U_o)[1, :] = {e(float(r6['dh'][n, j]))}",
        f"  lalu masuk ke cell state: δh{sb(T - 1)} × o{sb(T - 1)} × (1 − tanh²c{sb(T - 1)}) = {e(jalur)}",
        "",
        f"δc{sb(T - 1)} = {ke(tol)} + {ke(jalur)} = {e(tol + jalur)}",
    ])
    porsi = abs(tol) / (abs(tol) + abs(jalur)) if (tol or jalur) else 0.0
    d.teks(f"""
Untuk sampel dan neuron ini, {persen(porsi)} besar sinyal ke c{sb(T - 1)} datang dari jalan tol.
Jalan tol hanya melewati perkalian dengan f, tanpa matriks dan tanpa turunan aktivasi, jadi
sinyalnya tidak cepat mengecil. Inilah alasan LSTM bisa "mengingat" lebih lama daripada RNN
biasa. Proses yang sama diulang mundur sampai t = 1.
""")

    d.teks(f"## {J[4]}")
    rata_h = [float(np.mean(np.abs(R[t]["dh"]))) for t in range(T)]
    rata_c = [float(np.mean(np.abs(R[t]["dc"]))) for t in range(T)]
    d.teks(f"Rata-rata besar sinyal kesalahan (seluruh {sk.batch} sampel × {u} neuron) di setiap hari:")
    d.tabel(["t", "rata-rata \\|δhₜ\\|", "rata-rata \\|δcₜ\\|", f"\\|δcₜ\\| dibanding t = {T}"],
            [[str(t + 1), e(rata_h[t]), e(rata_c[t]), persen(rata_c[t] / rata_c[-1]) if rata_c[-1] else "-"]
             for t in reversed(range(T))])
    d.teks(f"""
Semakin jauh ke belakang, sinyalnya {'mengecil' if rata_c[0] < rata_c[-1] else 'tidak mengecil'}: di hari
pertama tinggal {persen(rata_c[0] / rata_c[-1]) if rata_c[-1] else '-'} dari sinyal di hari terakhir. Artinya
hari-hari terakhir dalam jendela paling berpengaruh pada perbaikan bobot, tetapi hari pertama
pun masih ikut menyumbang.
""")

    d.teks(f"## {J[5]}")
    d.teks(f"""
Gradien satu parameter adalah **jumlah sumbangan dari semua sampel dan semua hari**:

$$\\frac{{\\partial L}}{{\\partial \\mathbf{{W}}_g}} = \\sum_{{n=1}}^{{{sk.batch}}}\\sum_{{t=1}}^{{{T}}} \\mathbf{{x}}_{{t,n}}^{{\\top}}\\,\\delta \\mathbf{{a}}_{{g,t,n}}
\\qquad
\\frac{{\\partial L}}{{\\partial \\mathbf{{U}}_g}} = \\sum_{{n}}\\sum_{{t}} \\mathbf{{h}}_{{t-1,n}}^{{\\top}}\\,\\delta \\mathbf{{a}}_{{g,t,n}}
\\qquad
\\frac{{\\partial L}}{{\\partial \\mathbf{{b}}_g}} = \\sum_{{n}}\\sum_{{t}} \\delta \\mathbf{{a}}_{{g,t,n}}$$

untuk g ∈ {{f, i, c, o}}. Pada t = 1 bagian U tidak mendapat sumbangan karena h₀ = 0.
Ringkasan seluruh gradien iterasi pertama:
""")
    _ringkas_gradien(d, sk)
    gb = {g: float(np.mean(np.abs(sk.mundur1["g"]["W_" + g]))) for g in "ifco"}
    terbesar = max(gb, key=gb.get)
    nama_g = {"i": "input gate", "f": "forget gate", "c": "kandidat", "o": "output gate"}
    d.teks(f"""
Lapisan Dense (W_y, b_y) menerima gradien terbesar karena paling dekat dengan loss. Di dalam
sel, rata-rata gradien bobot masukan terbesar ada di **{nama_g[terbesar]}**.
""")

    d.teks(f"## {J[6]}")
    _cek_bptt(d, sk)

    d.teks(f"## {J[7]}")
    d.teks("""
Saran kalimat (subbab 1.5.11):

> Gradien setiap bobot dan bias dihitung dengan backpropagation through time, yaitu aturan
> rantai yang dijalankan mundur dari loss ke time step terakhir hingga time step pertama.
> Pada LSTM, sinyal kesalahan untuk cell state mengalir melalui dua jalur: jalur langsung
> cₜ₋₁ → cₜ yang hanya dikalikan forget gate, dan jalur melalui hidden state. Jalur langsung
> inilah yang membuat gradien tidak cepat menghilang.
""")
    return d.isi()


def _bptt_gru(sk, kt, info):
    S, R = sk.maju1["langkah"], sk.mundur1["rincian"]
    p, T, u = sk.p0, sk.lookback, sk.n_unit
    J = {1: "1. Ide Dasarnya", 2: f"2. Masuk ke Sel pada t = {T} (Sampel 1, Neuron ke-1)",
         3: f"3. Mundur ke t = {T - 1}: Jalan Langsung dan Jalan Lewat Gerbang",
         4: "4. Seberapa Jauh Sinyal Mengalir Mundur", 5: "5. Gradien Total Setiap Bobot dan Bias",
         6: "6. Pemeriksaan", 7: "7. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Langkah 3 dari siklus pelatihan GRU: menghitung gradien loss batch pertama
({tautan('GRU', 'forward_loss')}) terhadap **seluruh {ribu(sk.n_param)} parameter** model GRU
utama. Polanya sama dengan LSTM ({tautan('LSTM', 'bptt')}); yang berbeda rumus di dalam sel.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    _awal_bptt(d, sk, kt, info)

    d.teks(f"## {J[2]}")
    s, r = S[-1], R[-1]
    n, j = 0, 0
    v = lambda arr: float(arr[n, j])   # noqa: E731
    d.kode([
        f"δh{sb(T)} = ∂L/∂ŷ′₁ × W_y[1] = {ke(sk.mundur1['dy'][n, 0])} × {kr(p['W_y'][j, 0])} = {e(v(r['dh']))}",
        "",
        "Update gate, dari h = z × h_lama + (1 − z) × h̃:",
        f"  ∂L/∂z{sb(T)} = δh × (h{sb(T - 1)} − h̃{sb(T)}) = {ke(v(r['dh']))} × ({a(v(s['h_prev']))} − {kr(v(s['hh']))})"
        f" = {e(v(r['d_z']))}",
        f"  δa_z    = ∂L/∂z × z(1 − z) = {ke(v(r['d_z']))} × {a(v(s['z']) * (1 - v(s['z'])))} = {e(v(r['delta_z']))}",
        "Kandidat:",
        f"  ∂L/∂h̃{sb(T)} = δh × (1 − z) = {ke(v(r['dh']))} × {a(1 - v(s['z']))} = {e(v(r['d_hh']))}",
        f"  δa_h    = ∂L/∂h̃ × (1 − h̃²) = {ke(v(r['d_hh']))} × {a(1 - v(s['hh']) ** 2)} = {e(v(r['delta_h']))}",
        "Reset gate, dari a_h = x W_h + b_h(in) + r × q, dengan q = h_lama U_h + b_h(rec):",
        f"  ∂L/∂r{sb(T)} = δa_h × q = {ke(v(r['delta_h']))} × {kr(v(s['mh_h']))} = {e(v(r['d_r']))}",
        f"  δa_r    = ∂L/∂r × r(1 − r) = {ke(v(r['d_r']))} × {a(v(s['r']) * (1 - v(s['r'])))} = {e(v(r['delta_r']))}",
        f"  ∂L/∂q   = δa_h × r = {ke(v(r['delta_h']))} × {a(v(s['r']))} = {e(v(r['d_q']))}",
    ])
    d.teks(f"""
Perhatikan bahwa kandidat punya dua "jatah kesalahan": δa_h untuk jalur masukan (dipakai
W_h dan b_h(in)) dan ∂L/∂q = δa_h × r untuk jalur rekuren (dipakai U_h dan b_h(rec)). Inilah
sebabnya dua bias kandidat GRU mendapat gradien berbeda ({tautan('GRU', 'bias')}).
""")

    d.teks(f"## {J[3]}")
    r6 = R[-1]
    langsung = float(r6["dh_langsung"][n, j])
    lewat = float(r6["dh_prev"][n, j]) - langsung
    d.kode([
        f"Jalan 1, langsung (z × h_lama pada Gambar 7): cukup dikali update gate",
        f"  δh{sb(T)} × z{sb(T)} = {ke(v(r['dh']))} × {a(v(s['z']))} = {e(langsung)}",
        f"Jalan 2, lewat tiga gerbang × {u} neuron (δa_z U_z + δa_r U_r + ∂L/∂q U_h)",
        f"  = {e(lewat)}",
        "",
        f"δh{sb(T - 1)}[1] = {ke(langsung)} + {ke(lewat)} = {e(langsung + lewat)}",
    ])
    porsi = abs(langsung) / (abs(langsung) + abs(lewat)) if (langsung or lewat) else 0.0
    d.teks(f"""
Untuk sampel dan neuron ini, {persen(porsi)} besar sinyal ke h{sb(T - 1)} datang lewat jalan
langsung. Jalan ini hanya dikali z, tanpa matriks, jadi perannya mirip jalan tol cell state
pada LSTM.
""")

    d.teks(f"## {J[4]}")
    rata_h = [float(np.mean(np.abs(R[t]["dh"]))) for t in range(T)]
    d.tabel(["t", "rata-rata \\|δhₜ\\|", f"dibanding t = {T}"],
            [[str(t + 1), e(rata_h[t]), persen(rata_h[t] / rata_h[-1]) if rata_h[-1] else "-"]
             for t in reversed(range(T))])
    d.teks(f"""
Di hari pertama sinyalnya tinggal {persen(rata_h[0] / rata_h[-1]) if rata_h[-1] else '-'} dari hari
terakhir. Jadi hari-hari terakhir dalam jendela paling menentukan arah perbaikan bobot,
sedangkan hari-hari awal hanya ikut menyumbang sedikit.
""")

    d.teks(f"## {J[5]}")
    d.teks(f"""
Seperti LSTM, gradien setiap parameter adalah jumlah sumbangan dari {sk.batch} sampel × {T} hari:
∂L/∂W_g = Σ xᵀδa_g dan ∂L/∂b_g(in) = Σ δa_g untuk g ∈ {{z, r, h}}; untuk jalur rekuren
∂L/∂U_g = Σ hₜ₋₁ᵀ δ_g(rec) dan ∂L/∂b_g(rec) = Σ δ_g(rec), dengan δ_z(rec) = δa_z,
δ_r(rec) = δa_r, dan δ_h(rec) = ∂L/∂q.
""")
    _ringkas_gradien(d, sk)
    g = sk.mundur1["g"]
    kembar = float(max(np.max(np.abs(g["b_z_in"] - g["b_z_rec"])), np.max(np.abs(g["b_r_in"] - g["b_r_rec"]))))
    beda = float(np.max(np.abs(g["b_h_in"] - g["b_h_rec"])))
    assert kembar < 1e-12 and beda > 0
    d.teks(f"""
Lihat baris bias: gradien b_z(in) sama persis dengan b_z(rec), begitu pula b_r(in) dengan
b_r(rec) (selisihnya {'tepat 0' if kembar == 0 else f'{kembar:.0e}'}), sedangkan b_h(in) dan b_h(rec) berbeda.
""")

    d.teks(f"## {J[6]}")
    _cek_bptt(d, sk)

    d.teks(f"## {J[7]}")
    d.teks("""
Saran kalimat (subbab 1.5.11):

> Pada GRU, sinyal kesalahan untuk hidden state hari sebelumnya mengalir melalui jalur
> langsung zₜ ⊙ hₜ₋₁ yang hanya dikalikan update gate, serta melalui ketiga lapisan
> berbobot. Jalur langsung ini berperan serupa dengan jalur cell state pada LSTM.
""")
    return d.isi()


def bptt(sk, kt, info) -> str:
    return _bptt_lstm(sk, kt, info) if sk.model == "LSTM" else _bptt_gru(sk, kt, info)


# =========================================================================== #
# ISI BERKAS: LANGKAH 4, UPDATE ADAM (Tahap 7.9 dan 8.9)
# =========================================================================== #
def _contoh_parameter(model):
    if model == "LSTM":
        return [("b_y", (0,)), ("W_y", (0, 0)), ("b_f", (0,)), ("W_f", (0, 0)), ("U_f", (0, 0)), ("b_i", (0,))]
    return [("b_y", (0,)), ("W_y", (0, 0)), ("b_z_in", (0,)), ("W_z", (0, 0)), ("U_z", (0, 0)), ("b_h_rec", (0,))]


def _nama_param(nm, idx):
    return f"{nm.replace('_in', '(in)').replace('_rec', '(rec)')}{list(int(i) + 1 for i in idx)}"


def adam(sk, kt, info, bobot_akhir) -> str:
    m = sk.model
    p_akhir = pecah_bobot(m, bobot_akhir)
    J = {1: "1. Rumus Adam", 2: "2. Iterasi k = 1, Satu Parameter (b_y)", 3: "3. Iterasi k = 1, Semua Parameter",
         4: "4. Apakah Loss Langsung Turun?", 5: "5. Iterasi k = 2: Momentum Mulai Bekerja",
         6: "6. Dicocokkan dengan model.fit Keras", 7: "7. Siklus Diulang Sampai Selesai",
         8: "8. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Langkah terakhir siklus pelatihan: memakai gradien dari {tautan(m, 'bptt')} untuk
memperbaiki **seluruh {ribu(sk.n_param)} parameter** model {m} utama dengan optimizer Adam,
lalu memeriksa bahwa hasilnya sama dengan yang dilakukan `model.fit()`.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks(f"""
Adam bekerja **per parameter**: setiap bobot dan bias punya catatan m dan v sendiri, yang
mulai dari nol. Pada iterasi ke-k dengan gradien gₖ:

- **Momen pertama** (persamaan 29): mₖ = β₁mₖ₋₁ + (1 − β₁)gₖ, rata-rata bergerak arah gradien
  (β₁ = 0.9, kira-kira merangkum 10 gradien terakhir). Fungsinya seperti momentum, supaya
  langkah tidak gampang terombang-ambing oleh satu batch.
- **Momen kedua** (persamaan 29): vₖ = β₂vₖ₋₁ + (1 − β₂)gₖ², rata-rata bergerak besarnya
  gradien (β₂ = 0.999). Dipakai sebagai pembagi supaya ukuran langkah tiap parameter seimbang.
- **Koreksi bias** (persamaan 30): m̂ₖ = mₖ/(1 − β₁ᵏ), v̂ₖ = vₖ/(1 − β₂ᵏ), karena m dan v mulai
  dari nol sehingga di awal nilainya terlalu kecil.
- **Update** (persamaan 31): θₖ = θₖ₋₁ − η m̂ₖ/(√v̂ₖ + ε), dengan η = {kt['lr']} dan ε = 10⁻⁷.

Keras menulis rumus yang sama dalam bentuk ringkas: α = η√(1 − β₂ᵏ)/(1 − β₁ᵏ), lalu
θ ← θ − α m/(√v + ε). Bedanya hanya letak ε, dan pengaruhnya sangat kecil. Perhitungan di
bawah mengikuti bentuk Keras agar hasilnya bisa dicocokkan sampai digit terakhir.
""")

    d.teks(f"## {J[2]}")
    g1 = float(sk.mundur1["g"]["b_y"][0])
    m1, v1 = float(sk.m1["b_y"][0]), float(sk.v1["b_y"][0])
    dl1 = float(sk.delta1["b_y"][0])
    d.kode([f"θ₀ = b_y awal = {a(sk.p0['b_y'][0])}      g₁ = ∂L/∂b_y = {e(g1)}   ({label(m, 'bptt')})",
            "",
            f"m₁ = 0.9 × 0 + 0.1 × {ke(g1)} = {e(m1)}",
            f"v₁ = 0.999 × 0 + 0.001 × {ke(g1)}² = {e(v1)}",
            f"α₁ = {kt['lr']} × √(1 − 0.999) / (1 − 0.9) = {kt['lr']} × {a(math.sqrt(0.001))} / 0.1 = {a(sk.alpha1, 8)}",
            f"Δ  = α₁ × m₁ / (√v₁ + ε) = {a(sk.alpha1, 8)} × {ke(m1)} / ({e(math.sqrt(v1))} + 1e-7) = {e(dl1)}",
            f"θ₁ = θ₀ − Δ = {a(sk.p0['b_y'][0])} − {ke(dl1)} = {a(sk.p1['b_y'][0])}"])
    ambang = EPS / math.sqrt(1 - B2)
    d.teks(f"""
Gradien b_y {'negatif' if g1 < 0 else 'positif'}, artinya prediksi rata-rata terlalu
{'rendah' if g1 < 0 else 'tinggi'}, jadi b_y {'dinaikkan' if g1 < 0 else 'diturunkan'}. Besar langkahnya
hampir tepat η = {kt['lr']}. Ini bukan kebetulan. Kalau rumus di atas disederhanakan untuk k = 1
(m₁ = 0.1g, √v₁ = 0.0316|g|), hasilnya:

$$\\Delta_1 = \\eta\\,\\frac{{g}}{{|g| + {ambang * 1e6:.2f} \\times 10^{{-6}}}}$$

Angka {ambang * 1e6:.2f} × 10⁻⁶ = ε/√(1 − β₂) adalah "ambang" kecil. Kalau |g| jauh lebih besar dari ambang
ini, langkahnya ≈ η × tanda(g), berapa pun besar gradiennya. Kalau |g| sekecil ambang atau
lebih kecil, langkahnya ikut mengecil.
""")

    d.teks(f"## {J[3]}")
    baris = []
    for nm, idx in _contoh_parameter(m):
        baris.append([_nama_param(nm, idx), a(sk.p0[nm][idx]), e(sk.mundur1["g"][nm][idx]),
                      e(sk.m1[nm][idx]), e(sk.v1[nm][idx]), ke(sk.delta1[nm][idx]), a(sk.p1[nm][idx])])
    d.tabel(["Parameter", "θ₀", "g₁", "m₁", "v₁", "Δ", "θ₁ = θ₀ − Δ"], baris)
    semua_delta = np.abs(np.concatenate([np.ravel(v) for v in sk.delta1.values()]))
    lr_ = kt["lr"]
    kelompok = [("≥ 0.99η (hampir tepat η)", semua_delta >= 0.99 * lr_),
                ("0.5η sampai 0.99η", (semua_delta >= 0.5 * lr_) & (semua_delta < 0.99 * lr_)),
                ("0.1η sampai 0.5η", (semua_delta >= 0.1 * lr_) & (semua_delta < 0.5 * lr_)),
                ("< 0.1η", semua_delta < 0.1 * lr_)]
    d.teks(f"Sebaran besar langkah untuk seluruh {ribu(sk.n_param)} parameter pada iterasi pertama:")
    d.tabel(["Besar langkah \\|Δ\\|", "Jumlah parameter", "Porsi"],
            [[nm, ribu(int(msk.sum())), persen(float(msk.mean()))] for nm, msk in kelompok])
    setengah = float(np.mean(semua_delta >= 0.5 * lr_))
    d.teks(f"""
Jadi {persen(setengah)} parameter bergeser lebih dari separuh η, padahal gradiennya bisa berbeda
ribuan kali lipat (lihat tabel gradien di {label(m, 'bptt')}). Di iterasi pertama, Adam belum
"tahu" parameter mana yang penting; ia hanya melihat tanda gradien.
""")

    d.teks(f"## {J[4]}")
    L0, L1 = sk.maju1["L"], sk.L1_sesudah
    if L1 < L0:
        cerita = (f"Loss batch pertama turun dari {e(L0)} menjadi {e(L1)} ({persen(1 - L1 / L0)} lebih kecil). "
                  "Satu langkah kecil pada semua parameter sudah cukup memperbaiki prediksi batch ini.")
    else:
        cerita = (f"Loss batch pertama justru **naik**, dari {e(L0)} menjadi {e(L1)}. Ini wajar di awal "
                  "pelatihan dengan Adam. Seperti terlihat di bagian 3, pada iterasi pertama banyak parameter "
                  "digeser hampir sama jauh, padahal sebagian besar gradiennya sangat kecil. Untuk parameter "
                  "seperti itu langkahnya kebesaran, dan kalau ribuan parameter bergeser bersamaan, efek "
                  "gabungannya bisa melewati titik terbaik (overshoot). Begitu m dan v sudah merekam beberapa gradien, "
                  "ukuran langkah tiap parameter menyesuaikan diri, dan loss turun dengan jelas dalam "
                  f"beberapa epoch (lihat {label(m, 'kurva_loss')}).")
    d.teks(f"Hitung ulang forward pass batch pertama dengan bobot θ₁:\n\n{cerita}")

    d.teks(f"## {J[5]}")
    g2 = float(sk.mundur2["g"]["b_y"][0])
    m2, v2, dl2 = float(sk.m2["b_y"][0]), float(sk.v2["b_y"][0]), float(sk.delta2["b_y"][0])
    tb = info["tanggal_target"]
    d.teks(f"""
Iterasi kedua memakai **batch kedua** (target {tgl(tb[sk.batch])} s.d. {tgl(tb[2 * sk.batch - 1])}) dan
bobot θ₁. Forward pass dan BPTT diulang dengan cara yang sama: loss batch kedua =
{e(sk.maju2['L'])}. Untuk b_y:
""")
    d.kode([f"g₂ = {e(g2)}",
            f"m₂ = 0.9 × {ke(m1)} + 0.1 × {ke(g2)} = {e(m2)}",
            f"v₂ = 0.999 × {ke(v1)} + 0.001 × {ke(g2)}² = {e(v2)}",
            f"α₂ = {kt['lr']} × √(1 − 0.999²) / (1 − 0.9²) = {a(sk.alpha2, 8)}",
            f"Δ  = α₂ × m₂ / (√v₂ + ε) = {e(dl2)}",
            f"θ₂ = {a(sk.p1['b_y'][0])} − {ke(dl2)} = {a(sk.p2['b_y'][0])}"])
    searah = (g1 > 0) == (g2 > 0)
    d.teks(f"""
{'Gradien kedua searah dengan yang pertama, jadi momentum m₂ makin kuat dan b_y terus bergerak ke arah yang sama.' if searah else 'Gradien kedua berlawanan arah dengan yang pertama. Momentum m₂ meredam perubahan arah ini, sehingga b_y tidak langsung berbalik arah sepenuhnya.'}
Mulai iterasi ini ukuran langkah tidak lagi seragam: setiap parameter punya m dan v sendiri.
""")

    d.teks(f"## {J[6]}")
    d.teks(f"""
Sebagai pembuktian, notebook membangun model {m} yang sama (seed 42), menjalankan
`model.fit()` dengan pengaturan penelitian (batch {sk.batch}, `shuffle=False`), lalu mengambil
bobot Keras tepat setelah batch ke-1 dan ke-2. Hasilnya dibandingkan dengan perhitungan
manual di atas untuk seluruh {ribu(sk.n_param)} parameter:
""")
    d.tabel(["Setelah", "Selisih terbesar manual vs Keras"],
            [["iterasi k = 1 (batch 1)", f"{sk.selisih_adam1:.1e}"],
             ["iterasi k = 2 (batch 2)", f"{sk.selisih_adam2:.1e}"]])
    d.teks("Selisihnya hanya sisa pembulatan (Keras menghitung dengan float32), jadi siklus "
           "forward → loss → BPTT → Adam yang dihitung manual memang siklus yang dijalankan Keras.")

    d.teks(f"## {J[7]}")
    total = kt["iter_epoch"] * info["epoch"]
    baris = []
    for nm, idx in _contoh_parameter(m)[:3]:
        baris.append([_nama_param(nm, idx), a(sk.p0[nm][idx]), a(sk.p1[nm][idx]), a(sk.p2[nm][idx]),
                      a(p_akhir[nm][idx])])
    d.teks(f"""
Siklus yang sama diulang untuk batch ke-3, ke-4, dan seterusnya sampai batch ke-{kt['iter_epoch']}
(satu epoch), lalu diulang lagi selama {ribu(info['epoch'])} epoch. Totalnya **{ribu(total)} kali update**.
Perjalanan beberapa parameter:
""")
    d.tabel(["Parameter", "Awal", "k = 1", "k = 2", f"Akhir (k = {ribu(total)})"], baris)
    d.teks(f"""
Bobot akhir inilah bobot model {m} terbaik yang dipakai untuk prediksi (bentuknya di
Tahap {7 if m == 'LSTM' else 8}.5 dan forward pass-nya diverifikasi di {label(m, 'forward_pass')}).
""")

    d.teks(f"## {J[8]}")
    d.teks(f"""
Saran kalimat (subbab 1.5.11, setelah persamaan 31):

> Pada iterasi pertama, Adam menggeser setiap parameter ke arah yang berlawanan dengan
> tanda gradiennya dengan besar langkah mendekati learning rate η = {kt['lr']}, kecuali untuk
> parameter yang gradiennya sangat kecil. Pada iterasi berikutnya,
> momen pertama dan kedua yang terakumulasi membuat ukuran langkah setiap parameter
> menyesuaikan diri dengan riwayat gradiennya masing-masing. Model {m} terbaik
> diperbarui sebanyak {kt['iter_epoch']} × {ribu(info['epoch'])} = {ribu(total)} kali.
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
    d = Dokumen()
    d.teks(f"""
{tautan(model, 'forward_loss')} sampai {tautan(model, 'adam')} menghitung iterasi pertama dan kedua
pelatihan model {model} utama ({hasil['neuron']} neuron). Subbagian ini melihat hasil dari
**seluruh** iterasinya ({ribu(hasil['epoch'])} epoch) lewat kurva loss latih dan loss validasi.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks(f"""
Di akhir setiap epoch, Keras mencatat dua angka:

- **loss latih**: rata-rata MSE dari {kt['iter_epoch']} batch selama epoch itu
  ({kt['iter_epoch'] - 1} batch × {kt['batch']} sampel + 1 batch × {kt['batch_akhir']} sampel). Loss batch
  pertama yang dihitung manual di {label(model, 'forward_loss')} adalah salah satu penyusun
  loss latih epoch 1;
- **loss validasi**: MSE pada {kt['n_val']} sampel validasi memakai bobot akhir epoch.
  Data validasi **hanya diukur**, tidak dipakai untuk mengubah bobot.

Selama {ribu(hasil['epoch'])} epoch, Adam melakukan {kt['iter_epoch']} × {ribu(hasil['epoch'])} =
**{ribu(kt['iter_epoch'] * hasil['epoch'])} kali update**, masing-masing persis seperti yang
dihitung di {label(model, 'adam')}.
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
        posisi = (f"di epoch {k['epoch_min']}, dekat sekali dengan epoch terakhir; loss validasi akhir "
                  f"({k['val_akhir']:.8f}) hampir sama dengan minimumnya")
    else:
        posisi = (f"lebih awal, di epoch {k['epoch_min']}, lalu sedikit naik sampai epoch terakhir "
                  f"({k['val_min']:.8f} menjadi {k['val_akhir']:.8f})")
    d.teks(f"""
- Kalau **loss latih dan loss validasi turun bersama**, model sedang mempelajari pola yang
  berlaku umum.
- Kalau **loss latih terus turun tetapi loss validasi naik**, itu tanda *overfitting*: model
  mulai menghafal data latih.
- Pada model ini, loss latih turun {persen(turun)} dari epoch pertama, dan loss validasi
  mencapai minimum {posisi}.
""")

    d.teks(f"## {J[4]}")
    per = hasil["per_epoch"]
    d.teks(f"RMSE validasi model {hasil['neuron']} neuron untuk setiap jumlah epoch pada grid "
           f"(tabel tuning {model}):")
    d.tabel(["Epoch", "RMSE validasi (USD)"],
            [[ribu(ep), usd(per[ep]) + (" ← terbaik" if ep == hasil["epoch"] else "")] for ep in sorted(per)])
    e_maks = max(per)
    if e_maks == hasil["epoch"]:
        kalimat = (f"Pada neuron ini, epoch terbanyak ({ribu(e_maks)}) memberi RMSE validasi terkecil, "
                   "jadi pelatihan yang lebih lama masih membantu.")
    elif per[e_maks] > per[hasil["epoch"]]:
        kalimat = (f"Pada neuron ini, melatih sampai {ribu(e_maks)} epoch justru memperburuk RMSE validasi "
                   f"dibandingkan {ribu(hasil['epoch'])} epoch. Melatih lebih lama tidak selalu lebih baik; "
                   "kemungkinan besar model mulai terlalu menyesuaikan diri dengan data latih.")
    else:
        kalimat = "RMSE validasi berbeda-beda untuk setiap jumlah epoch."
    d.teks(f"{kalimat} Karena itulah jumlah epoch dipilih lewat data validasi, sedangkan data uji "
           "tetap netral sampai evaluasi akhir.")

    d.teks(f"## {J[5]}")
    rmse = kt["rentang"] * math.sqrt(k["val_akhir"])
    assert abs(rmse / hasil["rmse_val"] - 1) < 1e-3, "Rentang × √MSE harus sama dengan RMSE validasi"
    d.teks(f"""
Normalisasi min-max bersifat linear, jadi selisih dalam USD sama dengan selisih
ternormalisasi dikali (x_max − x_min). Akibatnya **RMSE (USD) = (x_max − x_min) × √MSE**.
Dengan rentang harga data latih {usd(kt['x_max'])} − {usd(kt['x_min'])} = {usd(kt['rentang'])}:
""")
    d.kode([f"RMSE = {usd(kt['rentang'])} × √{k['val_akhir']:.8f}",
            f"     = {usd(kt['rentang'])} × {math.sqrt(k['val_akhir']):.6f}",
            f"     = {usd(rmse)} USD",
            f"RMSE validasi pada tabel tuning = {usd(hasil['rmse_val'])} USD"])
    d.teks("""
Jadi **meminimalkan MSE saat pelatihan sama dengan meminimalkan RMSE dalam USD**. Ini jawaban
untuk pertanyaan "kenapa loss-nya MSE tetapi evaluasinya RMSE?".
""")

    d.teks(f"## {J[6]}")
    d.tabel(["Tahap penelitian", "Loss MSE", "Adam"],
            [["Model baru dibangun", "-", "m = 0, v = 0, k = 0"],
             [f"Setiap batch latih ({kt['iter_epoch']}× per epoch)", "dihitung, menjadi sumber gradien",
              "memperbarui semua bobot dan bias"],
             ["Akhir setiap epoch", "loss latih dan validasi dicatat (kurva loss)", "tidak ada update dari data validasi"],
             ["Pemilihan neuron dan epoch terbaik", "tidak (memakai RMSE validasi USD)", "tidak"],
             ["Prediksi data uji dan evaluasi", "tidak", "tidak (bobot sudah beku)"]])
    return d.isi()


# =========================================================================== #
# ISI BERKAS: BIAS (Tahap 7.12 dan 8.12)
# =========================================================================== #
def _bias_mengapa(d, sk, kt, hasil):
    m = sk.model
    s1 = sk.maju_satu["langkah"][0]
    d.teks("""
**(a) Tanpa bias, gerbang terkunci di σ(0) = 0.5 saat masukannya nol.** Kondisi ini sering
terjadi: h₀ = 0 di awal setiap jendela, dan normalisasi min-max membuat fitur bernilai dekat
0 ketika nilainya mendekati minimum data latih.

**(b) Bias menggeser ambang buka-tutup gerbang.** Pada σ(W·x + b), bobot W mengatur
kecuraman, sedangkan bias b mengatur posisinya (gerbang = 0.5 saat x = −b/W). Contoh satu
fitur ternormalisasi x ∈ [0, 1] dengan W = 5:
""")
    sg = lambda z: 1 / (1 + math.exp(-z))   # noqa: E731
    d.tabel(["x", "Dengan bias: σ(5x − 2.5)", "Tanpa bias: σ(5x)"],
            [[f"{x:g}", a(sg(5 * x - 2.5), 3), a(sg(5 * x), 3)] for x in (0, 0.25, 0.5, 0.75, 1)])
    d.teks("""
Tanpa bias, gerbang **tidak pernah di bawah 0.5** untuk data ternormalisasi yang positif,
sehingga model tidak bisa membuat aturan "tutup gerbang saat fitur rendah, buka saat tinggi".
""")
    b = hasil["bias"]
    if m == "LSTM":
        bf = b["f"]
        f1 = float(s1["f"][0, 0])
        d.teks(f"""
**(c) Bias menentukan sikap bawaan setiap gerbang.** Contoh paling penting adalah forget
gate. Di {label('LSTM', 'alur_sel')} terlihat bahwa pada bobot awal, forget gate neuron ke-1
sudah bernilai {a(f1, 3)} di hari pertama karena biasnya 1. Kalau gerbang hanya ditentukan
biasnya, sisa memori setelah {kt['lookback']} hari adalah f^{kt['lookback']}:
""")
        d.tabel(["Bias forget", "f = σ(b)", f"Memori tersisa setelah {kt['lookback']} hari"],
                [[lbl, a(sg(v), 3), persen(sg(v) ** kt["lookback"])]
                 for lbl, v in (("0 (tanpa bias)", 0.0), ("1 (bobot awal Keras)", 1.0),
                                (f"{a(bf)} (neuron ke-1 model terlatih)", bf), ("2", 2.0))])
        d.teks("""
Tanpa bias, memori langsung susut separuh setiap hari. Itulah alasan Keras mengisi
**b_f = 1** di awal pelatihan (*unit forget bias*). Sikap bawaan setiap gerbang pada **model
LSTM terlatih** (neuron ke-1, saat xW + hU = 0; nilai bias yang sama tampil di Tahap 7.11):
""")
        def sikap(v, rendah, tinggi):
            return "hampir netral (sekitar setengah terbuka)" if abs(sg(v) - 0.5) < 0.05 else \
                (rendah if sg(v) < 0.5 else tinggi)
        d.tabel(["Gerbang", "Bias", "Nilai bawaan", "Artinya"],
                [["input i", a(b["i"]), f"σ = {a(sg(b['i']), 3)}",
                  sikap(b["i"], "cenderung hemat menerima informasi baru", "cenderung menerima informasi baru")],
                 ["forget f", a(b["f"]), f"σ = {a(sg(b['f']), 3)}",
                  sikap(b["f"], "cenderung membuang memori", "cenderung menyimpan memori")],
                 ["kandidat c̃", a(b["c"]), f"tanh = {a(math.tanh(b['c']), 3)}",
                  "isi bawaan hampir netral" if abs(b["c"]) < 0.1 else "isi bawaan tidak netral"],
                 ["output o", a(b["o"]), f"σ = {a(sg(b['o']), 3)}",
                  sikap(b["o"], "cenderung menahan sebagian keluaran", "cenderung mengeluarkan memori")]])
    else:
        z, r = sum(b["z"]), sum(b["r"])
        d.teks("""
**(c) Bias menentukan sikap bawaan setiap gerbang.** Pada GRU, bias update gate menentukan
campuran bawaan memori lama dan baru, sedangkan bias reset gate menentukan seberapa banyak
masa lalu dipakai untuk kandidat. Semua bias GRU mulai dari 0 (sikap netral, σ = 0.5), lalu
bergeser selama pelatihan. Pada **model GRU terlatih** (neuron ke-1, saat xW + hU = 0;
nilai bias yang sama tampil di Tahap 8.11):
""")
        d.tabel(["Gerbang", "Bias masukan + bias rekuren", "Nilai bawaan", "Artinya"],
                [["update z", f"{a(b['z'][0])} + {kr(b['z'][1])} = {a(z)}", f"σ = {a(sg(z), 3)}",
                  f"cenderung mempertahankan {persen(sg(z), 0)} memori lama"],
                 ["reset r", f"{a(b['r'][0])} + {kr(b['r'][1])} = {a(r)}", f"σ = {a(sg(r), 3)}",
                  f"memakai sekitar {persen(sg(r), 0)} masa lalu untuk kandidat"],
                 ["kandidat h̃", f"b_h(in) = {a(b['h'][0])}, b_h(rec) = {a(b['h'][1])}", "-",
                  "b_h(rec) ikut dikalikan rₜ (bagian 4)"]])
    d.teks(f"""
Nilai ini hanya **titik awal**. Nilai gerbang sebenarnya berubah setiap hari mengikuti
xₜW + hₜ₋₁U; bias menentukan dari mana perubahan itu dimulai.

**(d) Bias dense menentukan tingkat dasar prediksi.** h_T selalu di rentang (−1, 1), jadi b_y
yang mengatur "tingkat dasar" prediksi dan W_y cukup menangani variasinya. Pada model {m}
terlatih, b_y = {a(hasil['b_y'])}, setara {a(hasil['b_y'])} × {usd(kt['rentang'])} ≈
**{usd(hasil['b_y'] * kt['rentang'])} USD**. Tanpa b_y, prediksi dipaksa jatuh ke harga terendah
data latih ({usd(kt['x_min'])} USD) setiap kali h_T = 0.

**(e) Bias selalu mendapat sinyal belajar.** ∂L/∂W = Σ δ·x dan ∂L/∂U = Σ δ·hₜ₋₁ bernilai nol
kalau masukannya nol (di {label(m, 'bptt')}, U tidak mendapat sumbangan dari t = 1 karena
h₀ = 0), sedangkan ∂L/∂b = Σ δ tidak dikalikan apa-apa.
""")
    n = hasil["neuron"]
    jumlah_bias = (4 * n if m == "LSTM" else 6 * n) + 1
    d.teks(f"**(f) Biayanya kecil.** Model {m} terlatih memiliki {ribu(jumlah_bias)} bias "
           f"(termasuk b_y) dari {ribu(hasil['param'])} parameter, atau {persen(jumlah_bias / hasil['param'])}.")


def _bias_cara(d, sk, kt, info, p_akhir):
    m = sk.model
    total = kt["iter_epoch"] * info["epoch"]
    T = sk.lookback
    R = sk.mundur1["rincian"]
    g = sk.mundur1["g"]
    if m == "LSTM":
        nm, kunci_delta, simbol = "b_f", "delta_f", "δa_f"
        nilai_awal = "b_f = 1 untuk semua neuron (unit forget bias), b_i = b_c = b_o = 0, dan b_y = 0."
    else:
        nm, kunci_delta, simbol = "b_h_in", "delta_h", "δa_h"
        nilai_awal = "semua bias gerbang (masukan dan rekuren) = 0, dan b_y = 0."
    per_t = [float(R[t][kunci_delta][:, 0].sum()) for t in range(T)]
    assert abs(sum(per_t) - float(g[nm][0])) < 1e-12
    namab = _nama_param(nm, (0,))
    d.teks(f"""
Bias tidak punya rumus langsung; bias **dipelajari** lewat siklus yang sama dengan bobot.
Berikut lima langkahnya pada model {m} utama, memakai angka yang sudah dihitung di
subbagian sebelumnya. Contohnya **b_y** dan **{namab}** (bias neuron ke-1).
""")
    d.kode([
        f"Langkah 1 — Nilai awal ({label(m, 'alur_sel')})",
        f"  {nilai_awal}",
        "",
        f"Langkah 2 — Dipakai di forward pass ({label(m, 'alur_sel')})",
        "  Gerbang : a = x W + h U + b",
        "  Dense   : ŷ′ = h_T W_y + b_y",
        "",
        f"Langkah 3 — Gradien ({label(m, 'bptt')})",
        "  Karena ∂a/∂b = 1, gradien bias = jumlah δ untuk semua sampel dan semua hari:",
        f"  ∂L/∂b_y = Σₙ ∂L/∂ŷ′ₙ = {e(g['b_y'][0])}",
        f"  ∂L/∂{namab} = Σₜ Σₙ {simbol}  (neuron ke-1), per hari:",
        "    " + ", ".join(f"t={t + 1}: {e(per_t[t])}" for t in range(T)),
        f"    jumlah = {e(g[nm][0])}",
        "",
        f"Langkah 4 — Update Adam ({label(m, 'adam')})",
        f"  b_y : {a(sk.p0['b_y'][0])} → {a(sk.p1['b_y'][0])} (k = 1) → {a(sk.p2['b_y'][0])} (k = 2)",
        f"  {namab}: {a(sk.p0[nm][0])} → {a(sk.p1[nm][0])} (k = 1) → {a(sk.p2[nm][0])} (k = 2)",
        "",
        f"Langkah 5 — Ulangi {ribu(total)} kali ({kt['iter_epoch']} batch × {ribu(info['epoch'])} epoch)",
        f"  b_y akhir = {a(p_akhir['b_y'][0])},  {namab} akhir = {a(p_akhir[nm][0])}",
    ])


def bias(sk, kt, hasil, info, bobot_akhir) -> str:
    m = sk.model
    p_akhir = pecah_bobot(m, bobot_akhir)
    if m == "LSTM":
        J = {1: "1. Intinya", 2: "2. Mengapa Bias Diperlukan", 3: "3. Cara Menghitung Bias Langkah demi Langkah",
             4: "4. Catatan untuk Naskah Skripsi"}
    else:
        J = {1: "1. Intinya", 2: "2. Mengapa Bias Diperlukan", 3: "3. Cara Menghitung Bias Langkah demi Langkah",
             4: "4. Dua Bias pada GRU: Asal dan Buktinya", 5: "5. Catatan untuk Naskah Skripsi"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini merangkum pembahasan **bias pada {m}**: kenapa bias perlu ada dan bagaimana
nilainya diperoleh{', termasuk kenapa GRU punya dua jenis bias' if m == 'GRU' else ''}. Semua contoh
angka diambil dari model {m} utama ({hasil['neuron']} neuron): bobot awal dan satu siklus
pelatihannya ({label(m, 'alur_sel')} sampai {label(m, 'adam')}) serta bobot akhirnya.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.teks("""
Bias itu **titik awal (intercept)** setiap gerbang dan neuron, mirip intercept *a* pada
regresi y = a + bx. Bobot hanya bisa *mengalikan* masukan. Tanpa bias, setiap kali
masukannya nol, gerbang dipaksa bernilai tetap (σ(0) = 0.5, selalu setengah terbuka;
tanh(0) = 0). Dengan bias, setiap gerbang bisa punya **posisi bawaannya sendiri**. Nilainya
tidak dihitung dengan satu rumus, tetapi **dipelajari** lewat siklus yang sama dengan bobot
(bagian 3).
""")

    d.teks(f"## {J[2]}")
    _bias_mengapa(d, sk, kt, hasil)

    d.teks(f"## {J[3]}")
    _bias_cara(d, sk, kt, info, p_akhir)

    if m == "GRU":
        g = sk.mundur1["g"]
        selisih_z = float(np.max(np.abs(p_akhir["b_z_in"] - p_akhir["b_z_rec"])))
        selisih_r = float(np.max(np.abs(p_akhir["b_r_in"] - p_akhir["b_r_rec"])))
        selisih_h = float(np.max(np.abs(p_akhir["b_h_in"] - p_akhir["b_h_rec"])))
        kembar = max(selisih_z, selisih_r) < 1e-6
        d.teks(f"## {J[4]}")
        d.teks("""
**Letaknya pada Gambar 7.** Bias tidak digambar; bias ada di dalam setiap kotak kuning (σ, σ,
tanh). Setiap kotak menerima dua garis masuk, yaitu xₜ dari bawah dan hₜ₋₁ dari kiri. Dengan
`reset_after=True`, Keras menghitung kedua garis itu terpisah, masing-masing dengan biasnya
sendiri: garis xₜ membawa **bias masukan** (xₜW + b(in)) dan garis hₜ₋₁ membawa **bias
rekuren** (hₜ₋₁U + b(rec)).
""")
        d.kode("""
Kotak σ untuk z (dan r, sama persis): kedua cabang langsung dijumlahkan
  h_(t-1) ──► h_(t-1) × U_z + b_z(rec) ──┐
                                         (+) ──► σ ──► z_t
  x_t ─────► x_t × W_z + b_z(in) ────────┘

Kotak tanh untuk kandidat: cabang h_(t-1) dikali r_t dulu
  h_(t-1) ──► h_(t-1) × U_h + b_h(rec) ──► (× r_t) ──┐
                                                     (+) ──► tanh ──► h̃_t
  x_t ─────► x_t × W_h + b_h(in) ─────────────────────┘
""")
        d.teks(f"""
**Pada z dan r, dua bias sebenarnya berlebih.** b(in) dan b(rec) langsung dijumlahkan, jadi
gradien keduanya selalu sama. Di iterasi pertama model ini, ∂L/∂b_z(in)[1] = ∂L/∂b_z(rec)[1] =
{e(g['b_z_in'][0])}. Karena nilai awalnya juga sama (0) dan gradiennya selalu kembar, keduanya
{'tetap kembar sampai akhir pelatihan (selisihnya hanya sisa pembulatan float32)' if kembar else 'hampir sama sampai akhir pelatihan'}: selisih
terbesar antara b_z(in) dan b_z(rec) di model terlatih = {selisih_z:.1e}, dan antara b_r(in) dan
b_r(rec) = {selisih_r:.1e}.

**Pada kandidat, dua bias berbeda peran.** b_h(rec) ikut dikali rₜ (kalau rₜ mendekati 0, bias ini
ikut "dimatikan" bersama memori lama), sedangkan b_h(in) selalu aktif. Gradiennya pun berbeda:
iterasi pertama memberi ∂L/∂b_h(in)[1] = {e(g['b_h_in'][0])} dan ∂L/∂b_h(rec)[1] =
{e(g['b_h_rec'][0])}. Di model terlatih, neuron ke-1 punya b_h(in) = {a(p_akhir['b_h_in'][0])} dan
b_h(rec) = {a(p_akhir['b_h_rec'][0])} (selisih terbesar di antara semua neuron {a(selisih_h)}).

**Asal-usulnya.** Persamaan asli Cho et al. (2014) tidak menuliskan bias sama sekali
(*"to make the equations uncluttered, we omit biases"*). Bentuk dua bias berasal dari
implementasi GRU pada pustaka **NVIDIA cuDNN** yang memisahkan bagian masukan dan rekuren
setiap gerbang, dan Keras memakainya sebagai bawaan (`reset_after=True`). LSTM cukup satu
bias karena semua biasnya hanya dijumlahkan sehingga selalu bisa digabung.

**Dampak ke jumlah parameter (persamaan 27).** GRU {hasil['neuron']} neuron memiliki
2 × 3 × {hasil['neuron']} = {6 * hasil['neuron']} bias gerbang. Dengan satu bias jumlahnya hanya
{3 * hasil['neuron']}, sehingga total parameter menjadi {ribu(hasil['param'] - 3 * hasil['neuron'])},
bukan {ribu(hasil['param'])}.
""")

    d.teks(f"## {J[len(J)]}")
    if m == "LSTM":
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
> b_h⁽ʳᵉᶜ⁾ ikut dikalikan dengan reset gate rₜ sehingga keduanya tidak dapat digabung.
""")
    return d.isi()


# =========================================================================== #
# ISI BERKAS: PERBANDINGAN STRUKTUR LSTM DAN GRU (Tahap 9.2)
# =========================================================================== #
def _porsi_jalan_tol(sk):
    """Porsi rata-rata sinyal ke time step T-1 yang lewat jalur langsung (semua sampel dan neuron)."""
    r = sk.mundur1["rincian"][-2] if sk.model == "LSTM" else sk.mundur1["rincian"][-1]
    if sk.model == "LSTM":
        tol, lain = np.abs(r["dc_lanjut"]), np.abs(r["jalur_h"])
    else:
        tol, lain = np.abs(r["dh_langsung"]), np.abs(r["dh_prev"] - r["dh_langsung"])
    return float(tol.sum() / (tol.sum() + lain.sum()))


def perbandingan(sk_lstm, sk_gru, kt, hasil_lstm, hasil_gru) -> str:
    J = {1: "1. Perbedaan Struktur Sel", 2: "2. Jalan Tol Gradien pada Kedua Model",
         3: "3. Jumlah Parameter pada Jumlah Neuron yang Sama", 4: "4. Satu Siklus Pelatihan Model Utama"}
    d = Dokumen()
    d.teks(f"""
Subbagian ini merangkum perbedaan struktur LSTM (Tahap 7) dan GRU (Tahap 8) yang sudah
dihitung terpisah, sebagai pelengkap Tabel 13 dan subbab 1.5.10 (Tabel 2). Semua angka
berasal dari model utama: LSTM {hasil_lstm['neuron']} neuron dan GRU {hasil_gru['neuron']} neuron.
""")
    d.daftar_isi(J)
    d.teks(CATATAN_ANGKA)

    d.teks(f"## {J[1]}")
    d.tabel(["Aspek", "LSTM", "GRU", "Penjelasan"],
            [["Gerbang", "3 (forget, input, output)", "2 (update, reset)", "GRU tidak punya output gate"],
             ["Simpan vs terima", "fₜ dan iₜ **bebas** satu sama lain", "zₜ dan (1 − zₜ) **terikat**",
              "pada GRU, menyimpan lebih banyak yang lama berarti menerima lebih sedikit yang baru"],
             ["Memori internal", "cₜ dan hₜ", "hanya hₜ", "GRU menyimpan memori langsung di hidden state"],
             ["Peran masa lalu pada kandidat", "lewat U_c", "diatur rₜ (reset gate)",
              "LSTM tidak punya padanan langsung reset gate"],
             ["Himpunan bobot", "4 (i, f, c, o)", "3 (z, r, h)", "parameter GRU lebih sedikit"],
             ["Bias per himpunan (Keras)", "n_u", "2n_u", "GRU memakai bias masukan dan bias rekuren"]])

    d.teks(f"## {J[2]}")
    T = kt["lookback"]
    d.teks(f"""
Kedua model punya jalur sederhana yang membuat gradien tidak cepat mengecil saat dikirim
mundur. Pada iterasi pertama pelatihan (rata-rata seluruh {sk_lstm.batch} sampel × neuron):

- **LSTM**: jalur cell state (Gambar 2), sinyal hanya dikali fₜ. Sebanyak
  **{persen(_porsi_jalan_tol(sk_lstm))}** besar sinyal kesalahan ke c{sb(T - 1)} lewat jalur ini
  ({tautan('LSTM', 'bptt')}).
- **GRU**: jalur langsung zₜ ⊙ hₜ₋₁ (Gambar 7), sinyal hanya dikali zₜ. Sebanyak
  **{persen(_porsi_jalan_tol(sk_gru))}** besar sinyal kesalahan ke h{sb(T - 1)} lewat jalur ini
  ({tautan('GRU', 'bptt')}).
""")

    d.teks(f"## {J[3]}")
    nf = kt["n_fitur"]
    for hasil in (hasil_lstm, hasil_gru):
        assert jumlah_parameter(hasil["model"], hasil["neuron"], nf) == hasil["param"]
    d.teks(f"Jumlah parameter total (lapisan rekuren + dense) dengan {nf} fitur, persamaan 22 dan 27:")
    d.tabel(["Neuron n_u", "LSTM", "GRU", "GRU / LSTM"],
            [[n,
              ribu(jumlah_parameter("LSTM", n, nf)) + (" ← terbaik" if n == hasil_lstm["neuron"] else ""),
              ribu(jumlah_parameter("GRU", n, nf)) + (" ← terbaik" if n == hasil_gru["neuron"] else ""),
              persen(jumlah_parameter("GRU", n, nf) / jumlah_parameter("LSTM", n, nf))]
             for n in kt["neuron_grid"]])
    d.teks(f"""
Pada jumlah neuron yang sama, GRU selalu butuh parameter lebih sedikit. Model terbaik
penelitian adalah LSTM {hasil_lstm['neuron']} neuron ({ribu(hasil_lstm['param'])} parameter) dan GRU
{hasil_gru['neuron']} neuron ({ribu(hasil_gru['param'])} parameter).
""")

    d.teks(f"## {J[4]}")
    kl, kg = hasil_lstm["kurva"], hasil_gru["kurva"]
    d.tabel(["Besaran", f"LSTM ({hasil_lstm['neuron']} neuron)", f"GRU ({hasil_gru['neuron']} neuron)"],
            [["Jumlah parameter", ribu(sk_lstm.n_param), ribu(sk_gru.n_param)],
             ["Loss batch 1 dengan bobot awal", e(sk_lstm.maju1["L"]), e(sk_gru.maju1["L"])],
             ["Loss batch 1 setelah 1 update", e(sk_lstm.L1_sesudah), e(sk_gru.L1_sesudah)],
             ["Jumlah epoch", ribu(hasil_lstm["epoch"]), ribu(hasil_gru["epoch"])],
             ["Total update Adam", ribu(kt["iter_epoch"] * hasil_lstm["epoch"]), ribu(kt["iter_epoch"] * hasil_gru["epoch"])],
             ["Loss latih epoch terakhir", f"{kl['latih_akhir']:.8f}", f"{kg['latih_akhir']:.8f}"],
             ["Loss validasi minimum", f"{kl['val_min']:.8f}", f"{kg['val_min']:.8f}"]])
    d.teks("""
Urutan siklus pelatihan (forward → loss → BPTT → Adam) dan pengaturan Adam pada kedua model
sama persis; yang berbeda hanya rumus di dalam sel. Loss pada awal pelatihan bergantung pada
undian bobot awal, jadi **tidak bisa dipakai untuk menyimpulkan** model mana yang lebih
baik. Perbandingan yang sah adalah hasil evaluasi data uji dan uji Diebold-Mariano pada
Tahap 11 dan 12.
""")
    return d.isi()
