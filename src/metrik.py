"""
Metrik evaluasi peramalan dan uji signifikansi Diebold-Mariano.

Seluruh metrik diimplementasikan langsung dengan NumPy/SciPy (tanpa memakai
fungsi siap pakai scikit-learn) agar setiap langkah perhitungan dapat
ditelusuri dan dicocokkan dengan perhitungan manual di skripsi.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Metrik dasar
# --------------------------------------------------------------------------- #
def rmse(y_aktual, y_prediksi) -> float:
    r"""
    Root Mean Squared Error.

    .. math:: RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(y_t - \hat{y}_t)^2}

    Urutan perhitungan: kuadratkan -> rata-ratakan -> akarkan.
    """
    y_aktual = np.asarray(y_aktual, dtype=np.float64).ravel()
    y_prediksi = np.asarray(y_prediksi, dtype=np.float64).ravel()
    galat = y_aktual - y_prediksi
    galat_kuadrat = galat ** 2
    rerata_galat_kuadrat = galat_kuadrat.mean()
    return float(np.sqrt(rerata_galat_kuadrat))


def mae(y_aktual, y_prediksi) -> float:
    r"""
    Mean Absolute Error.

    .. math:: MAE = \frac{1}{n}\sum_{t=1}^{n}\left|y_t - \hat{y}_t\right|
    """
    y_aktual = np.asarray(y_aktual, dtype=np.float64).ravel()
    y_prediksi = np.asarray(y_prediksi, dtype=np.float64).ravel()
    return float(np.abs(y_aktual - y_prediksi).mean())


def mape(y_aktual, y_prediksi) -> float:
    r"""
    Mean Absolute Percentage Error (dalam persen).

    .. math:: MAPE = \frac{100\%}{n}\sum_{t=1}^{n}
              \left|\frac{y_t - \hat{y}_t}{y_t}\right|
    """
    y_aktual = np.asarray(y_aktual, dtype=np.float64).ravel()
    y_prediksi = np.asarray(y_prediksi, dtype=np.float64).ravel()
    return float(np.abs((y_aktual - y_prediksi) / y_aktual).mean() * 100.0)


def akurasi_arah(y_aktual, y_prediksi, y_sebelumnya) -> float:
    r"""
    Akurasi arah (*directional accuracy*), yaitu persentase hari ketika arah
    perubahan harga (naik/turun) tertebak dengan benar.

    .. math:: DA = \frac{100\%}{n}\sum_{t=1}^{n}
              \mathbb{1}\left[\operatorname{sign}(y_t - y_{t-1}) =
              \operatorname{sign}(\hat{y}_t - y_{t-1})\right]

    ``y_sebelumnya`` adalah harga AKTUAL satu hari sebelum tanggal prediksi,
    sehingga arah aktual dan arah prediksi diukur dari titik acuan yang sama.
    """
    y_aktual = np.asarray(y_aktual, dtype=np.float64).ravel()
    y_prediksi = np.asarray(y_prediksi, dtype=np.float64).ravel()
    y_sebelumnya = np.asarray(y_sebelumnya, dtype=np.float64).ravel()
    arah_aktual = np.sign(y_aktual - y_sebelumnya)
    arah_prediksi = np.sign(y_prediksi - y_sebelumnya)
    return float((arah_aktual == arah_prediksi).mean() * 100.0)


def kategori_mape(nilai_mape: float) -> str:
    """Kategori kemampuan peramalan berdasarkan MAPE (Lewis, 1982)."""
    if nilai_mape < 10:
        return "Sangat baik (<10%)"
    if nilai_mape < 20:
        return "Baik (10-20%)"
    if nilai_mape < 50:
        return "Cukup (20-50%)"
    return "Buruk (>50%)"


def semua_metrik(y_aktual, y_prediksi, y_sebelumnya=None) -> dict:
    """Kumpulkan RMSE, MAE, MAPE, kategori MAPE, dan akurasi arah sekaligus."""
    hasil = {
        "RMSE (USD)": rmse(y_aktual, y_prediksi),
        "MAE (USD)": mae(y_aktual, y_prediksi),
        "MAPE (%)": mape(y_aktual, y_prediksi),
    }
    hasil["Kategori MAPE"] = kategori_mape(hasil["MAPE (%)"])
    if y_sebelumnya is not None:
        hasil["Akurasi Arah (%)"] = akurasi_arah(y_aktual, y_prediksi, y_sebelumnya)
    return hasil


# --------------------------------------------------------------------------- #
# Uji Diebold-Mariano dengan koreksi Harvey-Leybourne-Newbold
# --------------------------------------------------------------------------- #
def diebold_mariano(galat_1, galat_2, h: int = 1, daya: int = 2) -> dict:
    r"""
    Uji Diebold-Mariano (1995) dengan koreksi sampel kecil
    Harvey-Leybourne-Newbold (1997).

    Hipotesis:
        H0 : E[d_t] = 0  (akurasi kedua model sama)
        H1 : E[d_t] != 0 (akurasi kedua model berbeda)

    Langkah:
        1. d_t     = |e_{1,t}|^p - |e_{2,t}|^p     (di sini p = 2, loss kuadrat)
        2. d_bar   = (1/T) * sum d_t
        3. gamma_k = (1/T) * sum (d_t - d_bar)(d_{t-k} - d_bar)
        4. V(d_bar) = [gamma_0 + 2*sum_{k=1}^{h-1} gamma_k] / T
        5. DM      = d_bar / sqrt(V(d_bar))
        6. Koreksi HLN:
           DM* = DM * sqrt( (T + 1 - 2h + h(h-1)/T) / T )
        7. p-value dua sisi dari distribusi t dengan derajat bebas T - 1.

    Nilai DM* negatif berarti model 1 memiliki loss lebih kecil (lebih akurat).
    """
    e1 = np.asarray(galat_1, dtype=np.float64).ravel()
    e2 = np.asarray(galat_2, dtype=np.float64).ravel()
    if e1.shape != e2.shape:
        raise ValueError("Panjang galat kedua model harus sama.")

    d = np.abs(e1) ** daya - np.abs(e2) ** daya
    T = d.size
    d_bar = d.mean()

    def gamma(k: int) -> float:
        if k == 0:
            return float(((d - d_bar) ** 2).sum() / T)
        return float(((d[k:] - d_bar) * (d[:-k] - d_bar)).sum() / T)

    gamma_0 = gamma(0)
    jumlah_autokovarians = sum(gamma(k) for k in range(1, h))
    varians_d_bar = (gamma_0 + 2.0 * jumlah_autokovarians) / T

    dm = d_bar / np.sqrt(varians_d_bar)
    faktor_hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * faktor_hln
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))

    return {
        "d": d,
        "T": int(T),
        "d_bar": float(d_bar),
        "gamma_0": float(gamma_0),
        "varians_d_bar": float(varians_d_bar),
        "DM": float(dm),
        "faktor_HLN": float(faktor_hln),
        "DM_HLN": float(dm_hln),
        "df": int(T - 1),
        "p_value": float(p_value),
        "h": int(h),
        "daya": int(daya),
    }
