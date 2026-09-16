"""
Pengumpulan data harga Bitcoin dan variabel informasi blockchain.

Sumber data: Blockchain.com Charts API
    https://api.blockchain.info/charts/<nama-chart>?timespan=...&start=...
    &format=json&sampled=false

Urutan sumber data yang dipakai (lihat ``muat_atau_unduh``):
    1. Jika ``data/dataset_bitcoin.csv`` sudah ada -> pakai berkas tersebut.
    2. Jika belum ada -> unduh dari API, gabungkan per tanggal (UTC),
       lalu simpan ke ``data/dataset_bitcoin.csv``.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

URL_DASAR = "https://api.blockchain.info/charts"

# --------------------------------------------------------------------------- #
# Definisi operasional variabel penelitian
# --------------------------------------------------------------------------- #
# Urutan daftar ini menentukan urutan kolom dataset: Close Price selalu
# menjadi kolom pertama karena berperan ganda sebagai Y (target) dan X1 (fitur).
DEFINISI_VARIABEL = [
    {
        "kode": "Y / X1",
        "kolom": "close_price",
        "chart": "market-price",
        "nama": "Close Price (USD)",
        "satuan": "USD",
        "definisi": "Harga penutupan harian Bitcoin rata-rata pasar bursa utama "
                    "dalam dolar AS; menjadi variabel target (Y) sekaligus fitur (X1).",
    },
    {
        "kode": "X2",
        "kolom": "miners_revenue",
        "chart": "miners-revenue",
        "nama": "Miners Revenue (USD)",
        "satuan": "USD",
        "definisi": "Total pendapatan penambang per hari, yaitu nilai hadiah blok "
                    "(block reward) ditambah biaya transaksi, dinilai dalam USD.",
    },
    {
        "kode": "X3",
        "kolom": "difficulty",
        "chart": "difficulty",
        "nama": "Difficulty",
        "satuan": "tanpa satuan",
        "definisi": "Ukuran relatif tingkat kesulitan menemukan blok baru. Nilai "
                    "disesuaikan jaringan setiap 2.016 blok (sekitar dua minggu), "
                    "sehingga bersifat fungsi tangga (step function) pada data harian.",
    },
    {
        "kode": "X4",
        "kolom": "hash_rate",
        "chart": "hash-rate",
        "nama": "Hash Rate",
        "satuan": "TH/s",
        "definisi": "Estimasi jumlah tera-hash per detik yang dikerjakan jaringan "
                    "penambang Bitcoin; mencerminkan kapasitas komputasi jaringan.",
    },
    {
        "kode": "X5",
        "kolom": "median_confirmation_time",
        "chart": "median-confirmation-time",
        "nama": "Median Confirmation Time",
        "satuan": "menit",
        "definisi": "Waktu median yang dibutuhkan sebuah transaksi (dengan biaya) "
                    "untuk masuk ke blok yang telah ditambang.",
    },
    {
        "kode": "X6",
        "kolom": "avg_block_size",
        "chart": "avg-block-size",
        "nama": "Average Block Size",
        "satuan": "MB",
        "definisi": "Ukuran rata-rata blok yang ditambang per hari dalam megabyte.",
    },
    {
        "kode": "X7",
        "kolom": "n_unique_addresses",
        "chart": "n-unique-addresses",
        "nama": "Total Unique Addresses",
        "satuan": "alamat",
        "definisi": "Jumlah alamat Bitcoin unik yang digunakan pada hari tersebut; "
                    "menjadi proksi tingkat aktivitas pengguna jaringan.",
    },
    {
        "kode": "X8",
        "kolom": "n_transactions_per_block",
        "chart": "n-transactions-per-block",
        "nama": "Transaction per Block",
        "satuan": "transaksi/blok",
        "definisi": "Jumlah rata-rata transaksi yang tercatat dalam satu blok.",
    },
    {
        "kode": "X9",
        "kolom": "n_transactions",
        "chart": "n-transactions",
        "nama": "Confirmed Transaction",
        "satuan": "transaksi",
        "definisi": "Jumlah transaksi terkonfirmasi yang tercatat di blockchain "
                    "Bitcoin per hari.",
    },
    {
        "kode": "X10",
        "kolom": "cost_per_transaction_percent",
        "chart": "cost-per-transaction-percent",
        "nama": "Cost % per Transaction",
        "satuan": "%",
        "definisi": "Rasio total pendapatan penambang terhadap total nilai "
                    "transaksi (biaya jaringan sebagai persentase volume transaksi).",
    },
]

KOLOM_TARGET = "close_price"
KOLOM_FITUR = [v["kolom"] for v in DEFINISI_VARIABEL]


def tabel_definisi_variabel() -> pd.DataFrame:
    """Tabel definisi operasional variabel (kode, nama, satuan, definisi, sumber)."""
    return pd.DataFrame([
        {
            "Kode": v["kode"],
            "Variabel": v["nama"],
            "Nama Kolom": v["kolom"],
            "Satuan": v["satuan"],
            "Definisi Operasional": v["definisi"],
            "Sumber": f"Blockchain.com Charts API - chart '{v['chart']}'",
        }
        for v in DEFINISI_VARIABEL
    ])


# --------------------------------------------------------------------------- #
# Pengunduhan satu chart
# --------------------------------------------------------------------------- #
def ambil_chart(nama_chart: str, tanggal_mulai: str, tanggal_akhir: str,
                jeda: float = 0.4, percobaan_maks: int = 4) -> pd.DataFrame:
    """
    Unduh satu chart Blockchain.com dan kembalikan DataFrame kolom
    ``['date', <nama_chart>]`` berisi data harian pada rentang yang diminta.

    Tanggal dikonversi dari UNIX timestamp ke tanggal UTC.
    """
    mulai = datetime.strptime(tanggal_mulai, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    akhir = datetime.strptime(tanggal_akhir, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    rentang_hari = (akhir - mulai).days + 1

    url = (f"{URL_DASAR}/{nama_chart}"
           f"?timespan={rentang_hari}days"
           f"&start={tanggal_mulai}"
           f"&format=json&sampled=false")

    galat_terakhir = None
    for percobaan in range(percobaan_maks):
        try:
            with urllib.request.urlopen(url, timeout=90) as respon:
                muatan = json.loads(respon.read().decode("utf-8"))
            break
        except Exception as e:                                 # jaringan gagal
            galat_terakhir = e
            if percobaan == percobaan_maks - 1:
                raise RuntimeError(
                    f"Gagal mengunduh chart '{nama_chart}' setelah "
                    f"{percobaan_maks} percobaan: {galat_terakhir}") from e
            time.sleep(2 ** (percobaan + 1))                   # 2s, 4s, 8s

    nilai = muatan.get("values", [])
    if not nilai:
        raise RuntimeError(f"Chart '{nama_chart}' tidak mengembalikan data.")

    df = pd.DataFrame(nilai)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df[["date", "y"]].rename(columns={"y": nama_chart})
    df = df[(df["date"] >= mulai.replace(tzinfo=None)) &
            (df["date"] <= akhir.replace(tzinfo=None))]
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    time.sleep(jeda)   # jeda sopan agar tidak membebani server API
    return df


def periksa_ketersediaan(tanggal_mulai: str, tanggal_akhir: str) -> pd.DataFrame:
    """
    Verifikasi ketersediaan dan frekuensi kesepuluh chart pada rentang tanggal.

    Mengembalikan tabel: kode, chart, status, tanggal awal/akhir, jumlah
    observasi, interval dominan (hari), dan jumlah nilai berulang berturut-turut
    (indikasi data bukan-harian / fungsi tangga).
    """
    baris = []
    for v in DEFINISI_VARIABEL:
        try:
            df = ambil_chart(v["chart"], tanggal_mulai, tanggal_akhir)
            selisih = df["date"].diff().dt.days.dropna()
            interval_dominan = selisih.mode().iloc[0] if len(selisih) else float("nan")
            berulang = int((df[v["chart"]].diff() == 0).sum())
            baris.append({
                "Kode": v["kode"], "Chart": v["chart"], "Status": "TERSEDIA",
                "Tanggal Awal": df["date"].min().date(),
                "Tanggal Akhir": df["date"].max().date(),
                "Jumlah Observasi": len(df),
                "Interval Dominan (hari)": interval_dominan,
                "Nilai Berulang Berturut": berulang,
            })
        except Exception as e:
            baris.append({
                "Kode": v["kode"], "Chart": v["chart"],
                "Status": f"GAGAL: {str(e)[:50]}",
                "Tanggal Awal": None, "Tanggal Akhir": None,
                "Jumlah Observasi": 0, "Interval Dominan (hari)": None,
                "Nilai Berulang Berturut": None,
            })
    return pd.DataFrame(baris)


# --------------------------------------------------------------------------- #
# Penggabungan seluruh chart
# --------------------------------------------------------------------------- #
def unduh_dataset(tanggal_mulai: str, tanggal_akhir: str, verbose: bool = True):
    """
    Unduh kesepuluh chart, gabungkan berdasarkan tanggal (UTC) memakai
    ``outer join`` pada kalender harian penuh, lalu kembalikan
    ``(dataset, tabel_jumlah_observasi)``.

    Penggabungan memakai kalender harian penuh supaya tanggal yang benar-benar
    tidak tersedia di API tetap muncul sebagai baris kosong (NaN) dan dapat
    dideteksi pada Tahap 3 (pembersihan data).
    """
    kalender = pd.DataFrame({
        "date": pd.date_range(tanggal_mulai, tanggal_akhir, freq="D")
    })
    gabungan = kalender.copy()
    catatan = []

    for v in DEFINISI_VARIABEL:
        df = ambil_chart(v["chart"], tanggal_mulai, tanggal_akhir)
        df = df.rename(columns={v["chart"]: v["kolom"]})
        gabungan = gabungan.merge(df, on="date", how="left")
        catatan.append({
            "Kode": v["kode"], "Variabel": v["nama"], "Chart": v["chart"],
            "Jumlah Observasi API": len(df),
            "Tanggal Awal": df["date"].min().date(),
            "Tanggal Akhir": df["date"].max().date(),
            "Tanggal Hilang vs Kalender": len(kalender) - len(df),
        })
        if verbose:
            print(f"  - {v['kode']:<6} {v['chart']:<30} "
                  f"{len(df):>5} observasi  "
                  f"({df['date'].min().date()} s.d. {df['date'].max().date()})")

    return gabungan, pd.DataFrame(catatan)


def muat_atau_unduh(path_csv: str, tanggal_mulai: str, tanggal_akhir: str,
                    verbose: bool = True):
    """
    Muat dataset dari ``path_csv`` bila ada; jika tidak, unduh dari API lalu
    simpan ke ``path_csv``.

    Mengembalikan ``(dataset, sumber, tabel_jumlah_observasi)`` dengan
    ``sumber`` berisi 'berkas lokal' atau 'Blockchain.com Charts API'.
    """
    import os

    if os.path.exists(path_csv):
        if verbose:
            print(f"  Berkas '{os.path.basename(path_csv)}' ditemukan -> memuat dari berkas lokal.")
        df = pd.read_csv(path_csv, parse_dates=["date"])
        kolom_hilang = [k for k in KOLOM_FITUR if k not in df.columns]
        if kolom_hilang:
            raise ValueError(
                f"Berkas {path_csv} tidak memiliki kolom wajib: {kolom_hilang}")
        df = df[["date"] + KOLOM_FITUR].sort_values("date").reset_index(drop=True)
        ringkas = pd.DataFrame([{
            "Kode": v["kode"], "Variabel": v["nama"], "Chart": v["chart"],
            "Jumlah Observasi API": int(df[v["kolom"]].notna().sum()),
            "Tanggal Awal": df.loc[df[v["kolom"]].notna(), "date"].min().date(),
            "Tanggal Akhir": df.loc[df[v["kolom"]].notna(), "date"].max().date(),
            "Tanggal Hilang vs Kalender": int(df[v["kolom"]].isna().sum()),
        } for v in DEFINISI_VARIABEL])
        return df, "berkas lokal", ringkas

    if verbose:
        print(f"  Berkas '{os.path.basename(path_csv)}' belum ada -> mengunduh dari API.")
    df, ringkas = unduh_dataset(tanggal_mulai, tanggal_akhir, verbose=verbose)
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df.to_csv(path_csv, index=False)
    if verbose:
        print(f"  Dataset gabungan disimpan ke: {path_csv}")
    return df, "Blockchain.com Charts API", ringkas
