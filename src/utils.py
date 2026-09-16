"""
Fungsi pendukung umum: pencetakan header tahap, format angka, penyimpanan
tabel/gambar ke folder ``outputs/``, serta penulisan berkas perhitungan manual.

Seluruh fungsi di modul ini hanya mengurus TAMPILAN dan PENYIMPANAN, tidak
mengubah nilai numerik apa pun, sehingga aman dipakai di semua tahap.
"""

from __future__ import annotations

import os
import textwrap

import pandas as pd

# --------------------------------------------------------------------------- #
# Lokasi folder keluaran
# --------------------------------------------------------------------------- #
DIR_PROYEK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATA = os.path.join(DIR_PROYEK, "data")
DIR_OUT = os.path.join(DIR_PROYEK, "outputs")
DIR_TABEL = os.path.join(DIR_OUT, "tabel")
DIR_GAMBAR = os.path.join(DIR_OUT, "gambar")
DIR_MANUAL = os.path.join(DIR_OUT, "perhitungan_manual")

for _d in (DIR_DATA, DIR_TABEL, DIR_GAMBAR, DIR_MANUAL):
    os.makedirs(_d, exist_ok=True)

LEBAR = 70  # lebar garis pemisah pada header tahap


# --------------------------------------------------------------------------- #
# Header, sub-judul, dan ringkasan tahap
# --------------------------------------------------------------------------- #
def cetak_header(nomor_tahap, judul: str) -> None:
    """Cetak header tahap, contoh::

        ======================================================================
        TAHAP 5 - NORMALISASI DATA (MIN-MAX SCALING)
        ======================================================================
    """
    print("=" * LEBAR)
    print(f"TAHAP {nomor_tahap} - {judul.upper()}")
    print("=" * LEBAR)


def cetak_sub(judul: str) -> None:
    """Cetak sub-judul di dalam satu tahap."""
    print()
    print("-" * LEBAR)
    print(judul)
    print("-" * LEBAR)


def ringkasan_tahap(teks: str) -> None:
    """Cetak blok 'Ringkasan Tahap' berisi interpretasi 2-4 kalimat."""
    print()
    print("=" * LEBAR)
    print("RINGKASAN TAHAP")
    print("=" * LEBAR)
    for paragraf in textwrap.dedent(teks).strip().split("\n\n"):
        satu_baris = " ".join(paragraf.split())
        print(textwrap.fill(satu_baris, width=LEBAR))
        print()


def cetak_shape(nama: str, obj) -> None:
    """Cetak bentuk (shape) sebuah objek setiap kali bentuk data berubah."""
    bentuk = getattr(obj, "shape", None)
    print(f"  shape {nama:<28} = {bentuk}")


# --------------------------------------------------------------------------- #
# Format angka
# --------------------------------------------------------------------------- #
def fmt_usd(x, desimal: int = 2) -> str:
    """Format nilai USD: pemisah ribuan, 2 desimal. Contoh: 61,234.56"""
    try:
        return f"{float(x):,.{desimal}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_norm(x, desimal: int = 6) -> str:
    """Format nilai ternormalisasi: 6 desimal. Contoh: 0.123457"""
    try:
        return f"{float(x):,.{desimal}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_int(x) -> str:
    """Format bilangan bulat dengan pemisah ribuan. Contoh: 1,127"""
    try:
        return f"{int(round(float(x))):,d}"
    except (TypeError, ValueError):
        return str(x)


def aktifkan_format_pandas(desimal: int = 4) -> None:
    """Atur tampilan pandas: pemisah ribuan dan jumlah desimal seragam."""
    pd.set_option("display.float_format", lambda v: f"{v:,.{desimal}f}")
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)


# --------------------------------------------------------------------------- #
# Penyimpanan tabel dan gambar (bernomor + berjudul)
# --------------------------------------------------------------------------- #
def _slug(teks: str) -> str:
    aman = "".join(c.lower() if c.isalnum() else "_" for c in teks)
    while "__" in aman:
        aman = aman.replace("__", "_")
    return aman.strip("_")[:70]


def simpan_tabel(df: pd.DataFrame, nomor, judul: str, indeks: bool = False) -> str:
    """
    Simpan DataFrame ke ``outputs/tabel/`` dalam format .csv dan .xlsx,
    lalu cetak judul tabel. Mengembalikan path dasar (tanpa ekstensi).
    """
    nama = f"tabel_{nomor:02d}_{_slug(judul)}" if isinstance(nomor, int) else f"tabel_{nomor}_{_slug(judul)}"
    dasar = os.path.join(DIR_TABEL, nama)
    df.to_csv(dasar + ".csv", index=indeks)
    try:
        df.to_excel(dasar + ".xlsx", index=indeks)
    except Exception as e:  # pragma: no cover - hanya jika openpyxl tak ada
        print(f"  [catatan] gagal menulis .xlsx ({e}); file .csv tetap tersimpan.")
    print(f"  [tersimpan] Tabel {nomor}. {judul}")
    print(f"              -> {os.path.relpath(dasar, DIR_PROYEK)}.csv / .xlsx")
    return dasar


def simpan_gambar(fig, nomor, judul: str, dpi: int = 300) -> str:
    """Simpan figure matplotlib ke ``outputs/gambar/`` dengan resolusi 300 dpi."""
    nama = f"gambar_{nomor:02d}_{_slug(judul)}" if isinstance(nomor, int) else f"gambar_{nomor}_{_slug(judul)}"
    path = os.path.join(DIR_GAMBAR, nama + ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [tersimpan] Gambar {nomor}. {judul}")
    print(f"              -> {os.path.relpath(path, DIR_PROYEK)} ({dpi} dpi)")
    return path


def tulis_manual(tahap, judul: str, isi: str) -> str:
    """
    Tulis dokumentasi perhitungan manual satu tahap ke
    ``outputs/perhitungan_manual/tahap_X.md`` (siap disalin ke Bab III).
    """
    path = os.path.join(DIR_MANUAL, f"tahap_{tahap}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Perhitungan Manual - Tahap {tahap}: {judul}\n\n")
        f.write(textwrap.dedent(isi).strip() + "\n")
    print(f"  [tersimpan] Perhitungan manual Tahap {tahap} -> "
          f"{os.path.relpath(path, DIR_PROYEK)}")
    return path


def cetak_dan_kumpulkan(baris: list, teks: str) -> None:
    """Cetak ``teks`` ke layar sekaligus menyimpannya ke daftar ``baris``."""
    print(teks)
    baris.append(teks)


def blok_kode(teks: str) -> str:
    """Bungkus teks dalam blok kode markdown (untuk berkas perhitungan manual)."""
    return "```\n" + textwrap.dedent(teks).strip("\n") + "\n```"
