"""
Fungsi pendukung umum: pencetakan header tahap, format angka, penyimpanan
tabel/gambar ke folder ``outputs/``, serta penulisan berkas perhitungan manual.

Seluruh fungsi di modul ini hanya mengurus TAMPILAN dan PENYIMPANAN, tidak
mengubah nilai numerik apa pun, sehingga aman dipakai di semua tahap.
"""

from __future__ import annotations

import os
import re
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


def tulis_manual(nomor: str, slug: str, judul: str, isi: str) -> str:
    """
    Tulis dokumentasi perhitungan manual satu subbagian notebook ke
    ``outputs/perhitungan_manual/tahap_TT_SS_slug.md`` (siap disalin ke Bab III).

    ``nomor`` berbentuk ``"07_08"``: Tahap 7, subbagian 7.8 di notebook. Angka
    ber-nol membuat berkas terurut dari tahap pertama sampai terakhir.
    """
    tahap, sub = (int(v) for v in nomor.split("_"))
    path = os.path.join(DIR_MANUAL, f"tahap_{nomor}_{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Perhitungan Manual - Tahap {tahap}.{sub}: {judul}\n\n")
        f.write(textwrap.dedent(isi).strip() + "\n")
    print(f"  [tersimpan] Perhitungan manual Tahap {tahap}.{sub} -> "
          f"{os.path.relpath(path, DIR_PROYEK)}")
    return path


def tampilkan_markdown(isi: str) -> None:
    """
    Tampilkan teks Markdown sebagai keluaran yang terformat di Jupyter; bila
    dijalankan sebagai skrip biasa, teksnya dicetak apa adanya. Tautan relatif
    disesuaikan karena notebook berada di folder ``notebooks/``.
    """
    teks = (isi.replace("](tahap_", "](../outputs/perhitungan_manual/tahap_")
               .replace("](../gambar/", "](../outputs/gambar/"))
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            display(Markdown(teks))
            return
    except ImportError:  # pragma: no cover - IPython tidak terpasang
        pass
    print(isi)


NAMA_TAHAP = {
    0: "Persiapan lingkungan", 1: "Pengumpulan data", 2: "Eksplorasi data",
    3: "Pembersihan data", 4: "Pembagian data (sebelum normalisasi)", 5: "Normalisasi Min-Max",
    6: "Pembentukan sliding window", 7: "Model LSTM", 8: "Model GRU",
    9: "Perbandingan arsitektur dan kestabilan", 10: "Denormalisasi", 11: "Evaluasi model",
    12: "Uji signifikansi Diebold-Mariano", 13: "Visualisasi", 14: "Kesimpulan otomatis",
}
_PENANDA_NAVIGASI = "<!-- navigasi-perhitungan-manual -->"


def tulis_indeks_manual() -> str:
    """
    Susun ``outputs/perhitungan_manual/README.md`` (daftar isi berurutan dari
    tahap pertama sampai terakhir) dan tambahkan navigasi sebelumnya/berikutnya
    di akhir setiap berkas perhitungan manual.
    """
    pola = re.compile(r"^tahap_\d{2}_\d{2}_.+\.md$")
    berkas = sorted(f for f in os.listdir(DIR_MANUAL) if pola.match(f))
    entri = []
    for nama in berkas:
        with open(os.path.join(DIR_MANUAL, nama), encoding="utf-8") as f:
            judul = f.readline().strip().removeprefix("# Perhitungan Manual - ")
        label, _, isi_judul = judul.partition(": ")
        entri.append((nama, int(nama.split("_")[1]), label, isi_judul))

    baris = [
        "# Daftar Perhitungan Manual",
        "",
        "Berkas di folder ini ditulis otomatis oleh notebook `notebooks/btc_lstm_vs_gru.ipynb`.",
        "Nama berkas `tahap_TT_SS_nama.md` berarti **Tahap TT, subbagian TT.SS** di notebook,",
        "sehingga urutan berkas sama dengan urutan pengerjaan dari tahap pertama sampai",
        "terakhir. Setiap berkas diakhiri tautan ke berkas sebelumnya dan berikutnya.",
        "",
    ]
    ada = {t for _, t, _, _ in entri}
    for tahap in sorted(NAMA_TAHAP):
        if tahap not in ada:
            continue
        baris += [f"## Tahap {tahap} — {NAMA_TAHAP[tahap]}", ""]
        baris += [f"- [{label}: {isi_judul}]({nama})" for nama, t, label, isi_judul in entri if t == tahap]
        baris.append("")
    tanpa = [f"{t} ({NAMA_TAHAP[t]})" for t in sorted(NAMA_TAHAP) if t not in ada]
    baris += ["Tahap tanpa perhitungan manual: " + ", ".join(tanpa) + ".", ""]
    path_indeks = os.path.join(DIR_MANUAL, "README.md")
    with open(path_indeks, "w", encoding="utf-8") as f:
        f.write("\n".join(baris))

    for i, (nama, _, label, isi_judul) in enumerate(entri):
        path = os.path.join(DIR_MANUAL, nama)
        with open(path, encoding="utf-8") as f:
            isi = f.read().split(_PENANDA_NAVIGASI)[0].rstrip()
        nav = [_PENANDA_NAVIGASI, "", "---", ""]
        if i > 0:
            nav.append(f"← Sebelumnya: [{entri[i - 1][2]}: {entri[i - 1][3]}]({entri[i - 1][0]})")
            nav.append("")
        nav.append("[Daftar isi perhitungan manual](README.md)")
        if i + 1 < len(entri):
            nav.append("")
            nav.append(f"→ Berikutnya: [{entri[i + 1][2]}: {entri[i + 1][3]}]({entri[i + 1][0]})")
        with open(path, "w", encoding="utf-8") as f:
            f.write(isi + "\n\n" + "\n".join(nav) + "\n")
    print(f"  [tersimpan] Daftar isi {len(entri)} berkas perhitungan manual -> "
          f"{os.path.relpath(path_indeks, DIR_PROYEK)}")
    return path_indeks


def cetak_dan_kumpulkan(baris: list, teks: str) -> None:
    """Cetak ``teks`` ke layar sekaligus menyimpannya ke daftar ``baris``."""
    print(teks)
    baris.append(teks)


def blok_kode(teks: str) -> str:
    """Bungkus teks dalam blok kode markdown (untuk berkas perhitungan manual)."""
    return "```\n" + textwrap.dedent(teks).strip("\n") + "\n```"
