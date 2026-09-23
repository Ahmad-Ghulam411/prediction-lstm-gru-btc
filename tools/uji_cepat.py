# -*- coding: utf-8 -*-
"""
Uji cepat (smoke test): menjalankan SELURUH kode notebook dengan grid kecil
supaya galat ketik/logika terdeteksi sebelum eksekusi penuh yang memakan waktu.

Jalankan dari akar proyek:  python tools/uji_cepat.py
"""
import os
import sys

import nbformat as nbf

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(AKAR, "notebooks", "btc_lstm_vs_gru.ipynb")

PATCH = """
CONFIG["NEURON_GRID"] = [4, 6]
CONFIG["EPOCH_GRID"] = [2, 3]
CONFIG["SEED_ULANGAN"] = [42, 7]
print("\\n  [UJI CEPAT] CONFIG dikecilkan:", CONFIG["NEURON_GRID"],
      CONFIG["EPOCH_GRID"], CONFIG["SEED_ULANGAN"])
"""

nb = nbf.read(NOTEBOOK, as_version=4)
potongan = []
for sel in nb.cells:
    if sel.cell_type != "code":
        continue
    potongan.append(sel.source)
    if "CONFIG = {" in sel.source:
        potongan.append(PATCH)

skrip = "\n\n# ---- SEL BERIKUTNYA ----\n\n".join(potongan)
keluaran = os.path.join(AKAR, "tools", "_skrip_uji_cepat.py")
with open(keluaran, "w", encoding="utf-8") as f:
    f.write(skrip)
print(f"Skrip uji cepat ditulis: {keluaran} ({len(potongan)} potongan kode)")
