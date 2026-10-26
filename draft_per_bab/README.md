# Draft Skripsi per Bab

Hasil pemisahan `DRAFT_SKRIPSI_LSTM_GRU_-_AHMAD_GHULAM_GHAZI.docx` menjadi satu file per bab,
masing-masing dalam format Word (`word/`) dan PDF (`pdf/`).

| File | Isi | Halaman awal |
|---|---|---|
| `00_Bagian_Awal` | Halaman judul, Daftar Isi, Daftar Tabel, Daftar Gambar, Daftar Lampiran, Daftar Notasi | i |
| `01_BAB_I_Pendahuluan` | BAB I Pendahuluan (termasuk Landasan Teori) | 1 |
| `02_BAB_II_Metodologi_Penelitian` | BAB II Metodologi Penelitian | 23 |
| `03_BAB_III_Pembahasan` | BAB III Pembahasan | 28 |
| `04_BAB_IV_Penutup` | BAB IV Penutup | 61 |
| `05_Daftar_Pustaka` | Daftar Pustaka | 63 |
| `06_Lampiran` | Lampiran 1–3 | 66 |

## Yang dijaga tetap sama dengan file utama

- Seluruh teks, persamaan, tabel, gambar, dan format paragraf disalin apa adanya dari file utama
  (diperiksa elemen per elemen; gabungan teks semua bagian identik dengan file utama).
- Style, ukuran kertas, margin, header/footer, dan font yang di-*embed* tetap sama.
- Nomor halaman setiap bab dimulai dari nomor yang tercantum di Daftar Isi, dan halaman pertama
  setiap bab tetap tanpa nomor, sama seperti di file utama.
- Nomor Tabel, Gambar, dan Lampiran melanjutkan nomor di file utama (misalnya BAB III tetap dimulai
  dari Tabel 7 dan Gambar 9), juga jika *field* di-*update* di Word.

## Catatan

- File utama `DRAFT_SKRIPSI_LSTM_GRU_-_AHMAD_GHULAM_GHAZI.docx` tidak diubah. Jika ada revisi,
  sebaiknya edit di file utama lalu pisahkan ulang.
- Jangan meng-*update* Daftar Isi/Daftar Tabel/Daftar Gambar di `00_Bagian_Awal.docx`. Karena
  bab-babnya ada di file lain, Word akan mengosongkan daftar tersebut. Nomor halaman yang tampil
  sekarang adalah nomor dari file utama.
- PDF dibuat dengan LibreOffice (bukan Microsoft Word) memakai font yang sama dengan dokumen
  (Arial, serta Cambria Math dan lainnya dari font yang ter-*embed*). Jumlah halaman setiap bab
  sama dengan Daftar Isi (BAB I 22, BAB II 5, BAB III 33, BAB IV 2, Daftar Pustaka 3 halaman), dan
  semua persamaan sudah diperiksa tampil lengkap, termasuk vektor tebal (**W**, **h**, **x**, …),
  superskrip (in)/(rec), dan DM\*. Bentuk huruf rumus di PDF sedikit berbeda dari Cambria Math di
  Word, dan beberapa baris perhitungan panjang di BAB III sedikit masuk ke margin kanan (tidak
  terpotong). Untuk PDF yang persis seperti tampilan Word, buka file `.docx` di Word lalu pilih
  *File → Save As → PDF*.
