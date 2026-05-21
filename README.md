# MCF 2026 — Data Science Competition (DSC)

## 🏆 Result

| Metric | Score (MAPE) |
|---|---|
| Public Score | 4.98153 |
| Private Score | 13.10533 (8th Place) |


Naik **98 peringkat** pada private leaderboard🚀

## 📋 Overview

Produk asuransi kesehatan individu merupakan salah satu alat untuk mitigasi risiko finansial individu dari kejadian tidak terduga seperti penyakit atau kecelakaan, yang berpotensi menimbulkan biaya besar dan berdampak ke perencanaan keuangan pribadi.

Salah satu isu yang terjadi beberapa tahun terakhir adalah terkait peningkatan klaim asuransi kesehatan individu, dimana terjadi kenaikan sebesar 25.5% antara periode Januari–Juni 2025 jika dibandingkan periode yang sama di tahun 2024. Peningkatan klaim tersebut berdampak ke penyesuaian premi yang berpotensi mengakibatkan harga premi produk asuransi kesehatan individu menjadi lebih tidak terjangkau. Oleh karena itu, diperlukan analisa berbasis data untuk memprediksi faktor-faktor yang paling berpengaruh terhadap nilai klaim, sehingga dapat dilakukan inisiatif dari segi seleksi risiko, pencegahan, ataupun deteksi dini untuk meminimalisir dampak peningkatan klaim.

**Task:** Membangun model prediktif untuk memprediksi **trend frekuensi**, **trend severitas**, dan **trend total klaim** untuk periode Agustus hingga Desember 2025 pada level per-polis per-bulan.

**Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

## 🧠 Approach

### Arsitektur: Trinity Blending

Pendekatan utama menggunakan **Trinity Blending**, yaitu kombinasi empat model LightGBM yang masing-masing memiliki objektif berbeda dan diprediksi secara simultan:

| Model | Objektif LightGBM | Target |
|---|---|---|
| Model 1 — Classifier | `binary` | Probabilitas klaim terjadi (`Is_Claim`) |
| Model 2 — Frequency | `poisson` | Frekuensi klaim bulanan (`Target_Freq`) |
| Model 3 — Severity | `tweedie` (power=1.5) | Rata-rata severitas klaim yang di-scale |
| Model 4 — Total | `tweedie` (power=1.5) | Total nominal klaim yang di-scale |

Prediksi akhir merupakan hasil blending dari keempat komponen tersebut, yang kemudian di-denormalisasi (dikali 1 juta) untuk mendapatkan nilai dalam Rupiah.

### Feature Engineering

Fitur dibangun melalui fungsi `build_features_stable` dengan pendekatan **panel data** (polis × bulan), menggunakan fitur-fitur berikut:

- **Demografi statis:** Usia saat anchor (`Usia_Saat_Anchor`), durasi polis (`Durasi_Polis_Anchor`), Plan Code, Gender, Domisili
- **Rekam jejak historis:** Rata-rata frekuensi klaim per bulan (`Avg_Freq_Per_Bulan`), rata-rata nominal per bulan (`Avg_Nominal_Per_Bulan`)
- **Lag features:** Frekuensi dan nominal bulan terakhir (`Freq_Bulan_Terakhir`, `Nominal_Bulan_Terakhir`)
- **Fitur temporal:** Bulan angka (`Bulan_Angka`), horizon prediksi ke depan dalam bulan (`Horizon_Bulan`)

Fitur kategoris (Plan Code, Gender, Domisili) di-encode sebagai tipe `category` native LightGBM untuk menghindari overfitting.

### Training Strategy

- **Target scaling:** Nilai nominal dibagi 1 juta sebelum training untuk stabilitas numerik
- **Stacked training:** Data training dibangun menggunakan beberapa titik anchor historis (Juni 2024 – Januari 2025), masing-masing dengan target 1–5 bulan ke depan, lalu di-stack menjadi satu dataset training yang lebih besar
- **Walk-forward CV:** Validasi dilakukan dengan mensimulasikan prediksi pada periode Maret–Juli 2025 sebelum final submission

## 📁 Repository Structure

```
mcf-2026-dsc/
├── notebooks/
│   ├── notebook-utama.ipynb       # Notebook utama: EDA, feature engineering, Trinity Blending
│   └── notebook-eksperimen.ipynb  # Eksperimen tambahan (bukan solusi final)
└── README.md
```

> **Catatan:** Folder `dataset/` tidak disertakan dalam repository karena data bersifat kompetitif dan tidak boleh dipublikasikan. Dataset terdiri dari `Data_Klaim.csv` dan `Data_Polis.csv`.

## 👥 Team

**Nama Tim:** DC Comic

| Nama |
|---|
| Muhammad Naufal Muzaki |
| Marvel Irawan |
| Rafsanjani |

## Referensi

B. So and E. A. Valdez, "Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics," *arXiv preprint* arXiv:2406.16206v2, Oct. 2024.
