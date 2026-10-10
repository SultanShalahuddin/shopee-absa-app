# ReviewScope ABSA — Project Handoff

Terakhir diperbarui: 11 Oktober 2026

## 1. Ringkasan status

ReviewScope menganalisis satu ulasan Shopee berbahasa Indonesia terhadap
delapan aspek tetap. Setiap aspek mendapat salah satu label
`tidak_disebut`, `negatif`, `netral`, atau `positif`.

Status saat ini:

- Frontend Next.js production aktif di
  `https://reviewscope-absa-app.vercel.app`.
- Backend FastAPI production aktif di
  `https://reviewscope-absa-api.vercel.app` menggunakan ONNX Runtime CPU dan
  model dynamic INT8.
- Parity FP32 ONNX dan evaluasi INT8 pada golden test set sudah lulus.
- Model deployment sudah berada di repository Hugging Face private.
- Runtime tokenizer, smoke test, ukuran lokal, latency, dan memori sudah diuji.
- Source tersedia di `https://github.com/SultanShalahuddin/shopee-absa-app`.
- Bundle backend production dilaporkan sebesar 286,98 MB dan memakai Vercel
  Large Functions pada Fluid Compute.
- Health check production memuat model asli dengan `demo_mode=false`; satu uji
  prediction menghasilkan delapan aspek dalam 250 ms dan tersimpan ke Supabase.
- CORS production sudah membolehkan domain frontend final.
- Pengukuran cold start dan memory telemetry pada runtime Vercel masih tersisa.

Alur inti frontend, API, model, dan database sudah tersedia di Vercel. Pekerjaan
tersisa berfokus pada observability, rotasi secret, dan monitoring pascadeploy.

## 2. Arsitektur final

```text
Browser
  └─ Next.js frontend — Vercel Hobby
       └─ FastAPI backend — Vercel Python Function
            ├─ ONNX Runtime CPU + IndoBERT INT8
            ├─ Supabase Free — persistence
            └─ Private Hugging Face repo — model fetched saat build
```

Keputusan yang berlaku:

- Frontend: Next.js 15 dan React 19.
- Backend: FastAPI sebagai satu Vercel Python Function.
- Inference: ONNX Runtime CPU, dynamic INT8.
- Database: Supabase Free.
- Model storage: private Hugging Face repository.
- Runtime target: 2 GB RAM dan 1 vCPU.

Google Cloud Run, Oracle Cloud, Docker deployment, dan Streamlit bukan bagian
dari jalur deployment aktif.

## 3. Kontrak model

Model referensi adalah satu shared `BertForSequenceClassification` hasil
fine-tuning `indobenchmark/indobert-base-p2`, bukan delapan model terpisah.

| Properti | Nilai |
|---|---|
| Classifier outputs | 4 |
| Parameters | 124,444,420 |
| Hidden size | 768 |
| Encoder layers | 12 |
| Attention heads | 12 |
| Vocabulary | 50,000 |
| Maximum length | 128 token |
| Training seed | 42 |
| Selected trial | `sqrt_weight` |
| API model version | `indobert-sqrt-weight-int8-v1` |

Label mapping:

```json
{
  "0": "tidak_disebut",
  "1": "negatif",
  "2": "netral",
  "3": "positif"
}
```

Satu request membentuk delapan pasangan `(review_text, aspect_text)`, lalu
memproses seluruh pasangan dalam satu batch. Urutan dan prompt berikut tidak
boleh diubah tanpa evaluasi ulang:

| Aspect key | Prompt text |
|---|---|
| `iklan_dan_konten` | `iklan dan konten` |
| `pengiriman_dan_kurir` | `pengiriman dan kurir` |
| `performa_aplikasi` | `performa aplikasi` |
| `akun_login_keamanan` | `akun login dan keamanan` |
| `pembayaran_cod_dan_keuangan` | `pembayaran COD dan keuangan` |
| `promo_voucher_harga_dan_biaya` | `promo voucher harga dan biaya` |
| `penjual_dan_kualitas_produk` | `penjual dan kualitas produk` |
| `retur_refund_dan_layanan` | `retur refund dan layanan` |

## 4. Data dan sumber yang otoritatif

| Item | Path / nilai |
|---|---|
| Dataset terkurasi | `research/data/processed/Label_ABSA_Shopee_6176.xlsx` |
| Dataset SHA-256 | `73482b976a2daf573e28abba1f5ed7c891435899ff096bf5ee43825a9d8e2acb` |
| Jumlah review | 6,176 |
| Train / validation / test | 4,940 / 618 / 618 review |
| Golden test manifest | `research/artifacts/absa-indobert-tuning-20261003/test_predictions.csv` |
| Golden test pairs | 4,944 pasangan |
| Per-aspect baseline | `research/artifacts/absa-indobert-tuning-20261003/metrics_per_aspect.csv` |
| Training notebook | `research/notebooks/03_absa_indobert_tuning.ipynb` |
| Model config | `backend/model/config.json` |
| ABSA config | `backend/model/absa_config.json` |

`backend/model/config.json` adalah sumber kebenaran untuk jumlah label. Arsip
training lama memiliki `_num_labels=5`, tetapi mapping label dan classifier
weights hanya memiliki empat output. Config backend sudah dikoreksi menjadi 4.

## 5. Baseline PyTorch

### Validation

| Metric | Nilai |
|---|---:|
| Accuracy | 0.965008 |
| Macro F1 | 0.826657 |
| Weighted F1 | 0.965018 |
| Aspect-detection F1 | 0.951601 |
| Sentiment-mentioned F1 | 0.796350 |
| Loss | 0.478749 |

### Test

| Metric | Nilai |
|---|---:|
| Accuracy | 0.969458 |
| Macro F1 | 0.842376 |
| Weighted F1 | 0.969273 |
| Aspect-detection F1 | 0.949179 |
| Sentiment-mentioned F1 | 0.815427 |

### Test per aspek

| Aspect | Accuracy | Macro F1 | Weighted F1 | Detection F1 | Mentioned F1 |
|---|---:|---:|---:|---:|---:|
| `performa_aplikasi` | 0.957929 | 0.881794 | 0.957794 | 0.955326 | 0.873472 |
| `retur_refund_dan_layanan` | 0.982201 | 0.829902 | 0.980853 | 0.958333 | 0.782105 |
| `promo_voucher_harga_dan_biaya` | 0.959547 | 0.822462 | 0.959260 | 0.977778 | 0.776844 |
| `pengiriman_dan_kurir` | 0.959547 | 0.787806 | 0.957466 | 0.959596 | 0.730778 |
| `akun_login_keamanan` | 0.978964 | 0.746410 | 0.978391 | 0.972222 | 0.662452 |
| `pembayaran_cod_dan_keuangan` | 0.972492 | 0.741428 | 0.972345 | 0.915888 | 0.692814 |
| `iklan_dan_konten` | 0.983819 | 0.683158 | 0.983671 | 0.986486 | 0.583136 |
| `penjual_dan_kualitas_produk` | 0.961165 | 0.607867 | 0.961774 | 0.787234 | 0.563158 |

Accuracy saja tidak cukup karena kelas `tidak_disebut` dominan. Setiap
perubahan model harus tetap mengevaluasi macro F1, weighted F1, detection F1,
mentioned-sentiment F1, dan metrik per aspek.

## 6. Hasil migrasi ONNX INT8

Evaluasi dilakukan pada golden manifest yang sama, berisi 4,944 pasangan dari
618 review.

| Metric | PyTorch | FP32 ONNX | INT8 ONNX | Delta INT8 |
|---|---:|---:|---:|---:|
| Accuracy | 0.969458 | 0.969458 | 0.969256 | -0.000202 |
| Macro F1 | 0.842376 | 0.842376 | 0.840736 | -0.001640 |
| Weighted F1 | 0.969273 | 0.969273 | 0.969070 | -0.000203 |
| Aspect-detection F1 | 0.949179 | 0.949179 | 0.949922 | +0.000743 |
| Sentiment-mentioned F1 | 0.815427 | 0.815427 | 0.812758 | -0.002668 |

Parity dan quality gates:

| Gate | Hasil |
|---|---|
| PyTorch mereproduksi golden labels | Lulus, 100% |
| FP32 ONNX vs PyTorch labels | Lulus, 100% |
| Maximum absolute logit delta | `0.0000185966` |
| INT8 vs golden prediction agreement | 99.8786% |
| INT8 macro F1 minimal sekitar 0.837 | Lulus |
| Macro F1 drop maksimal 0.005 | Lulus |
| Mentioned-sentiment F1 minimal 0.810 | Lulus |
| Per-aspect macro F1 drop maksimal 0.01 | Lulus |

Penurunan macro F1 per aspek terbesar adalah `0.007292` pada
`promo_voucher_harga_dan_biaya`, masih di bawah gate `0.01`.

Laporan lengkap berada di `research/artifacts/onnx-int8-evaluation/`.

## 7. Artefak dan performa lokal

| Artefak | Ukuran | SHA-256 |
|---|---:|---|
| PyTorch FP32 | 474.740 MiB | `b29e93a0fc885915b90348c99cacddff3ebbe5c3273cc3d6c6880b6547dc948b` |
| FP32 ONNX | 474.991 MiB | `5b6aede74171e3c5595bd6401c36b337bd77204eb4841554dbd81e8765ed4699c` |
| INT8 ONNX | 230.897 MiB | `bc9668f83d7e40bc96feb79a5fd39c4bc2a8b07b618b26b63d529fe41fca3086` |

Benchmark lokal Windows dengan ONNX Runtime dibatasi ke satu intra-op dan satu
inter-op thread:

| Pengukuran | Nilai |
|---|---:|
| Model load | 403.639 ms |
| First eight-aspect inference | 98.320 ms |
| Warm median inference | 96.875 ms |
| Peak working set | 454.949 MiB |
| Private usage | 1,211.973 MiB |

Batch-32 evaluation memperkirakan waktu model INT8 sebesar 131.619 ms per
review delapan aspek, dibandingkan 288.025 ms untuk PyTorch. Angka tersebut
adalah hasil lokal, bukan jaminan cold start Vercel.

Estimasi payload produksi bersih:

| Komponen | Ukuran |
|---|---:|
| Model assets | 231.577 MiB |
| Runtime dependencies | 126.873 MiB |
| Total estimasi | 358.469 MiB |
| Estimasi ruang ke batas 500 MiB | 141.531 MiB |

Vercel melaporkan bundle function production sebesar 286,98 MB. Standard path
deployment menerapkan batas efektif 225 MB pada build ini, sehingga backend
dideploy sebagai Large Function dengan `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` pada
Fluid Compute. Angka ini adalah ukuran yang dilaporkan Vercel dan tidak harus
sama dengan estimasi payload lokal di atas.

## 8. Backend dan API

Backend production memakai:

- `onnxruntime`, `tokenizers`, dan `numpy` untuk inference.
- FastAPI, Pydantic, dan HTTPX untuk API dan persistence.
- Cached ONNX Runtime session dengan satu thread.
- `tokenizer.json` langsung; `torch` dan `transformers` hanya ada pada
  dependency tooling konversi.
- Demo fallback dinonaktifkan secara default. Model yang tidak tersedia harus
  menghasilkan startup failure, bukan prediksi palsu.

Kontrak endpoint:

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | Status model dan konfigurasi database |
| POST | `/api/v1/predict` | Prediksi delapan aspek dan simpan hasil |
| GET | `/api/v1/history?limit=8` | Ambil riwayat analisis |

Respons prediction harus mempertahankan delapan aspect key, sentiment label,
confidence, model version, latency, status persistence, dan demo flag yang
sudah digunakan frontend.

## 9. Database

- Provider: Supabase Free.
- Tabel: `reviews` dan `aspect_predictions`.
- Satu request tersimpan sebagai satu row `reviews` dan delapan row
  `aspect_predictions`.
- Row-level security tetap aktif; privileged write hanya dilakukan backend.
- `SUPABASE_URL` harus berupa base project URL tanpa suffix `/rest/v1`.
- Smoke test lokal sengaja menonaktifkan persistence; integrasi database harus
  diuji terpisah sebelum production.
- Privileged Supabase key harus disimpan server-side dan sebaiknya dirotasi
  sebelum deployment publik.

## 10. Model deployment

Repository model private:

```text
MODEL_ID=sultan028/reviewscope-absa-int8
MODEL_REVISION=4230a98d3bf26a3c154b57b76c1827d604dc285d
MODEL_SHA256=bc9668f83d7e40bc96feb79a5fd39c4bc2a8b07b618b26b63d529fe41fca3086
```

File yang tersedia pada revision tersebut:

```text
model_int8.onnx
config.json
absa_config.json
tokenizer.json
tokenizer_config.json
```

Dedicated fine-grained read-only token sudah diuji terhadap repository private.
`backend/scripts/download_model.py` mengunduh kelima file saat build dan
memverifikasi checksum serta kontrak config. Token tidak boleh dicetak,
dikirim ke browser, atau dimasukkan ke Git.

## 11. Environment variables

Backend:

```text
APP_ENV
APP_NAME
MODEL_DIR
MODEL_ID
MODEL_REVISION
MODEL_SHA256
MODEL_VERSION
MAX_LENGTH
ALLOW_DEMO_MODE
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
CORS_ORIGINS
HF_TOKEN
```

Frontend:

```text
NEXT_PUBLIC_API_URL
```

`HF_TOKEN` dan `SUPABASE_SERVICE_ROLE_KEY` adalah server-only secrets. Jangan
menaruhnya dalam `NEXT_PUBLIC_*`, source code, log, dokumentasi, atau response.

## 12. Validasi yang sudah selesai

- Model directory dan classifier output tervalidasi.
- PyTorch reference mereproduksi golden manifest.
- FP32 ONNX parity lulus.
- INT8 quality gates lulus.
- Runtime tokenizer identik dengan Transformers untuk seluruh 4,944 pasangan.
- Backend smoke test lulus dalam mode `ONNX INT8 IndoBERT`.
- Frontend type-check/build terakhir lulus.
- Dependency frontend terakhir memiliki 0 audit vulnerability.
- Secret scan pada kandidat commit terakhir tidak menemukan credential.
- Tidak ada file non-ignored lebih besar dari 10 MiB.
- FP32 model, ONNX weights, `.env`, virtual environments, dan folder impor asli
  tetap lokal serta diabaikan oleh Git.
- Model private Hugging Face dan read-only access sudah diverifikasi.
- Repository GitHub dan kedua Vercel project production sudah aktif.
- `/health` production mengembalikan `model_loaded=true`, `demo_mode=false`,
  `model_format=onnx-int8`, dan `database_enabled=true`.
- Prediction production menghasilkan delapan aspek dalam 250 ms dengan
  `stored=true`; satu row review dan delapan row aspek berhasil dipersistenkan.
- Frontend production merespons HTTP 200 dan preflight CORS backend mengizinkan
  origin `https://reviewscope-absa-app.vercel.app`.

## 13. Perintah verifikasi lokal

Dari folder `backend`:

```powershell
python -m scripts.verify_model model
python -m scripts.verify_tokenizer_parity
python -m scripts.smoke_test
python -m scripts.benchmark_runtime
```

Pipeline konversi lengkap membutuhkan dependency tooling:

```powershell
pip install -r requirements-onnx.txt
python -m scripts.onnx_pipeline
```

## 14. Pekerjaan tersisa

Tindak lanjut pascadeploy:

1. Jalankan smoke test UI dari browser dan periksa hasil serta history.
2. Ukur cold start, warm inference, dan memory pada runtime Vercel secara lebih
   terkontrol; latency 250 ms yang tercatat berasal dari satu request production.
3. Rotasi privileged Supabase key yang pernah digunakan selama debugging, lalu
   perbarui environment variable Vercel tanpa memasukkannya ke Git.
4. Pantau log untuk kegagalan unduhan model, startup, persistence, dan timeout.
5. Pastikan tidak ada token pada browser bundle, network response, atau log.

## 15. Batasan dan prinsip keselamatan

- Jangan menghapus model FP32; model tersebut tetap menjadi reference/fallback.
- Jangan mengubah aspect order, prompt, label mapping, atau `max_length=128`
  tanpa evaluasi ulang.
- Jangan menerima model baru hanya berdasarkan accuracy.
- Jangan mengaktifkan demo mode pada production.
- Jangan memasukkan `research/` atau model reference ke runtime bundle.
- Jangan mengunduh model saat cold start; model harus tersedia dari build.
- Jangan melakukan deployment cloud alternatif tanpa keputusan arsitektur baru.
- Supabase Free dapat pause setelah tidak aktif; health dan error handling harus
  memperhitungkannya.
