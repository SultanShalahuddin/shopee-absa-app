# Research artifacts

Folder ini menyimpan materi penelitian yang mendukung model ReviewScope. Isinya bukan bagian dari runtime backend atau bundle Vercel.

## Isi

```text
data/          Dataset raw, hasil cleaning, lexicon, dan label ABSA final
notebooks/     Cleaning/EDA, topic modeling, dan tuning IndoBERT
scripts/       Script pengumpulan data
figures/       Gambar hasil EDA
artifacts/     Metrik, prediksi validation/test, dan confusion matrix
```

Dataset final adalah `data/processed/Label_ABSA_Shopee_6176.xlsx` dengan SHA-256:

```text
73482b976a2daf573e28abba1f5ed7c891435899ff096bf5ee43825a9d8e2acb
```

`artifacts/absa-indobert-tuning-20261003/test_predictions.csv` merupakan golden test manifest. File tersebut berisi 4.944 pasangan dari 618 review x 8 aspek dan digunakan untuk membandingkan PyTorch, FP32 ONNX, dan INT8 ONNX pada input serta label yang sama.

Bobot FP32 tidak diduplikasi di sini. Model referensi lokal tetap berada di `backend/model/` dan dikecualikan dari Git karena ukurannya sekitar 475 MiB.

Folder impor asli tetap dipertahankan secara lokal tetapi dikecualikan dari Git:

- `final-project-shopee/`
- `ABSA_IndoBERT_Tuning_20261003_044443/`
