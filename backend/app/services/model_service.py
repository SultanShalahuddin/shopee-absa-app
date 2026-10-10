import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.schemas import AspectPrediction

logger = logging.getLogger(__name__)

ASPECT_DISPLAY_NAMES: dict[str, str] = {
    "iklan_dan_konten": "Iklan dan konten",
    "pengiriman_dan_kurir": "Pengiriman dan kurir",
    "performa_aplikasi": "Performa aplikasi",
    "akun_login_keamanan": "Akun, login, dan keamanan",
    "pembayaran_cod_dan_keuangan": "Pembayaran, COD, dan keuangan",
    "promo_voucher_harga_dan_biaya": "Promo, voucher, harga, dan biaya",
    "penjual_dan_kualitas_produk": "Penjual dan kualitas produk",
    "retur_refund_dan_layanan": "Retur, refund, dan layanan",
}

DEFAULT_ASPECT_TEXT: dict[str, str] = {
    "iklan_dan_konten": "iklan dan konten",
    "pengiriman_dan_kurir": "pengiriman dan kurir",
    "performa_aplikasi": "performa aplikasi",
    "akun_login_keamanan": "akun login dan keamanan",
    "pembayaran_cod_dan_keuangan": "pembayaran COD dan keuangan",
    "promo_voucher_harga_dan_biaya": "promo voucher harga dan biaya",
    "penjual_dan_kualitas_produk": "penjual dan kualitas produk",
    "retur_refund_dan_layanan": "retur refund dan layanan",
}

DEFAULT_ID2LABEL = {0: "tidak_disebut", 1: "negatif", 2: "netral", 3: "positif"}
ONNX_FILENAMES = ("model_int8.onnx", "onnx/model_int8.onnx")


class ABSAModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: Any = None
        self.tokenizer: Any = None
        self.input_names: set[str] = set()
        self.device = "cpu"
        self.id2label = DEFAULT_ID2LABEL.copy()
        self.aspect_keys = list(ASPECT_DISPLAY_NAMES)
        self.aspect_text = DEFAULT_ASPECT_TEXT.copy()
        self.demo_mode = False
        self.source = ""
        self.model_format = "demo-rules"
        self._load()

    def _resolve_source(self) -> tuple[Path, Path] | None:
        model_dir = self.settings.model_dir.resolve()
        if not model_dir.exists():
            return None
        required = (model_dir / "config.json", model_dir / "tokenizer.json")
        if not all(path.is_file() for path in required):
            return None
        for filename in ONNX_FILENAMES:
            model_path = model_dir / filename
            if model_path.is_file():
                return model_dir, model_path
        return None

    def _load(self) -> None:
        resolved = self._resolve_source()
        if resolved is None:
            if not self.settings.allow_demo_mode:
                location = self.settings.model_dir.resolve()
                repository_hint = (
                    f" MODEL_ID={self.settings.model_id!r} dikonfigurasi; unduh artefak "
                    "saat build dengan scripts.download_model."
                    if self.settings.model_id
                    else ""
                )
                raise RuntimeError(
                    "Model ONNX INT8 atau tokenizer tidak ditemukan di "
                    f"{location}.{repository_hint} ALLOW_DEMO_MODE=false."
                )
            self.demo_mode = True
            self.source = "demo-rules"
            logger.warning("Model tidak ditemukan; API berjalan dalam mode demo.")
            return

        model_dir, model_path = resolved
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_names = {item.name for item in self.session.get_inputs()}
            expected_inputs = {"input_ids", "attention_mask", "token_type_ids"}
            if self.input_names != expected_inputs:
                raise RuntimeError(
                    f"Input ONNX tidak sesuai: {sorted(self.input_names)}"
                )

            self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            self.tokenizer.enable_truncation(max_length=self.settings.max_length)
            self.tokenizer.enable_padding(pad_id=0, pad_type_id=0, pad_token="[PAD]")
            self._load_model_config(model_dir)
            self._load_absa_config(model_dir)
            self.source = str(model_path)
            self.model_format = "onnx-int8"
            logger.info("Model ONNX INT8 berhasil dimuat dari %s", model_path)
        except Exception:
            logger.exception("Gagal memuat model ONNX dari %s", model_path)
            if not self.settings.allow_demo_mode:
                raise
            self.demo_mode = True
            self.source = "demo-rules"
            self.model_format = "demo-rules"

    def _load_model_config(self, source: Path) -> None:
        payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
        raw_mapping = payload.get("id2label", {})
        parsed = {int(key): str(value).lower() for key, value in raw_mapping.items()}
        if parsed != DEFAULT_ID2LABEL:
            raise RuntimeError(f"Mapping label model tidak sesuai: {parsed}")

        configured_count = int(payload.get("_num_labels", len(parsed)))
        if configured_count != 4:
            raise RuntimeError(
                f"Classifier ONNX harus memiliki empat output, bukan {configured_count}."
            )
        output_shape = self.session.get_outputs()[0].shape
        if output_shape[-1] != 4:
            raise RuntimeError(f"Output ONNX tidak sesuai: {output_shape}")
        self.id2label = parsed

    def _load_absa_config(self, source: Path) -> None:
        config_path = source / "absa_config.json"
        if not config_path.exists():
            raise RuntimeError("absa_config.json wajib tersedia bersama model ONNX.")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if int(payload.get("max_length", -1)) != self.settings.max_length:
            raise RuntimeError(
                "MAX_LENGTH berbeda dari absa_config.json: "
                f"{self.settings.max_length} != {payload.get('max_length')}"
            )

        raw_mapping = payload.get("id2label", {})
        parsed = {int(key): str(value).lower() for key, value in raw_mapping.items()}
        if parsed != DEFAULT_ID2LABEL:
            raise RuntimeError(f"Mapping label ABSA tidak sesuai: {parsed}")

        configured_keys = payload.get("aspect_columns", [])
        configured_text = payload.get("aspect_text", {})
        if set(configured_keys) != set(ASPECT_DISPLAY_NAMES):
            raise RuntimeError("Daftar aspek pada absa_config.json tidak lengkap.")
        if set(configured_text) != set(ASPECT_DISPLAY_NAMES):
            raise RuntimeError("Prompt aspek pada absa_config.json tidak lengkap.")
        self.aspect_keys = list(configured_keys)
        self.aspect_text = {
            key: str(configured_text[key]) for key in self.aspect_keys
        }

    def predict(self, review_text: str) -> list[AspectPrediction]:
        if self.demo_mode:
            return self._demo_predict(review_text)

        aspect_keys = self.aspect_keys
        encoded = self.tokenizer.encode_batch(
            [(review_text, self.aspect_text[key]) for key in aspect_keys]
        )
        inputs = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [item.type_ids for item in encoded], dtype=np.int64
            ),
        }
        logits = self.session.run(
            ["logits"],
            {name: value for name, value in inputs.items() if name in self.input_names},
        )[0]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        predicted_ids = np.argmax(probabilities, axis=-1)
        confidence = np.max(probabilities, axis=-1)

        return [
            AspectPrediction(
                aspect=aspect,
                aspect_label=ASPECT_DISPLAY_NAMES[aspect],
                sentiment=self.id2label[int(label_id)],
                confidence=round(float(score), 6),
            )
            for aspect, label_id, score in zip(
                aspect_keys, predicted_ids, confidence
            )
        ]

    def _demo_predict(self, review_text: str) -> list[AspectPrediction]:
        text = review_text.lower()
        mentions = {
            "iklan_dan_konten": r"iklan|konten|notifikasi|live",
            "pengiriman_dan_kurir": r"kirim|kurir|paket|antar|sampai",
            "performa_aplikasi": r"aplikasi|app|error|lemot|lambat|cepat|bug",
            "akun_login_keamanan": r"akun|login|masuk|otp|keamanan|hack",
            "pembayaran_cod_dan_keuangan": r"bayar|pembayaran|cod|saldo|transfer|spaylater",
            "promo_voucher_harga_dan_biaya": r"promo|voucher|harga|diskon|ongkir|biaya",
            "penjual_dan_kualitas_produk": r"penjual|seller|produk|barang|kualitas|original",
            "retur_refund_dan_layanan": r"retur|refund|kembali|pengembalian|layanan|cs",
        }
        positive = r"bagus|baik|cepat|mudah|mantap|puas|murah|ramah|original|suka"
        negative = r"buruk|jelek|lambat|lemot|error|gagal|mahal|kecewa|rusak|susah|tidak bisa|ga bisa|nggak bisa"
        neutral = r"biasa|cukup|standar|lumayan"
        results = []
        for aspect, pattern in mentions.items():
            if not re.search(pattern, text):
                sentiment, confidence = "tidak_disebut", 0.93
            else:
                neg = len(re.findall(negative, text))
                pos = len(re.findall(positive, text))
                neu = len(re.findall(neutral, text))
                if neg > pos and neg >= neu:
                    sentiment, confidence = "negatif", 0.84
                elif pos > neg and pos >= neu:
                    sentiment, confidence = "positif", 0.86
                else:
                    sentiment, confidence = "netral", 0.68
            results.append(
                AspectPrediction(
                    aspect=aspect,
                    aspect_label=ASPECT_DISPLAY_NAMES[aspect],
                    sentiment=sentiment,
                    confidence=confidence,
                )
            )
        return results
