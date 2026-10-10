"""Export, quantize, and evaluate the ReviewScope IndoBERT model.

Run this module from ``backend`` so the active environment and relative paths
match the normal backend workflow::

    python -m scripts.onnx_pipeline

Generated ONNX weights stay under ``backend/model`` (ignored by Git). Compact
evaluation reports are written to ``research/artifacts/onnx-int8-evaluation``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
BACKEND_DIR = SCRIPT_PATH.parents[1]
REPOSITORY_DIR = SCRIPT_PATH.parents[2]
DEFAULT_MODEL_DIR = BACKEND_DIR / "model"
DEFAULT_ONNX_DIR = DEFAULT_MODEL_DIR / "onnx"
DEFAULT_GOLDEN_PATH = (
    REPOSITORY_DIR
    / "research"
    / "artifacts"
    / "absa-indobert-tuning-20261003"
    / "test_predictions.csv"
)
DEFAULT_REFERENCE_METRICS_PATH = DEFAULT_GOLDEN_PATH.with_name("test_metrics.json")
DEFAULT_REFERENCE_PER_ASPECT_PATH = DEFAULT_GOLDEN_PATH.with_name(
    "metrics_per_aspect.csv"
)
DEFAULT_REPORT_DIR = (
    REPOSITORY_DIR / "research" / "artifacts" / "onnx-int8-evaluation"
)

LABEL_IDS = (0, 1, 2, 3)
DEFAULT_LABELS = {
    0: "tidak_disebut",
    1: "negatif",
    2: "netral",
    3: "positif",
}
MACRO_F1_FLOOR = 0.837
MAX_MACRO_F1_DROP = 0.005
SENTIMENT_MENTIONED_F1_FLOOR = 0.810
MAX_PER_ASPECT_MACRO_F1_DROP = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Evaluate only ONNX artifacts; parity against PyTorch is then omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated ONNX artifacts.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPOSITORY_DIR)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "mib": round(path.stat().st_size / (1024 * 1024), 3),
        "sha256": sha256(path),
    }


def load_model_metadata(model_dir: Path) -> tuple[dict[int, str], dict[str, Any]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    absa_config = json.loads(
        (model_dir / "absa_config.json").read_text(encoding="utf-8")
    )
    labels = {
        int(key): str(value).lower() for key, value in config["id2label"].items()
    }
    if labels != DEFAULT_LABELS:
        raise RuntimeError(f"Mapping label berubah dan tidak aman diekspor: {labels}")
    if int(config.get("_num_labels", len(labels))) != 4:
        raise RuntimeError("Model referensi harus memiliki empat output classifier.")
    if int(absa_config["max_length"]) != 128:
        raise RuntimeError("absa_config.json tidak lagi menggunakan max_length=128.")
    return labels, absa_config


def load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "reviewId",
        "content",
        "aspect",
        "aspect_text",
        "labels",
        "sentiment_pred",
        "confidence",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise RuntimeError(f"Kolom golden manifest tidak lengkap: {sorted(missing)}")
    if len(rows) != 4_944:
        raise RuntimeError(f"Golden manifest harus memiliki 4.944 baris, bukan {len(rows)}.")
    return rows


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    true_positive = int(np.sum((y_true == label) & (y_pred == label)))
    false_positive = int(np.sum((y_true != label) & (y_pred == label)))
    false_negative = int(np.sum((y_true == label) & (y_pred != label)))
    return safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    class_f1 = [f1_for_label(y_true, y_pred, label) for label in LABEL_IDS]
    supports = [int(np.sum(y_true == label)) for label in LABEL_IDS]

    true_mentioned = (y_true != 0).astype(np.int64)
    pred_mentioned = (y_pred != 0).astype(np.int64)
    mentioned_mask = true_mentioned == 1
    sentiment_f1 = [
        f1_for_label(y_true[mentioned_mask], y_pred[mentioned_mask], label)
        for label in (1, 2, 3)
    ]
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_macro": float(statistics.fmean(class_f1)),
        "f1_weighted": safe_divide(
            sum(score * support for score, support in zip(class_f1, supports)),
            sum(supports),
        ),
        "f1_aspect_detection": f1_for_label(
            true_mentioned, pred_mentioned, 1
        ),
        "f1_sentiment_mentioned": float(statistics.fmean(sentiment_f1)),
    }


def metrics_per_aspect(
    rows: list[dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray
) -> list[dict[str, Any]]:
    aspect_order = list(dict.fromkeys(str(row["aspect"]) for row in rows))
    result = []
    for aspect in aspect_order:
        indices = np.array(
            [index for index, row in enumerate(rows) if row["aspect"] == aspect],
            dtype=np.int64,
        )
        result.append(
            {"aspect": aspect, **calculate_metrics(y_true[indices], y_pred[indices])}
        )
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def encode_batches(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, np.ndarray]], float]:
    encoded_batches = []
    started = time.perf_counter()
    for batch in batched(rows, batch_size):
        encoded = tokenizer(
            [str(row["content"]) for row in batch],
            [str(row["aspect_text"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        encoded_batches.append(
            {key: np.asarray(value, dtype=np.int64) for key, value in encoded.items()}
        )
    return encoded_batches, time.perf_counter() - started


def run_pytorch(model_dir: Path, batches: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    outputs = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            tensors = {key: torch.from_numpy(value) for key, value in batch.items()}
            outputs.append(model(**tensors).logits.detach().cpu().numpy())
    elapsed = time.perf_counter() - started
    del model
    gc.collect()
    return np.concatenate(outputs), elapsed


def run_onnx(path: Path, batches: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, float]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_names = {item.name for item in session.get_inputs()}
    outputs = []
    started = time.perf_counter()
    for batch in batches:
        feed = {key: value for key, value in batch.items() if key in input_names}
        outputs.append(session.run(["logits"], feed)[0])
    elapsed = time.perf_counter() - started
    del session
    gc.collect()
    return np.concatenate(outputs), elapsed


def export_fp32(model_dir: Path, output_path: Path, max_length: int) -> None:
    import onnx
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, wrapped: torch.nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
        ) -> torch.Tensor:
            return self.wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    wrapped = LogitsWrapper(model)
    sample = tokenizer(
        ["Aplikasinya mudah tetapi kurir lambat."],
        ["performa aplikasi"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        tuple(sample[name] for name in input_names),
        str(output_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17,
    )
    onnx.checker.check_model(onnx.load(str(output_path), load_external_data=True))
    del wrapped, model
    gc.collect()


def quantize_int8(input_path: Path, output_path: Path) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    onnx.checker.check_model(onnx.load(str(output_path), load_external_data=True))


def load_reference_per_aspect(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["aspect"]: {
                key: float(value) for key, value in row.items() if key != "aspect"
            }
            for row in csv.DictReader(handle)
        }


def compare_metrics(
    current: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    return {key: current[key] - float(reference[key]) for key in current}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size minimal 1")

    model_dir = args.model_dir.resolve()
    golden_path = args.golden.resolve()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    fp32_path = output_dir / "model_fp32.onnx"
    int8_path = output_dir / "model_int8.onnx"

    labels, absa_config = load_model_metadata(model_dir)
    if args.max_length != int(absa_config["max_length"]):
        raise SystemExit("--max-length harus sama dengan absa_config.json (128).")
    rows = load_golden(golden_path)

    reference_metrics = json.loads(
        DEFAULT_REFERENCE_METRICS_PATH.read_text(encoding="utf-8")
    )
    reference_per_aspect = load_reference_per_aspect(
        DEFAULT_REFERENCE_PER_ASPECT_PATH
    )
    y_true = np.array([int(row["labels"]) for row in rows], dtype=np.int64)
    golden_predictions = np.array(
        [next(key for key, value in labels.items() if value == row["sentiment_pred"]) for row in rows],
        dtype=np.int64,
    )
    reproduced_reference = calculate_metrics(y_true, golden_predictions)
    largest_reference_delta = max(
        abs(value) for value in compare_metrics(reproduced_reference, reference_metrics).values()
    )
    if largest_reference_delta > 1e-12:
        raise RuntimeError(
            "Implementasi metrik tidak mereproduksi test_metrics.json; "
            f"delta terbesar={largest_reference_delta}."
        )

    if fp32_path.exists() and not args.force:
        print(f"Memakai FP32 ONNX yang sudah ada: {fp32_path}")
    else:
        if fp32_path.exists():
            fp32_path.unlink()
        print("Mengekspor FP32 ONNX...")
        export_fp32(model_dir, fp32_path, args.max_length)

    if int8_path.exists() and not args.force:
        print(f"Memakai INT8 ONNX yang sudah ada: {int8_path}")
    else:
        if int8_path.exists():
            int8_path.unlink()
        print("Menerapkan dynamic INT8 quantization...")
        quantize_int8(fp32_path, int8_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    print(f"Tokenisasi {len(rows):,} pasangan...")
    batches, tokenization_seconds = encode_batches(
        tokenizer, rows, args.batch_size, args.max_length
    )

    logits_by_engine: dict[str, np.ndarray] = {}
    inference_seconds: dict[str, float] = {}
    if not args.skip_pytorch:
        print("Evaluasi PyTorch referensi...")
        logits_by_engine["pytorch"], inference_seconds["pytorch"] = run_pytorch(
            model_dir, batches
        )
    print("Evaluasi FP32 ONNX...")
    logits_by_engine["onnx_fp32"], inference_seconds["onnx_fp32"] = run_onnx(
        fp32_path, batches
    )
    print("Evaluasi INT8 ONNX...")
    logits_by_engine["onnx_int8"], inference_seconds["onnx_int8"] = run_onnx(
        int8_path, batches
    )

    predictions = {
        engine: np.argmax(logits, axis=1).astype(np.int64)
        for engine, logits in logits_by_engine.items()
    }
    probabilities = {engine: softmax(logits) for engine, logits in logits_by_engine.items()}
    metrics = {
        engine: calculate_metrics(y_true, values)
        for engine, values in predictions.items()
    }
    per_aspect = {
        engine: metrics_per_aspect(rows, y_true, values)
        for engine, values in predictions.items()
    }

    macro_floor = max(
        MACRO_F1_FLOOR,
        float(reference_metrics["f1_macro"]) - MAX_MACRO_F1_DROP,
    )
    int8_per_aspect_drops = {
        row["aspect"]: reference_per_aspect[row["aspect"]]["f1_macro"]
        - float(row["f1_macro"])
        for row in per_aspect["onnx_int8"]
    }
    gates: dict[str, Any] = {
        "golden_metrics_reproduced": largest_reference_delta <= 1e-12,
        "int8_macro_f1": metrics["onnx_int8"]["f1_macro"] >= macro_floor,
        "int8_sentiment_mentioned_f1": (
            metrics["onnx_int8"]["f1_sentiment_mentioned"]
            >= SENTIMENT_MENTIONED_F1_FLOOR
        ),
        "int8_per_aspect_macro_f1": all(
            drop <= MAX_PER_ASPECT_MACRO_F1_DROP
            for drop in int8_per_aspect_drops.values()
        ),
    }
    if "pytorch" in predictions:
        gates["pytorch_matches_golden_labels"] = bool(
            np.array_equal(predictions["pytorch"], golden_predictions)
        )
        gates["fp32_onnx_matches_pytorch_labels"] = bool(
            np.array_equal(predictions["onnx_fp32"], predictions["pytorch"])
        )
        gates["fp32_onnx_max_abs_logit_delta_le_1e-3"] = bool(
            np.max(
                np.abs(logits_by_engine["onnx_fp32"] - logits_by_engine["pytorch"])
            )
            <= 1e-3
        )

    comparisons: dict[str, Any] = {
        "reference_metrics": reference_metrics,
        "metrics": metrics,
        "metric_delta_from_reference": {
            engine: compare_metrics(values, reference_metrics)
            for engine, values in metrics.items()
        },
        "prediction_match_rate_with_golden": {
            engine: float(np.mean(values == golden_predictions))
            for engine, values in predictions.items()
        },
        "int8_per_aspect_macro_f1_drop": int8_per_aspect_drops,
        "gates": gates,
        "all_gates_passed": all(bool(value) for value in gates.values()),
    }
    if "pytorch" in logits_by_engine:
        comparisons["fp32_onnx_vs_pytorch"] = {
            "prediction_match_rate": float(
                np.mean(predictions["onnx_fp32"] == predictions["pytorch"])
            ),
            "max_abs_logit_delta": float(
                np.max(
                    np.abs(
                        logits_by_engine["onnx_fp32"]
                        - logits_by_engine["pytorch"]
                    )
                )
            ),
            "mean_abs_logit_delta": float(
                np.mean(
                    np.abs(
                        logits_by_engine["onnx_fp32"]
                        - logits_by_engine["pytorch"]
                    )
                )
            ),
        }

    report_dir.mkdir(parents=True, exist_ok=True)
    for engine, rows_for_engine in per_aspect.items():
        write_csv(report_dir / f"metrics_per_aspect_{engine}.csv", rows_for_engine)

    prediction_rows = []
    for index, row in enumerate(rows):
        output = {
            "reviewId": row["reviewId"],
            "aspect": row["aspect"],
            "label": labels[int(y_true[index])],
            "golden_prediction": labels[int(golden_predictions[index])],
        }
        for engine in predictions:
            predicted_id = int(predictions[engine][index])
            output[f"{engine}_prediction"] = labels[predicted_id]
            output[f"{engine}_confidence"] = round(
                float(probabilities[engine][index, predicted_id]), 8
            )
        prediction_rows.append(output)
    write_csv(report_dir / "predictions.csv", prediction_rows)

    runtime = {
        engine: {
            "inference_seconds": round(seconds, 6),
            "milliseconds_per_pair": round(seconds * 1000 / len(rows), 6),
            "pairs_per_second": round(len(rows) / seconds, 3),
            "estimated_ms_per_eight_aspect_review": round(
                seconds * 1000 * 8 / len(rows), 3
            ),
        }
        for engine, seconds in inference_seconds.items()
    }
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "test_pairs": len(rows),
        "test_reviews": len({row["reviewId"] for row in rows}),
        "tokenization_seconds": round(tokenization_seconds, 6),
        "artifacts": {
            "pytorch_fp32": file_info(model_dir / "model.safetensors"),
            "onnx_fp32": file_info(fp32_path),
            "onnx_int8": file_info(int8_path),
        },
        "runtime": runtime,
    }
    (report_dir / "comparison.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(comparisons, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if comparisons["all_gates_passed"]:
        print("ONNX INT8 LOLOS seluruh acceptance gate.")
        return 0
    print("ONNX INT8 BELUM LOLOS seluruh acceptance gate.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
