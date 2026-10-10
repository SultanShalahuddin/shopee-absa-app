"""Verify that the lightweight runtime tokenizer matches Transformers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = BACKEND_DIR.parent
MODEL_DIR = BACKEND_DIR / "model"
GOLDEN_PATH = (
    REPOSITORY_DIR
    / "research"
    / "artifacts"
    / "absa-indobert-tuning-20261003"
    / "test_predictions.csv"
)
MAX_LENGTH = 128
BATCH_SIZE = 32


def main() -> int:
    with GOLDEN_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    reference = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    runtime = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    runtime.enable_truncation(max_length=MAX_LENGTH)
    runtime.enable_padding(pad_id=0, pad_type_id=0, pad_token="[PAD]")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        contents = [row["content"] for row in batch]
        aspects = [row["aspect_text"] for row in batch]
        expected = reference(
            contents,
            aspects,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        encoded = runtime.encode_batch(list(zip(contents, aspects)))
        actual = {
            "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.asarray(
                [item.attention_mask for item in encoded], dtype=np.int64
            ),
            "token_type_ids": np.asarray(
                [item.type_ids for item in encoded], dtype=np.int64
            ),
        }
        for name, values in actual.items():
            if not np.array_equal(np.asarray(expected[name]), values):
                print(f"ERROR: tokenizer berbeda pada batch {start}, input {name}.")
                return 1

    print(
        "Tokenizer runtime parity passed "
        f"({len(rows):,} pasangan, max_length={MAX_LENGTH})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
