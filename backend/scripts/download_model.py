"""Download and verify the private Hugging Face deployment artifacts.

This script is intended for the Vercel build step. It never prints the Hugging
Face token and refuses an unpinned model or a model without an expected SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import httpx


REQUIRED_FILES = (
    "model_int8.onnx",
    "config.json",
    "absa_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
LABELS = {
    "0": "tidak_disebut",
    "1": "negatif",
    "2": "netral",
    "3": "positif",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} wajib diisi.")
    return value


def download_file(client: httpx.Client, url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                handle.write(chunk)
    temporary.replace(destination)


def validate_assets(model_dir: Path, expected_model_sha256: str) -> None:
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Artefak model belum lengkap: {missing}")

    actual_sha256 = sha256(model_dir / "model_int8.onnx")
    if actual_sha256.lower() != expected_model_sha256.lower():
        raise RuntimeError(
            "SHA-256 model_int8.onnx tidak cocok: "
            f"expected={expected_model_sha256.lower()}, actual={actual_sha256.lower()}"
        )

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("id2label") != LABELS:
        raise RuntimeError("Mapping label config.json tidak sesuai kontrak API.")
    if int(config.get("_num_labels", len(config.get("id2label", {})))) != 4:
        raise RuntimeError("config.json harus mendefinisikan empat output classifier.")

    absa = json.loads((model_dir / "absa_config.json").read_text(encoding="utf-8"))
    if int(absa.get("max_length", -1)) != 128:
        raise RuntimeError("absa_config.json harus menggunakan max_length=128.")
    if len(absa.get("aspect_columns", [])) != 8:
        raise RuntimeError("absa_config.json harus mendefinisikan delapan aspek.")


def main() -> int:
    model_id = required_environment("MODEL_ID")
    revision = required_environment("MODEL_REVISION")
    expected_sha256 = required_environment("MODEL_SHA256")
    token = required_environment("HF_TOKEN")
    model_dir = Path(os.environ.get("MODEL_DIR", "./model")).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    headers = {"Authorization": f"Bearer {token}"}
    encoded_model_id = quote(model_id, safe="/")
    encoded_revision = quote(revision, safe="")
    with httpx.Client(headers=headers, follow_redirects=True, timeout=300.0) as client:
        for filename in REQUIRED_FILES:
            destination = model_dir / filename
            if filename == "model_int8.onnx" and destination.exists():
                if sha256(destination).lower() == expected_sha256.lower():
                    print(f"Artefak sudah valid, melewati unduhan: {filename}")
                    continue
            url = (
                f"https://huggingface.co/{encoded_model_id}/resolve/"
                f"{encoded_revision}/{quote(filename, safe='/')}"
            )
            print(f"Mengunduh {filename} dari model repository private...")
            download_file(client, url, destination)

    validate_assets(model_dir, expected_sha256)
    print(
        "Model deployment terverifikasi: "
        f"{(model_dir / 'model_int8.onnx').stat().st_size} bytes, "
        f"sha256={expected_sha256.lower()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, httpx.HTTPError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
