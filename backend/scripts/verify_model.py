import json
import struct
import sys
from pathlib import Path

REQUIRED = {"config.json", "tokenizer_config.json"}
WEIGHTS = {"model.safetensors", "pytorch_model.bin"}
TOKENIZER_DATA = {"tokenizer.json", "vocab.txt"}


def safetensors_classifier_size(path: Path) -> int | None:
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
    classifier = header.get("classifier.weight")
    if not classifier:
        return None
    return int(classifier["shape"][0])


def main() -> int:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
    if not model_dir.exists():
        print(f"ERROR: folder tidak ditemukan: {model_dir}")
        return 1
    names = {path.name for path in model_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED - names)
    if not names.intersection(WEIGHTS):
        missing.append("model.safetensors atau pytorch_model.bin")
    if not names.intersection(TOKENIZER_DATA):
        missing.append("tokenizer.json atau vocab.txt")
    if missing:
        print("ERROR: file model belum lengkap:")
        for name in missing:
            print(f"- {name}")
        return 1
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label", {})
    label_count = len(id2label)
    configured_count = config.get("_num_labels", config.get("num_labels", label_count))
    if configured_count != label_count:
        print(
            "ERROR: jumlah label config tidak konsisten: "
            f"num_labels={configured_count}, id2label={label_count}"
        )
        return 1

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        classifier_size = safetensors_classifier_size(safetensors_path)
        if classifier_size is not None and classifier_size != label_count:
            print(
                "ERROR: output classifier tidak sesuai mapping label: "
                f"classifier={classifier_size}, id2label={label_count}"
            )
            return 1

    print("Model directory valid")
    print("architecture:", config.get("architectures"))
    print("num_labels:", label_count)
    print("id2label:", id2label)
    if safetensors_path.exists():
        print("classifier_outputs:", safetensors_classifier_size(safetensors_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
