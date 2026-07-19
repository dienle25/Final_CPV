"""Fine-tune the three-class YOLO helmet detector.

The bundled checkpoint is a YOLOv8s model trained in two stages on the remapped
``helmet``, ``no_helmet`` and ``rider`` dataset. This script provides a local,
reproducible training entry point; it does not claim to reproduce the original
Kaggle run bit-for-bit because the original runtime and package versions are not
fully recorded.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-class YOLOv8 helmet detector")
    parser.add_argument("--data", default="data/helmet_3class/data.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a GPU index such as '0'")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="helmet_yolov8s_3class")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(requested: str) -> str | int:
    if requested != "auto":
        if requested.isdigit() and not torch.cuda.is_available():
            print("[device] CUDA is unavailable; falling back to CPU.")
            return "cpu"
        return int(requested) if requested.isdigit() else requested
    return 0 if torch.cuda.is_available() else "cpu"


def validate_dataset_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("train", "val", "names"):
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {path}")
    names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])
    if names != ["helmet", "no_helmet", "rider"]:
        raise ValueError(
            "Expected class order ['helmet', 'no_helmet', 'rider']; "
            f"received {names}. Class order must match the bundled checkpoint."
        )
    return data


def _dataset_root(data_path: Path, data_config: dict[str, Any]) -> Path:
    configured_root = data_config.get("path")
    if configured_root is None:
        return data_path.parent.resolve()
    root = Path(str(configured_root))
    if not root.is_absolute():
        root = data_path.parent / root
    return root.resolve()


def _images_from_list(list_path: Path, dataset_root: Path) -> set[Path]:
    images: set[Path] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            local_candidate = list_path.parent / candidate
            candidate = local_candidate if local_candidate.exists() else dataset_root / candidate
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            images.add(candidate)
    return images


def count_split_images(
    data_path: Path,
    data_config: dict[str, Any],
    split: str,
) -> int | None:
    value = data_config.get(split)
    if value is None:
        return None

    entries = value if isinstance(value, (list, tuple)) else [value]
    dataset_root = _dataset_root(data_path, data_config)
    images: set[Path] = set()
    for entry in entries:
        candidate = Path(str(entry))
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        matches = [Path(item) for item in glob.glob(str(candidate), recursive=True)] or [candidate]
        for match in matches:
            match = match.resolve()
            if match.is_dir():
                images.update(
                    file.resolve()
                    for file in match.rglob("*")
                    if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
                )
            elif match.is_file() and match.suffix.lower() in IMAGE_SUFFIXES:
                images.add(match)
            elif match.is_file() and match.suffix.lower() == ".txt":
                images.update(_images_from_list(match, dataset_root))
    return len(images)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    data_config = validate_dataset_yaml(data_path)
    device = resolve_device(args.device)

    print("=== Three-class helmet detector training ===")
    print(f"Dataset : {data_path.resolve()}")
    print(f"Classes : {data_config['names']}")
    print(f"Model   : {args.model}")
    print(f"Device  : {device}")
    for split in ("train", "val", "test"):
        count = count_split_images(data_path, data_config, split)
        print(f"{split:>5} images: {count if count is not None else 'not defined'}")

    model = YOLO(args.model)
    try:
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            project=args.project,
            name=args.name,
            pretrained=True,
            optimizer="AdamW",
            cache=False,
            amp=True,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            cos_lr=True,
            close_mosaic=8,
            translate=0.04,
            scale=0.20,
            fliplr=0.5,
            mosaic=0.15,
            mixup=0.0,
            plots=True,
            save=True,
            exist_ok=True,
            verbose=True,
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            raise RuntimeError(
                "Training ran out of memory. Retry with --batch 2 --imgsz 512 --workers 0."
            ) from exc
        raise

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    evaluation_model = YOLO(str(best_path if best_path.exists() else args.model))
    split = "test" if data_config.get("test") else "val"
    metrics = evaluation_model.val(
        data=str(data_path),
        split=split,
        imgsz=args.imgsz,
        batch=max(1, min(args.batch, 4)),
        device=device,
        plots=True,
    )

    summary: dict[str, Any] = {"split": split}
    for key, value in getattr(metrics, "results_dict", {}).items():
        try:
            summary[str(key)] = float(value)
        except (TypeError, ValueError):
            summary[str(key)] = str(value)

    output = Path(args.project) / args.name / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best weights: {best_path}")
    print(f"Metrics JSON: {output}")


if __name__ == "__main__":
    main()
