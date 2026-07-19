"""Evaluate a YOLO checkpoint on the included test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the helmet model on the test split")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="data/helmet_3class/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="metrics/test_metrics_summary.generated.json")
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--name", default="test_split")
    return parser.parse_args()


def resolve_device(requested: str) -> str | int:
    if requested == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if requested.isdigit() and not torch.cuda.is_available():
        print("[device] CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return int(requested) if requested.isdigit() else requested


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model = YOLO(str(model_path))
    device = resolve_device(args.device)
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True,
    )

    box = metrics.box
    names = model.names
    class_metrics = []
    for class_id, class_name in names.items():
        class_metrics.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name),
                "map50": float(box.ap50[int(class_id)]),
                "map50_95": float(box.ap[int(class_id)]),
            }
        )

    summary = {
        "model": "models/best.pt",
        "architecture": "YOLOv8s",
        "data": "data/helmet_3class/data.yaml",
        "split": "test",
        "imgsz": args.imgsz,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "class_metrics": class_metrics,
        "save_dir": str(metrics.save_dir),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
