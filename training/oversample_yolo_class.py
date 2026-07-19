from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oversample YOLO training images that contain a target class."
    )
    parser.add_argument(
        "--dataset",
        default="data_3class",
        help="YOLO dataset root that contains train/images and train/labels.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=1,
        help="Class id to oversample. For 3-class helmet dataset, 1 means no_helmet.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=2,
        help="How many extra copies to create for each matching training image.",
    )
    return parser.parse_args()


def label_contains_class(label_path: Path, target_class: int) -> bool:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id = int(line.split()[0])
        if class_id == target_class:
            return True
    return False


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"

    if not image_dir.exists() or not label_dir.exists():
        raise SystemExit(f"Missing train/images or train/labels under {dataset_root}")

    matched = 0
    created = 0

    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        if not label_contains_class(label_path, args.target_class):
            continue

        matched += 1
        for copy_idx in range(1, args.copies + 1):
            new_image_name = f"{image_path.stem}_os{copy_idx}{image_path.suffix}"
            new_label_name = f"{image_path.stem}_os{copy_idx}.txt"
            shutil.copy2(image_path, image_dir / new_image_name)
            shutil.copy2(label_path, label_dir / new_label_name)
            created += 1

    print(f"Dataset: {dataset_root}")
    print(f"Target class: {args.target_class}")
    print(f"Matched images: {matched}")
    print(f"Created image-label copies: {created}")


if __name__ == "__main__":
    main()
