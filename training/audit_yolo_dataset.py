from __future__ import annotations

import argparse
import ast
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_COLORS = [
    "red",
    "lime",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "white",
]


def parse_data_yaml(data_yaml: Path) -> dict:
    info: dict[str, object] = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"train", "val", "test"}:
            info[key] = value
        elif key == "nc":
            info[key] = int(value)
        elif key == "names":
            info[key] = ast.literal_eval(value)
    return info


def resolve_split_path(data_yaml: Path, split_value: str) -> Path:
    return (data_yaml.parent / split_value).resolve()


def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, x, y, w, h = line.split()
        labels.append((int(class_id), float(x), float(y), float(w), float(h)))
    return labels


def draw_samples(
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    class_names: list[str],
    output_dir: Path,
    samples_per_split: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(image_dir.glob("*"))[:samples_per_split]:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for class_id, x, y, w, h in load_labels(label_path):
            x1 = (x - w / 2) * width
            y1 = (y - h / 2) * height
            x2 = (x + w / 2) * width
            y2 = (y + h / 2) * height
            color = DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(0, y1 - 14)), f"{class_id}:{class_name}", fill=color)

        image.save(output_dir / f"{split_name}_{image_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YOLO dataset exported from Roboflow/Kaggle.")
    parser.add_argument(
        "--data-yaml",
        default="data/data.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=3,
        help="How many annotated preview images to export per split.",
    )
    parser.add_argument(
        "--output-dir",
        default="audit_samples",
        help="Directory where annotated preview images are written.",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml).resolve()
    dataset_info = parse_data_yaml(data_yaml)
    class_names = list(dataset_info.get("names", []))
    if not class_names and "nc" in dataset_info:
        class_names = [str(i) for i in range(int(dataset_info["nc"]))]

    print(f"Dataset YAML: {data_yaml}")
    print(f"Classes ({len(class_names)}): {class_names}")

    class_counts: Counter[int] = Counter()
    per_split_class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    box_sizes: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for split_name, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        split_value = dataset_info.get(yaml_key)
        if not split_value:
            print(f"\n[{split_name}] not declared in data.yaml")
            continue

        image_dir = resolve_split_path(data_yaml, str(split_value))
        label_dir = image_dir.parent / "labels"

        if not image_dir.exists() or not label_dir.exists():
            print(f"\n[{split_name}] missing on disk")
            print(f"images: {image_dir}")
            print(f"labels: {label_dir}")
            continue

        image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
        label_files = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix == ".txt"])

        box_count_per_image = []
        missing_labels = 0

        for image_path in image_files:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels += 1
                continue

            labels = load_labels(label_path)
            box_count_per_image.append(len(labels))
            for class_id, _x, _y, w, h in labels:
                class_counts[class_id] += 1
                per_split_class_counts[split_name][class_id] += 1
                box_sizes[class_id].append((w, h))

        print(f"\n[{split_name}]")
        print(f"images={len(image_files)} labels={len(label_files)} missing_labels={missing_labels}")
        if box_count_per_image:
            print(
                "avg_boxes_per_image="
                f"{statistics.mean(box_count_per_image):.2f} "
                f"min={min(box_count_per_image)} max={max(box_count_per_image)}"
            )
        print("class_counts:")
        for class_id in sorted(per_split_class_counts[split_name]):
            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            print(f"  {class_id}:{class_name} -> {per_split_class_counts[split_name][class_id]}")

        draw_samples(
            split_name=split_name,
            image_dir=image_dir,
            label_dir=label_dir,
            class_names=class_names,
            output_dir=Path(args.output_dir),
            samples_per_split=args.samples_per_split,
        )

    print("\n[overall]")
    for class_id in sorted(class_counts):
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        widths = [w for w, _h in box_sizes[class_id]]
        heights = [h for _w, h in box_sizes[class_id]]
        areas = [w * h for w, h in box_sizes[class_id]]
        print(
            f"{class_id}:{class_name} count={class_counts[class_id]} "
            f"avg_w={statistics.mean(widths):.4f} "
            f"avg_h={statistics.mean(heights):.4f} "
            f"avg_area={statistics.mean(areas):.4f}"
        )

    print(f"\nAnnotated previews written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
