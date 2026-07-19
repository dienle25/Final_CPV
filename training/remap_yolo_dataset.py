from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap a YOLO dataset into a smaller class set for fine-tuning."
    )
    parser.add_argument(
        "--source",
        default="data",
        help="Source YOLO dataset root containing data.yaml, train, valid, and optional test.",
    )
    parser.add_argument(
        "--output",
        default="data_3class",
        help="Output dataset root for the remapped dataset.",
    )
    parser.add_argument(
        "--helmet-id",
        type=int,
        default=2,
        help="Original class id that represents helmet.",
    )
    parser.add_argument(
        "--no-helmet-id",
        type=int,
        default=1,
        help="Original class id that represents no_helmet.",
    )
    parser.add_argument(
        "--rider-id",
        type=int,
        default=4,
        help="Original class id that represents rider/motorcycle.",
    )
    parser.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help="Keep images even when all boxes are filtered out.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def remap_split(
    source_root: Path,
    output_root: Path,
    split: str,
    class_map: dict[int, int],
    keep_empty_labels: bool,
) -> tuple[int, int]:
    src_image_dir = source_root / split / "images"
    src_label_dir = source_root / split / "labels"
    if not src_image_dir.exists() or not src_label_dir.exists():
        return 0, 0

    dst_image_dir, dst_label_dir = ensure_dirs(output_root, split)

    kept_images = 0
    kept_boxes = 0

    for image_path in sorted(src_image_dir.glob("*")):
        if not image_path.is_file():
            continue
        label_path = src_label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        remapped_lines: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            old_class_id = int(parts[0])
            if old_class_id not in class_map:
                continue
            new_class_id = class_map[old_class_id]
            remapped_lines.append(" ".join([str(new_class_id), *parts[1:]]))
            kept_boxes += 1

        if remapped_lines or keep_empty_labels:
            shutil.copy2(image_path, dst_image_dir / image_path.name)
            (dst_label_dir / f"{image_path.stem}.txt").write_text(
                "\n".join(remapped_lines) + ("\n" if remapped_lines else ""),
                encoding="utf-8",
            )
            kept_images += 1

    return kept_images, kept_boxes


def write_data_yaml(output_root: Path) -> None:
    lines = [
        "train: train/images",
        "val: valid/images",
        "",
        "nc: 3",
        "names: ['helmet', 'no_helmet', 'rider']",
        "",
    ]
    if (output_root / "test" / "images").exists():
        lines.insert(2, "test: test/images")
    (output_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_map = {
        args.helmet_id: 0,
        args.no_helmet_id: 1,
        args.rider_id: 2,
    }

    for readme_name in ("README.dataset.txt", "README.roboflow.txt"):
        readme_path = source_root / readme_name
        if readme_path.exists():
            shutil.copy2(readme_path, output_root / readme_name)

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Class map: {class_map} -> ['helmet', 'no_helmet', 'rider']")

    for split in ("train", "valid", "test"):
        images, boxes = remap_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            class_map=class_map,
            keep_empty_labels=args.keep_empty_labels,
        )
        if images:
            print(f"{split}: images={images}, boxes={boxes}")

    write_data_yaml(output_root)
    print(f"Wrote {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
