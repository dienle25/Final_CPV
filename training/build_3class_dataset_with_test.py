from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3-class YOLO dataset with a separate test split."
    )
    parser.add_argument(
        "--source",
        default="data",
        help="Source YOLO dataset root containing train/valid labels and images.",
    )
    parser.add_argument(
        "--output",
        default="data_3class_with_test",
        help="Output dataset root.",
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
        help="Original class id that represents rider.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of remapped train images to move into test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def remap_label_lines(label_path: Path, class_map: dict[int, int]) -> list[str]:
    remapped: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        old_class_id = int(parts[0])
        if old_class_id not in class_map:
            continue
        new_class_id = class_map[old_class_id]
        remapped.append(" ".join([str(new_class_id), *parts[1:]]))
    return remapped


def label_signature(remapped_lines: list[str]) -> str:
    class_ids = sorted({int(line.split()[0]) for line in remapped_lines})
    return "+".join(str(class_id) for class_id in class_ids)


def write_split(
    entries: list[tuple[Path, list[str]]],
    output_root: Path,
    split: str,
) -> tuple[int, int]:
    image_dir, label_dir = ensure_dirs(output_root, split)
    image_count = 0
    box_count = 0
    for image_path, remapped_lines in entries:
        shutil.copy2(image_path, image_dir / image_path.name)
        (label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(remapped_lines) + "\n",
            encoding="utf-8",
        )
        image_count += 1
        box_count += len(remapped_lines)
    return image_count, box_count


def split_train_entries(
    train_entries: list[tuple[Path, list[str]]],
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, list[str]]], list[tuple[Path, list[str]]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[Path, list[str]]]] = defaultdict(list)
    for entry in train_entries:
        grouped[label_signature(entry[1])].append(entry)

    train_out: list[tuple[Path, list[str]]] = []
    test_out: list[tuple[Path, list[str]]] = []

    for signature, items in sorted(grouped.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_ratio)) if len(shuffled) > 1 else 0
        test_subset = shuffled[:test_count]
        train_subset = shuffled[test_count:]
        if not train_subset and test_subset:
            train_subset.append(test_subset.pop())
        train_out.extend(train_subset)
        test_out.extend(test_subset)
        print(
            f"group {signature or 'empty'}: total={len(items)} "
            f"train={len(train_subset)} test={len(test_subset)}"
        )

    rng.shuffle(train_out)
    rng.shuffle(test_out)
    return train_out, test_out


def write_data_yaml(output_root: Path) -> None:
    lines = [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "nc: 3",
        "names: ['helmet', 'no_helmet', 'rider']",
        "",
    ]
    (output_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def collect_entries(
    source_root: Path,
    split: str,
    class_map: dict[int, int],
) -> list[tuple[Path, list[str]]]:
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    if not image_dir.exists() or not label_dir.exists():
        return []

    entries: list[tuple[Path, list[str]]] = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        remapped = remap_label_lines(label_path, class_map)
        if remapped:
            entries.append((image_path, remapped))
    return entries


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

    train_entries = collect_entries(source_root, "train", class_map)
    valid_entries = collect_entries(source_root, "valid", class_map)

    split_train, split_test = split_train_entries(
        train_entries=train_entries,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_images, train_boxes = write_split(split_train, output_root, "train")
    valid_images, valid_boxes = write_split(valid_entries, output_root, "valid")
    test_images, test_boxes = write_split(split_test, output_root, "test")
    write_data_yaml(output_root)

    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Class map: {class_map} -> ['helmet', 'no_helmet', 'rider']")
    print(f"train: images={train_images}, boxes={train_boxes}")
    print(f"valid: images={valid_images}, boxes={valid_boxes}")
    print(f"test: images={test_images}, boxes={test_boxes}")
    print(f"Wrote {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
