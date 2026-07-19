"""Tạo ZIP bàn giao sạch, không kèm môi trường/cache/lịch sử cá nhân."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "helmet_classroom_demo"
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "runs",
    "work",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}
EXCLUDED_FILENAMES = {".env", "secrets.toml"}


def should_include(path: Path, *, excluded_paths: set[Path] | None = None) -> bool:
    relative = path.relative_to(ROOT)
    if excluded_paths and path.resolve() in excluded_paths:
        return False
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.name.lower() in EXCLUDED_FILENAMES or path.name.lower().startswith(".env."):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if (
        len(relative.parts) >= 2
        and relative.parts[0].lower() == "data"
        and relative.parts[1].lower().startswith("demo.")
    ):
        return False
    if relative.parts and relative.parts[0] == "outputs":
        return path.name == ".gitkeep"
    return True


def iter_files(*, excluded_paths: set[Path] | None = None) -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and should_include(path, excluded_paths=excluded_paths)
        ),
        key=lambda path: path.relative_to(ROOT).as_posix().lower(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_zip(destination: Path) -> tuple[int, int, str]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    files = iter_files(
        excluded_paths={destination.resolve(), temporary.resolve(), sidecar.resolve()}
    )
    total_bytes = sum(path.stat().st_size for path in files)

    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in files:
            archive.write(path, f"{PROJECT_NAME}/{path.relative_to(ROOT).as_posix()}")

    os.replace(temporary, destination)
    digest = sha256_file(destination)
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    return len(files), total_bytes, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=ROOT / "outputs" / "helmet_classroom_demo.zip",
    )
    args = parser.parse_args()
    count, total_bytes, digest = build_zip(args.destination)
    print(f"Đã đóng gói {count} tệp ({total_bytes / 1024 / 1024:.1f} MiB).")
    print(f"ZIP: {args.destination.resolve()}")
    print(f"SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
