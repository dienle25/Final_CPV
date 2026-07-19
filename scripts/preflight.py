"""Kiem tra nhanh moi truong truoc khi mo demo lop hoc."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_FILES = {
    "Mô hình mũ": ROOT / "models" / "best.onnx",
    "Mô hình người": ROOT / "models" / "person" / "object_detection_nanodet_2022nov.onnx",
    "YuNet": ROOT / "models" / "face" / "face_detection_yunet_2023mar.onnx",
    "SFace": ROOT / "models" / "face" / "face_recognition_sface_2021dec.onnx",
}
PACKAGES = {
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "onnxruntime": "ONNX Runtime",
    "streamlit": "Streamlit",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checks() -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        errors.append("Cần Python 3.11 hoặc 3.12 (64-bit).")
    if sys.maxsize <= 2**32:
        errors.append("Cần Python 64-bit.")

    versions: dict[str, str] = {}
    for module_name, display_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            versions[display_name] = str(getattr(module, "__version__", "OK"))
        except Exception as exc:  # pragma: no cover - thong bao chan doan
            errors.append(f"Thiếu hoặc lỗi {display_name}: {exc}")
    details["packages"] = versions

    model_sizes: dict[str, int] = {}
    for display_name, path in MODEL_FILES.items():
        if not path.is_file():
            errors.append(f"Thiếu {display_name}: {path.relative_to(ROOT)}")
        elif path.stat().st_size < 100_000:
            errors.append(f"{display_name} có kích thước bất thường: {path.relative_to(ROOT)}")
        else:
            model_sizes[display_name] = path.stat().st_size
    details["models_bytes"] = model_sizes

    manifest_path = ROOT / "models" / "MODEL_MANIFEST.json"
    verified_hashes: dict[str, str] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("models", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError("danh sách models trống")
        for entry in entries:
            relative_path = str(entry["path"])
            model_path = ROOT / relative_path
            if not model_path.is_file():
                errors.append(f"Manifest trỏ tới tệp bị thiếu: {relative_path}")
                continue
            expected_size = int(entry["bytes"])
            actual_size = model_path.stat().st_size
            if actual_size != expected_size:
                errors.append(
                    f"Sai kích thước model {relative_path}: {actual_size} != {expected_size}"
                )
                continue
            expected_hash = str(entry["sha256"]).lower()
            actual_hash = sha256_file(model_path)
            if actual_hash != expected_hash:
                errors.append(f"Sai SHA256 model: {relative_path}")
                continue
            verified_hashes[relative_path] = actual_hash
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"MODEL_MANIFEST.json không hợp lệ: {exc}")
    details["verified_model_sha256"] = verified_hashes

    roster = ROOT / "data" / "students.csv"
    student_root = ROOT / "data" / "students"
    roster_ids: set[str] = set()
    active_roster_ids: set[str] = set()
    if not roster.is_file():
        errors.append("Thiếu data/students.csv.")
    else:
        try:
            with roster.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"student_id", "full_name", "active"}
                if not required.issubset(set(reader.fieldnames or [])):
                    errors.append("data/students.csv thiếu cột student_id, full_name hoặc active.")
                else:
                    for line_number, row in enumerate(reader, start=2):
                        student_id = str(row.get("student_id", "")).strip()
                        full_name = str(row.get("full_name", "")).strip()
                        if not student_id or not full_name:
                            errors.append(f"Roster thiếu MSSV/họ tên tại dòng {line_number}.")
                            continue
                        if student_id in roster_ids:
                            errors.append(f"Roster trùng MSSV: {student_id}")
                        roster_ids.add(student_id)
                        if str(row.get("active", "1")).strip().casefold() not in {
                            "0",
                            "false",
                            "no",
                        }:
                            active_roster_ids.add(student_id)
        except (OSError, csv.Error) as exc:
            errors.append(f"Không đọc được data/students.csv: {exc}")
    reference_counts: dict[str, int] = {}
    if student_root.is_dir():
        for folder in sorted(student_root.iterdir()):
            if folder.is_dir():
                count = sum(
                    1
                    for item in folder.iterdir()
                    if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                )
                reference_counts[folder.name] = count
                if count < 5:
                    warnings.append(
                        f"{folder.name} mới có {count} ảnh; nên đăng ký 6-15 góc mặt trước khi demo."
                    )
    else:
        errors.append("Thiếu thư mục data/students.")
    missing_reference_folders = sorted(active_roster_ids - set(reference_counts))
    if missing_reference_folders:
        errors.append(
            "Sinh viên active chưa có thư mục ảnh: " + ", ".join(missing_reference_folders)
        )
    orphan_reference_folders = sorted(set(reference_counts) - roster_ids)
    if orphan_reference_folders:
        warnings.append(
            "Thư mục ảnh không có trong roster: " + ", ".join(orphan_reference_folders)
        )
    details["student_reference_counts"] = reference_counts

    output_dir = ROOT / "outputs" / "classroom"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        errors.append(f"Không ghi được outputs/classroom: {exc}")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        details["onnx_providers"] = providers
        model_io: dict[str, dict[str, int]] = {}
        for display_name, path in MODEL_FILES.items():
            if not path.is_file():
                continue
            try:
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3
                session = ort.InferenceSession(
                    str(path),
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                model_io[display_name] = {
                    "inputs": len(session.get_inputs()),
                    "outputs": len(session.get_outputs()),
                }
            except Exception as exc:
                errors.append(f"Không nạp được {display_name} bằng ONNX Runtime: {exc}")
        details["onnx_model_io"] = model_io
        session = None
        if platform.system() == "Windows" and "DmlExecutionProvider" not in providers:
            warnings.append("Không thấy DirectML; demo sẽ chạy CPU và chậm hơn.")
    except Exception:
        pass

    smoke_inference: dict[str, object] = {}
    try:
        import cv2
        import numpy as np

        from src.classroom_demo.detectors import (
            NanoDetPersonDetector,
            YoloHelmetOnnxDetector,
            YuNetFaceDetector,
        )
        from src.classroom_demo.face_recognition import SFaceEncoder

        reference_path = next(
            path
            for path in sorted((ROOT / "data" / "students").glob("*/*"))
            if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        frame = cv2.imread(str(reference_path))
        if frame is None:
            raise RuntimeError(f"không đọc được ảnh smoke test: {reference_path}")
        cpu_provider = ("CPUExecutionProvider",)
        face_detector = YuNetFaceDetector(
            MODEL_FILES["YuNet"],
            score_threshold=0.80,
        )
        faces = face_detector.detect(frame)
        if not faces:
            raise RuntimeError("YuNet không tìm thấy mặt trong ảnh smoke test")
        face_encoder = SFaceEncoder(MODEL_FILES["SFace"])
        embedding = face_encoder.extract(frame, max(faces, key=lambda face: face.score))
        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
            raise RuntimeError("SFace trả embedding không hợp lệ")

        helmet_detections = YoloHelmetOnnxDetector(
            MODEL_FILES["Mô hình mũ"],
            providers=cpu_provider,
        ).detect(frame)
        person_detections = NanoDetPersonDetector(
            MODEL_FILES["Mô hình người"],
            providers=cpu_provider,
        ).detect(frame)
        smoke_inference = {
            "reference_image": str(reference_path.relative_to(ROOT)),
            "face_count": len(faces),
            "embedding_dimensions": int(embedding.size),
            "helmet_detection_count": len(helmet_detections),
            "person_detection_count": len(person_detections),
        }
    except (ImportError, OSError, RuntimeError, StopIteration, ValueError) as exc:
        errors.append(f"Smoke inference thất bại: {exc}")
    details["smoke_inference"] = smoke_inference

    return errors, warnings, details


def main() -> int:
    errors, warnings, details = run_checks()
    print(json.dumps(details, ensure_ascii=False, indent=2))
    for warning in warnings:
        print(f"[CẢNH BÁO] {warning}")
    for error in errors:
        print(f"[LỖI] {error}")
    if errors:
        print(f"Preflight thất bại: {len(errors)} lỗi, {len(warnings)} cảnh báo.")
        return 1
    print(f"Preflight đạt: 0 lỗi, {len(warnings)} cảnh báo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
