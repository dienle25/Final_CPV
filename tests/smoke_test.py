"""Fast checks that do not require loading the YOLO checkpoint or PaddleOCR."""

from __future__ import annotations

import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.association import (  # noqa: E402
    Detection,
    associate_heads_to_riders,
    is_alias,
    normalize_label,
    stable_label,
    NO_HELMET_ALIASES,
)
from src.ocr import LicensePlateOCR  # noqa: E402
from src.violation_logger import ViolationLogger  # noqa: E402


def test_association_logic() -> None:
    assert normalize_label(" No Helmet ") == "nohelmet"
    assert normalize_label("NO_HELMET") == "no_helmet"
    assert is_alias("withoutHelmet", NO_HELMET_ALIASES)

    rider_left = Detection((20, 20, 180, 230), 0.90, 2, "rider", 7)
    rider_right = Detection((140, 20, 300, 230), 0.88, 2, "rider", 8)
    head_left = Detection((65, 35, 115, 95), 0.92, 1, "no_helmet")
    head_right = Detection((205, 35, 255, 95), 0.89, 0, "helmet")

    matches = associate_heads_to_riders(
        [rider_left, rider_right],
        [head_left, head_right],
    )
    assert matches[7] is head_left
    assert matches[8] is head_right
    assert len({id(value) for value in matches.values()}) == 2

    votes = deque(["no_helmet", "no_helmet", "helmet", "no_helmet"], maxlen=12)
    assert stable_label(votes, min_votes=3, min_ratio=0.60) == "no_helmet"
    assert stable_label(votes, min_votes=4, min_ratio=0.60) == "unknown"


def test_logger() -> None:
    temporary = ROOT / "outputs" / "_smoke_test"
    if temporary.exists():
        shutil.rmtree(temporary)

    logger = ViolationLogger(
        db_path=temporary / "violations.db",
        image_dir=temporary / "images",
        min_hits=3,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    plate_candidate = np.zeros((40, 120, 3), dtype=np.uint8)

    common = {
        "track_id": 7,
        "video_time_s": 0.0,
        "confidence": 0.85,
        "rider_bbox": (60, 20, 180, 220),
        "head_bbox": (90, 30, 135, 90),
        "frame": frame,
        "plate_crop": plate_candidate,
        "plate_source": "test_candidate",
    }
    assert logger.observe(frame_index=0, **common) is None
    assert logger.observe(frame_index=1, **common) is None
    event = logger.observe(frame_index=2, **common)
    assert event is not None
    assert event["violation_type"] == "no_helmet"
    assert Path(event["image_path"]).exists()
    assert Path(event["plate_image_path"]).exists()
    assert logger.count() == 1

    logger.reject(9)
    assert LicensePlateOCR.normalize_plate("59-a1 123.45") == "59A112345"
    logger.close()
    shutil.rmtree(temporary)


def main() -> None:
    test_association_logic()
    test_logger()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
