"""License-plate OCR wrapper with graceful degradation.

Why PaddleOCR:
- It combines text detection and recognition and works well for irregular scene
  text without requiring a custom OCR model for this two-day MVP.
- OCR is initialized lazily and called only after a violation is confirmed.
  This design protects the 8 GB RAM machine and keeps normal video frames fast.

The wrapper supports PaddleOCR 3.x and keeps a fallback parser for the older 2.x
API because student Windows environments often contain mixed package versions.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class OCRResult:
    text: str = "UNREAD"
    confidence: float = 0.0


class LicensePlateOCR:
    def __init__(self, enabled: bool = True, min_confidence: float = 0.25) -> None:
        self.enabled = enabled
        self.min_confidence = min_confidence
        self._engine: Any | None = None
        self._init_error: str | None = None

    def _initialize(self) -> None:
        if self._engine is not None or self._init_error is not None or not self.enabled:
            return
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 3.x parameters. CPU OCR is selected intentionally on
            # Windows so YOLO can keep the GTX 1650 VRAM for detection.
            try:
                self._engine = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
            except TypeError:
                # Compatibility with PaddleOCR 2.x.
                self._engine = PaddleOCR(
                    lang="en",
                    use_angle_cls=False,
                    use_gpu=False,
                    show_log=False,
                )
        except Exception as exc:  # OCR is optional; detection must continue.
            self._init_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "PaddleOCR could not be initialized. OCR will return UNREAD. "
                f"Reason: {self._init_error}",
                RuntimeWarning,
            )

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Upscale and enhance a small plate crop before OCR.

        License plates are often only a few dozen pixels high. Upscaling,
        denoising and local contrast enhancement improve character separation.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty license-plate crop")

        h, w = image.shape[:2]
        scale = max(2.0, min(4.0, 160.0 / max(h, 1)))
        enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def normalize_plate(text: str) -> str:
        """Normalize OCR output while preserving a readable VN-like plate form."""
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        # Do not invent characters. Formatting is applied only when enough
        # characters exist; otherwise keep the raw normalized candidate.
        if 7 <= len(cleaned) <= 10:
            return cleaned
        return cleaned or "UNREAD"

    @staticmethod
    def _to_plain_dict(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: LicensePlateOCR._to_plain_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [LicensePlateOCR._to_plain_dict(v) for v in obj]
        for attr in ("json", "res"):
            value = getattr(obj, attr, None)
            if value is not None:
                try:
                    return LicensePlateOCR._to_plain_dict(value() if callable(value) else value)
                except Exception:
                    pass
        return obj

    @classmethod
    def _collect_candidates(cls, obj: Any) -> list[tuple[str, float]]:
        """Recursively extract text/score pairs from PaddleOCR 2.x or 3.x output."""
        candidates: list[tuple[str, float]] = []
        plain = cls._to_plain_dict(obj)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                # PaddleOCR 3.x recognition output.
                if "rec_text" in node:
                    candidates.append((str(node["rec_text"]), float(node.get("rec_score", 0.0))))
                texts = node.get("rec_texts")
                scores = node.get("rec_scores")
                if isinstance(texts, (list, tuple)):
                    score_list = scores if isinstance(scores, (list, tuple)) else [0.0] * len(texts)
                    for text, score in zip(texts, score_list):
                        candidates.append((str(text), float(score)))
                for value in node.values():
                    walk(value)
            elif isinstance(node, (list, tuple)):
                # PaddleOCR 2.x line: [polygon, (text, score)]
                if (
                    len(node) == 2
                    and isinstance(node[1], (list, tuple))
                    and len(node[1]) >= 2
                    and isinstance(node[1][0], str)
                ):
                    candidates.append((node[1][0], float(node[1][1])))
                else:
                    for value in node:
                        walk(value)

        walk(plain)
        return candidates

    def read(self, plate_crop: np.ndarray | None) -> OCRResult:
        if not self.enabled or plate_crop is None or plate_crop.size == 0:
            return OCRResult()

        self._initialize()
        if self._engine is None:
            return OCRResult()

        prepared = self.preprocess(plate_crop)
        try:
            if hasattr(self._engine, "predict"):
                raw = list(self._engine.predict(prepared))
            else:
                raw = self._engine.ocr(prepared, cls=False)
        except Exception as exc:
            warnings.warn(f"OCR inference failed: {exc}", RuntimeWarning)
            return OCRResult()

        candidates = self._collect_candidates(raw)
        candidates = [
            (self.normalize_plate(text), score)
            for text, score in candidates
            if score >= self.min_confidence
        ]
        candidates = [(text, score) for text, score in candidates if text != "UNREAD"]
        if not candidates:
            return OCRResult()

        # Prefer confidence, then a plausible 7–10 character plate length.
        candidates.sort(
            key=lambda item: (item[1] + (0.15 if 7 <= len(item[0]) <= 10 else 0.0)),
            reverse=True,
        )
        text, score = candidates[0]
        return OCRResult(text=text, confidence=float(score))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaddleOCR on one plate crop")
    parser.add_argument("image")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    print(LicensePlateOCR(enabled=True).read(image))
