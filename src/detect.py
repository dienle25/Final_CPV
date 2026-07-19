"""End-to-end helmet violation detection for video or webcam input.

Pipeline:
frame -> YOLOv8s detections -> ByteTrack rider IDs -> one-to-one head/rider
association -> temporal majority vote -> multi-frame event confirmation ->
full-frame evidence + SQLite + CSV -> optional event-driven PaddleOCR/email.

The bundled checkpoint has three classes: ``helmet``, ``no_helmet`` and
``rider``. It does not contain a license-plate detector. When OCR is explicitly
enabled, a lower-rider-region crop is used only as an experimental candidate;
``UNREAD`` is the expected fallback and must not be reported as validated plate
recognition.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from .association import (
        HELMET_ALIASES,
        NO_HELMET_ALIASES,
        RIDER_ALIASES,
        Detection,
        RiderVoteState,
        associate_heads_to_riders,
        is_alias,
        stable_label,
    )
    from .notify import EmailNotifier
    from .ocr import LicensePlateOCR
    from .violation_logger import ViolationLogger
except ImportError:  # Allows: python src/detect.py
    from association import (
        HELMET_ALIASES,
        NO_HELMET_ALIASES,
        RIDER_ALIASES,
        Detection,
        RiderVoteState,
        associate_heads_to_riders,
        is_alias,
        stable_label,
    )
    from notify import EmailNotifier
    from ocr import LicensePlateOCR
    from violation_logger import ViolationLogger


class HelmetViolationProcessor:
    """Process frames, maintain rider state and persist confirmed events."""

    def __init__(
        self,
        *,
        model_path: str = "models/best.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        tracker: str = "bytetrack.yaml",
        history_size: int = 12,
        min_votes: int = 4,
        vote_ratio: float = 0.60,
        vote_timeout_frames: int = 20,
        upper_rider_ratio: float = 0.55,
        match_padding: float = 0.08,
        min_hits: int = 3,
        max_frame_gap: int = 2,
        enable_ocr: bool = False,
        enable_email: bool = False,
        output_dir: str | Path = "outputs",
        source_name: str = "video",
    ) -> None:
        checkpoint = Path(model_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Model not found: {checkpoint}. The merged repository expects "
                "models/best.pt."
            )

        self.model = YOLO(str(checkpoint))
        self.model_path = checkpoint
        self.model_variant = self._infer_model_variant()
        self.class_names = self._normalized_model_names()
        self._validate_required_classes()

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = self._resolve_device(device)
        self.tracker = tracker
        self.history_size = max(1, int(history_size))
        self.min_votes = max(1, int(min_votes))
        self.vote_ratio = min(1.0, max(0.0, float(vote_ratio)))
        self.vote_timeout_frames = max(1, int(vote_timeout_frames))
        self.upper_rider_ratio = min(1.0, max(0.1, float(upper_rider_ratio)))
        self.match_padding = max(0.0, float(match_padding))

        self.rider_states: dict[int, RiderVoteState] = {}
        self.detection_counts: Counter[str] = Counter()
        self.stable_status_counts: Counter[str] = Counter()
        self.seen_rider_tracks: set[int] = set()
        self.confirmed_no_helmet_tracks: set[int] = set()
        self.ocr = LicensePlateOCR(enabled=enable_ocr)
        self.notifier = EmailNotifier(enabled=enable_email)

        output_root = Path(output_dir)
        self.logger = ViolationLogger(
            db_path=output_root / "db" / "violations.db",
            image_dir=output_root / "violations",
            min_hits=min_hits,
            max_frame_gap=max_frame_gap,
            source_name=source_name,
        )
        self.events: list[dict[str, Any]] = []

    def _normalized_model_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def _validate_required_classes(self) -> None:
        values = list(self.class_names.values())
        missing = []
        if not any(is_alias(name, HELMET_ALIASES) for name in values):
            missing.append("helmet")
        if not any(is_alias(name, NO_HELMET_ALIASES) for name in values):
            missing.append("no_helmet")
        if not any(is_alias(name, RIDER_ALIASES) for name in values):
            missing.append("rider")
        if missing:
            raise ValueError(
                "The model is incompatible with the merged pipeline. Missing "
                f"class roles: {missing}; model classes: {self.class_names}"
            )

    def _infer_model_variant(self) -> str:
        yaml_config = getattr(getattr(self.model, "model", None), "yaml", {}) or {}
        depth = float(yaml_config.get("depth_multiple", -1))
        width = float(yaml_config.get("width_multiple", -1))
        variants = {
            (0.33, 0.25): "YOLOv8n",
            (0.33, 0.50): "YOLOv8s",
            (0.67, 0.75): "YOLOv8m",
            (1.00, 1.00): "YOLOv8l",
            (1.00, 1.25): "YOLOv8x",
        }
        for (expected_depth, expected_width), name in variants.items():
            if abs(depth - expected_depth) < 1e-3 and abs(width - expected_width) < 1e-3:
                return name
        return "YOLO detector"

    @staticmethod
    def _resolve_device(device: str) -> str | int:
        if device != "auto":
            if str(device).isdigit() and not torch.cuda.is_available():
                print("[device] CUDA is unavailable; falling back to CPU.")
                return "cpu"
            return int(device) if str(device).isdigit() else str(device)
        return 0 if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _clip_bbox(
        box: np.ndarray | list[float] | tuple[float, ...],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        return (
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(1, min(x2, width)),
            max(1, min(y2, height)),
        )

    @staticmethod
    def _extract_plate_candidate(
        frame: np.ndarray,
        rider_bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        """Crop an experimental lower-center rider region for optional OCR.

        This is not a trained license-plate detector. It is deliberately isolated
        behind the OCR switch and documented as an experimental fallback.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = rider_bbox
        rider_width = max(1, x2 - x1)
        rider_height = max(1, y2 - y1)
        crop_x1 = max(0, int(x1 + 0.12 * rider_width))
        crop_x2 = min(width, int(x2 - 0.12 * rider_width))
        crop_y1 = max(0, int(y1 + 0.52 * rider_height))
        crop_y2 = min(height, int(y2))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        return crop if crop.size > 0 else None

    def _extract_detections(
        self,
        result: Any,
        shape: tuple[int, ...],
    ) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        height, width = shape[:2]
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids = boxes.id.detach().cpu().numpy().astype(int)
        else:
            track_ids = np.full(len(xyxy), -1, dtype=int)

        detections: list[Detection] = []
        for box, confidence, class_id, track_id in zip(
            xyxy, confidences, classes, track_ids
        ):
            detections.append(
                Detection(
                    bbox=self._clip_bbox(box, width, height),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=self.class_names.get(int(class_id), str(class_id)),
                    track_id=int(track_id),
                )
            )
        return detections

    @staticmethod
    def _color_for_status(status: str) -> tuple[int, int, int]:
        if status == "helmet":
            return (0, 190, 0)
        if status == "no_helmet":
            return (0, 0, 255)
        return (170, 170, 170)

    @staticmethod
    def _draw_box(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        text: str,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
        )
        text_width, text_height = text_size
        label_y1 = max(0, y1 - text_height - baseline - 8)
        label_x2 = min(frame.shape[1] - 1, x1 + text_width + 8)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, max(text_height + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _state_for(self, track_id: int) -> RiderVoteState:
        state = self.rider_states.get(track_id)
        if state is None:
            state = RiderVoteState(votes=deque(maxlen=self.history_size))
            self.rider_states[track_id] = state
        return state

    def _purge_stale_riders(self, frame_index: int, stale_after_frames: int = 600) -> None:
        stale = [
            track_id
            for track_id, state in self.rider_states.items()
            if frame_index - state.last_seen_frame > stale_after_frames
        ]
        for track_id in stale:
            self.rider_states.pop(track_id, None)

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        fps: float,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        result = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        detections = self._extract_detections(result, frame.shape)
        for detection in detections:
            self.detection_counts[detection.class_name] += 1

        riders = [
            detection
            for detection in detections
            if is_alias(detection.class_name, RIDER_ALIASES)
            and detection.track_id >= 0
        ]
        heads = [
            detection
            for detection in detections
            if is_alias(detection.class_name, HELMET_ALIASES)
            or is_alias(detection.class_name, NO_HELMET_ALIASES)
        ]
        assignments = associate_heads_to_riders(
            riders,
            heads,
            upper_rider_ratio=self.upper_rider_ratio,
            match_padding=self.match_padding,
        )

        annotated = frame.copy()
        new_events: list[dict[str, Any]] = []

        for rider in riders:
            self.seen_rider_tracks.add(rider.track_id)
            state = self._state_for(rider.track_id)
            state.last_seen_frame = frame_index
            matched_head = assignments.get(rider.track_id)
            current_vote = matched_head.helmet_status if matched_head is not None else None

            if current_vote is not None:
                state.votes.append(current_vote)
                state.last_vote_frame = frame_index
            elif (
                state.last_vote_frame >= 0
                and frame_index - state.last_vote_frame > self.vote_timeout_frames
            ):
                state.votes.clear()
                self.logger.reject(rider.track_id)

            status = stable_label(
                state.votes,
                min_votes=self.min_votes,
                min_ratio=self.vote_ratio,
            )
            if state.last_vote_frame < 0 or (
                frame_index - state.last_vote_frame > self.vote_timeout_frames
            ):
                status = "unknown"

            self.stable_status_counts[status] += 1
            rider_color = self._color_for_status(status)
            self._draw_box(
                annotated,
                rider.bbox,
                f"Rider ID {rider.track_id} | {status} | {rider.confidence:.2f}",
                rider_color,
                thickness=3 if status == "no_helmet" else 2,
            )

            if current_vote == "helmet":
                self.logger.reject(rider.track_id)
            elif current_vote == "no_helmet" and status == "no_helmet":
                plate_crop = None
                plate_source = ""
                if self.ocr.enabled:
                    plate_crop = self._extract_plate_candidate(frame, rider.bbox)
                    plate_source = "heuristic_lower_rider_region"

                event = self.logger.observe(
                    track_id=rider.track_id,
                    frame_index=frame_index,
                    video_time_s=frame_index / max(fps, 1.0),
                    confidence=matched_head.confidence if matched_head else 0.0,
                    rider_bbox=rider.bbox,
                    head_bbox=matched_head.bbox if matched_head else None,
                    frame=frame,
                    plate_crop=plate_crop,
                    plate_source=plate_source,
                    ocr_reader=self.ocr.read if self.ocr.enabled else None,
                )
                if event is not None:
                    self.confirmed_no_helmet_tracks.add(rider.track_id)
                    self.events.append(event)
                    new_events.append(event)
                    if self.notifier.enabled:
                        self.notifier.send_violation(event)

        for head in heads:
            status = head.helmet_status or "unknown"
            self._draw_box(
                annotated,
                head.bbox,
                f"{status} {head.confidence:.2f}",
                self._color_for_status(status),
            )

        cv2.putText(
            annotated,
            f"Confirmed violations: {self.logger.count()}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Tracked riders: {len(riders)}",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if frame_index % 300 == 0:
            self.logger.purge_stale(frame_index)
            self._purge_stale_riders(frame_index)

        return annotated, new_events

    def process_source(
        self,
        source: str | int,
        *,
        output_video: str | Path = "outputs/videos/result.mp4",
        display: bool = False,
        frame_callback: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
        callback_every: int = 3,
    ) -> dict[str, Any]:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 1:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_video}")

        processed_frames = 0
        started = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, new_events = self.process_frame(
                    frame,
                    frame_index=processed_frames,
                    fps=fps,
                )
                writer.write(annotated)

                elapsed = max(time.perf_counter() - started, 1e-6)
                stats = {
                    "frame_index": processed_frames,
                    "total_frames": total_frames,
                    "processing_fps": (processed_frames + 1) / elapsed,
                    "violation_count": self.logger.count(),
                    "active_rider_count": sum(
                        1
                        for state in self.rider_states.values()
                        if state.last_seen_frame == processed_frames
                    ),
                    "new_events": new_events,
                }
                if (
                    frame_callback is not None
                    and processed_frames % max(1, callback_every) == 0
                ):
                    frame_callback(annotated, stats)

                if display:
                    cv2.imshow(
                        "Motorcycle Helmet Violation Detection - press Q to stop",
                        annotated,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        processed_frames += 1
                        break
                processed_frames += 1
        finally:
            capture.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        output_root = output_video.parent.parent
        csv_path = self.logger.export_csv(output_root / "violations.csv")
        elapsed = max(time.perf_counter() - started, 1e-6)
        summary = {
            "output_video": str(output_video),
            "csv_path": str(csv_path),
            "db_path": str(self.logger.db_path),
            "processed_frames": processed_frames,
            "average_processing_fps": round(processed_frames / elapsed, 2),
            "violation_count": self.logger.count(),
            "detection_counts": dict(self.detection_counts),
            "unique_rider_tracks": len(self.seen_rider_tracks),
            "confirmed_no_helmet_tracks": len(self.confirmed_no_helmet_tracks),
            "model_path": str(self.model_path),
            "model_variant": self.model_variant,
            "model_classes": self.class_names,
            "device": str(self.device),
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "tracker": self.tracker,
            "history_size": self.history_size,
            "min_votes": self.min_votes,
            "vote_ratio": self.vote_ratio,
            "event_min_hits": self.logger.min_hits,
            "ocr_enabled": self.ocr.enabled,
            "ocr_plate_source": (
                "heuristic_lower_rider_region" if self.ocr.enabled else "disabled"
            ),
        }
        summary_path = output_root / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        return summary

    def close(self) -> None:
        self.logger.close()


def parse_source(value: str) -> str | int:
    """Accept a local video path or a webcam index; network streams are excluded."""
    return int(value) if value.lstrip("-").isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8s + ByteTrack motorcycle helmet violation detection"
    )
    parser.add_argument(
        "--source",
        default="data/demo.mp4",
        help="Local video path or webcam index, for example 0",
    )
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--output", default="outputs/videos/result.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--vote-ratio", type=float, default=0.60)
    parser.add_argument("--vote-timeout", type=int, default=20)
    parser.add_argument("--upper-rider-ratio", type=float, default=0.55)
    parser.add_argument("--match-padding", type=float, default=0.08)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-frame-gap", type=int, default=2)
    parser.add_argument("--device", default="auto")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument(
        "--ocr",
        action="store_true",
        help="Enable experimental event-driven OCR on a heuristic rider crop",
    )
    ocr_group.add_argument(
        "--no-ocr",
        action="store_true",
        help="Compatibility flag; OCR is disabled by default",
    )
    parser.add_argument("--email", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    processor = HelmetViolationProcessor(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        history_size=args.history,
        min_votes=args.min_votes,
        vote_ratio=args.vote_ratio,
        vote_timeout_frames=args.vote_timeout,
        upper_rider_ratio=args.upper_rider_ratio,
        match_padding=args.match_padding,
        min_hits=args.min_hits,
        max_frame_gap=args.max_frame_gap,
        enable_ocr=bool(args.ocr and not args.no_ocr),
        enable_email=args.email,
        output_dir=(
            Path(args.output).parent.parent
            if Path(args.output).parent.name == "videos"
            else Path(args.output).parent
        ),
        source_name=str(args.source),
    )
    try:
        summary = processor.process_source(
            parse_source(args.source),
            output_video=args.output,
            display=args.show,
        )
        print("\n=== Run summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
    finally:
        processor.close()


if __name__ == "__main__":
    main()
