"""Detector adapters used by the classroom demo.

The helmet checkpoint is exported as a raw YOLOv8 ONNX tensor with shape
``(1, 7, 8400)``: four ``cxcywh`` values followed by three class scores.  This
module intentionally does not depend on Ultralytics at runtime.

OpenCV and ONNX Runtime are imported lazily.  Geometry/decoder unit tests can
therefore run on a machine that has neither package installed.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .geometry import BBox, area, as_bbox, clip_box, non_max_suppression


DEFAULT_HELMET_CLASSES: dict[int, str] = {
    0: "helmet",
    1: "no_helmet",
    2: "rider",
}


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in original-frame coordinates."""

    bbox: BBox
    score: float
    class_id: int
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", as_bbox(self.bbox))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "class_id", int(self.class_id))
        if not self.label:
            raise ValueError("Detection label cannot be empty")

    @property
    def int_bbox(self) -> tuple[int, int, int, int]:
        return tuple(int(round(value)) for value in self.bbox)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """YuNet face box and its five landmarks."""

    bbox: BBox
    score: float
    landmarks: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox", as_bbox(self.bbox))
        object.__setattr__(self, "score", float(self.score))
        if len(self.landmarks) != 5:
            raise ValueError("YuNet face detections must contain five landmarks")

    def to_yunet_row(self) -> np.ndarray:
        """Return OpenCV's 15-value YuNet row for SFace alignment."""

        x1, y1, x2, y2 = self.bbox
        values: list[float] = [x1, y1, x2 - x1, y2 - y1]
        for point_x, point_y in self.landmarks:
            values.extend((float(point_x), float(point_y)))
        values.append(float(self.score))
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Parameters needed to map ONNX input boxes back to a source frame."""

    scale: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int
    input_width: int
    input_height: int

    def restore(self, box: Sequence[float]) -> BBox:
        if self.scale <= 0.0:
            raise ValueError("Letterbox scale must be positive")
        x1, y1, x2, y2 = as_bbox(box)
        restored = (
            (x1 - self.pad_x) / self.scale,
            (y1 - self.pad_y) / self.scale,
            (x2 - self.pad_x) / self.scale,
            (y2 - self.pad_y) / self.scale,
        )
        return clip_box(restored, self.source_width, self.source_height)


class _OnnxSession(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Any, input_feed: Mapping[str, np.ndarray]) -> list[Any]: ...


def create_ort_session_options(ort_module: Any) -> Any:
    """Create options that are safe for ONNX Runtime DirectML sessions.

    DirectML requires sequential graph execution and disabled memory patterns.
    A per-session inference lock in :class:`YoloHelmetOnnxDetector` completes
    the protection when several UI/worker threads share one detector.
    """

    options = ort_module.SessionOptions()
    options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    if hasattr(options, "inter_op_num_threads"):
        options.inter_op_num_threads = 1
    if hasattr(options, "log_severity_level"):
        # NanoDet contains many initializers that ORT reports as graph inputs.
        # They are harmless; keep the live demo console focused on real errors.
        options.log_severity_level = 3
    return options


def _default_ort_providers(ort_module: Any) -> list[str]:
    available = list(ort_module.get_available_providers())
    preferred = (
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    selected = [provider for provider in preferred if provider in available]
    if not selected:
        raise RuntimeError(
            "ONNX Runtime reports no usable execution provider. "
            f"Available providers: {available}"
        )
    return selected


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize spatially with OpenCV, falling back to Pillow for tests."""

    target_width, target_height = size
    try:
        import cv2  # type: ignore

        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
    except ImportError:
        from PIL import Image

        resized = Image.fromarray(image).resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )
        return np.asarray(resized)


class YoloHelmetOnnxDetector:
    """ONNX Runtime wrapper for ``models/best.onnx``.

    ``session`` is injectable for deterministic tests.  Production callers
    normally leave it unset and optionally choose execution ``providers``.
    """

    def __init__(
        self,
        model_path: str | Path = "models/best.onnx",
        *,
        class_names: Mapping[int, str] | None = None,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        providers: Sequence[str] | None = None,
        session: _OnnxSession | None = None,
        ort_module: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.class_names = dict(class_names or DEFAULT_HELMET_CLASSES)
        if sorted(self.class_names) != list(range(len(self.class_names))):
            raise ValueError("class_names keys must be contiguous from zero")
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("ONNX input size must be positive")
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max(1, int(max_detections))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in [0, 1]")

        self._run_lock = threading.Lock()
        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX helmet model not found: {self.model_path}")
            if ort_module is None:
                try:
                    import onnxruntime as ort_module  # type: ignore[no-redef]
                except ImportError as error:
                    raise RuntimeError(
                        "onnxruntime or onnxruntime-directml is required for "
                        "the classroom helmet detector"
                    ) from error
            session_options = create_ort_session_options(ort_module)
            selected_providers = (
                list(providers)
                if providers is not None
                else _default_ort_providers(ort_module)
            )
            session = ort_module.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=selected_providers,
            )
        self.session = session
        inputs = list(self.session.get_inputs())
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {len(inputs)}")
        self.input_name = str(inputs[0].name)
        shape = getattr(inputs[0], "shape", None)
        if isinstance(shape, Sequence) and len(shape) == 4:
            shape_height, shape_width = shape[2], shape[3]
            if isinstance(shape_width, int) and shape_width > 0:
                self.input_width = shape_width
            if isinstance(shape_height, int) and shape_height > 0:
                self.input_height = shape_height

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        """Letterbox a BGR frame and return a normalized RGB NCHW tensor."""

        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError("frame must be a two- or three-dimensional numpy array")
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.shape[2] != 3:
            raise ValueError(f"Expected 3 frame channels, found {frame.shape[2]}")

        source_height, source_width = frame.shape[:2]
        if source_width <= 0 or source_height <= 0:
            raise ValueError("frame cannot be empty")
        scale = min(
            self.input_width / source_width,
            self.input_height / source_height,
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = _resize_image(frame, (resized_width, resized_height))
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.full(
            (self.input_height, self.input_width, 3),
            114,
            dtype=np.uint8,
        )
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        transform = LetterboxTransform(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            source_width=source_width,
            source_height=source_height,
            input_width=self.input_width,
            input_height=self.input_height,
        )
        return tensor[None, ...], transform

    def decode(
        self,
        output: np.ndarray,
        transform: LetterboxTransform,
    ) -> list[Detection]:
        """Decode raw ``(1, 7, 8400)`` or transposed ``(1, 8400, 7)`` output."""

        prediction = np.asarray(output)
        if prediction.ndim == 3:
            if prediction.shape[0] != 1:
                raise ValueError(f"Expected ONNX batch size 1, got {prediction.shape}")
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {prediction.shape}")

        feature_count = 4 + len(self.class_names)
        if prediction.shape[0] == feature_count:
            prediction = prediction.T
        elif prediction.shape[1] != feature_count:
            raise ValueError(
                "Expected YOLO output shaped (1, 7, 8400) or (1, 8400, 7) "
                f"for three classes, got {np.asarray(output).shape}"
            )

        class_scores = prediction[:, 4:feature_count]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(prediction)), class_ids]
        selected = np.flatnonzero(scores >= self.confidence_threshold)
        if selected.size == 0:
            return []

        boxes: list[BBox] = []
        selected_scores: list[float] = []
        selected_classes: list[int] = []
        for index in selected.tolist():
            center_x, center_y, box_width, box_height = (
                float(value) for value in prediction[index, :4]
            )
            input_box = (
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            )
            restored = transform.restore(input_box)
            if area(restored) <= 0.0:
                continue
            boxes.append(restored)
            selected_scores.append(float(scores[index]))
            selected_classes.append(int(class_ids[index]))

        keep: list[int] = []
        for class_id in sorted(set(selected_classes)):
            local_indexes = [
                index
                for index, candidate_class in enumerate(selected_classes)
                if candidate_class == class_id
            ]
            retained_local = non_max_suppression(
                [boxes[index] for index in local_indexes],
                [selected_scores[index] for index in local_indexes],
                self.iou_threshold,
            )
            keep.extend(local_indexes[index] for index in retained_local)
        keep.sort(key=lambda index: (-selected_scores[index], index))
        keep = keep[: self.max_detections]
        return [
            Detection(
                bbox=boxes[index],
                score=selected_scores[index],
                class_id=selected_classes[index],
                label=self.class_names[selected_classes[index]],
            )
            for index in keep
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, transform = self.preprocess(frame)
        # DirectML sessions must not be entered concurrently.  Pre/postprocess
        # stays outside the lock so multiple callers only serialize ORT itself.
        with self._run_lock:
            outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            raise RuntimeError("ONNX Runtime returned no output tensor")
        return self.decode(np.asarray(outputs[0]), transform)


# Short alias for callers that do not need to mention the export format.
HelmetDetector = YoloHelmetOnnxDetector


class YuNetFaceDetector:
    """Thin, thread-safe wrapper around OpenCV ``FaceDetectorYN``."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.85,
        nms_threshold: float = 0.30,
        top_k: int = 100,
        backend_id: int = 0,
        target_id: int = 0,
        detector: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.score_threshold = float(score_threshold)
        self._lock = threading.Lock()
        if detector is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YuNet model not found: {self.model_path}")
            try:
                import cv2  # type: ignore
            except ImportError as error:
                raise RuntimeError("opencv-python is required for YuNet") from error
            detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                (320, 320),
                self.score_threshold,
                float(nms_threshold),
                int(top_k),
                int(backend_id),
                int(target_id),
            )
        self.detector = detector

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("YuNet expects a color numpy frame")
        frame_height, frame_width = frame.shape[:2]
        with self._lock:
            self.detector.setInputSize((frame_width, frame_height))
            _retval, raw_faces = self.detector.detect(frame)
        if raw_faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in np.asarray(raw_faces):
            if len(row) < 15:
                continue
            x, y, box_width, box_height = (float(value) for value in row[:4])
            score = float(row[14])
            if score < self.score_threshold:
                continue
            landmarks = tuple(
                (float(row[4 + offset]), float(row[5 + offset]))
                for offset in range(0, 10, 2)
            )
            detections.append(
                FaceDetection(
                    bbox=clip_box(
                        (x, y, x + box_width, y + box_height),
                        frame_width,
                        frame_height,
                    ),
                    score=score,
                    landmarks=landmarks,  # type: ignore[arg-type]
                )
            )
        return sorted(detections, key=lambda face: face.score, reverse=True)


class NanoDetPersonDetector:
    """COCO-person wrapper for OpenCV Zoo NanoDet-Plus (416x416).

    The model emits three classification tensors and three DFL box tensors for
    strides 8/16/32.  Outputs are paired by their spatial element count rather
    than by name or list position because OpenCV and ONNX Runtime may expose
    the six graph outputs in different orders.
    """

    _MEAN = np.asarray((103.53, 116.28, 123.675), dtype=np.float32)
    _STD = np.asarray((57.375, 57.12, 58.395), dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path = "models/person/object_detection_nanodet_2022nov.onnx",
        *,
        input_size: tuple[int, int] = (416, 416),
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        reg_max: int = 7,
        providers: Sequence[str] | None = None,
        session: _OnnxSession | None = None,
        ort_module: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_width, self.input_height = (int(value) for value in input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = max(1, int(max_detections))
        self.reg_max = int(reg_max)
        self.strides = (8, 16, 32)
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("NanoDet input size must be positive")
        if self.reg_max < 1:
            raise ValueError("reg_max must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in [0, 1]")

        self._run_lock = threading.Lock()
        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"NanoDet person model not found: {self.model_path}")
            if ort_module is None:
                try:
                    import onnxruntime as ort_module  # type: ignore[no-redef]
                except ImportError as error:
                    raise RuntimeError(
                        "onnxruntime or onnxruntime-directml is required for NanoDet"
                    ) from error
            session = ort_module.InferenceSession(
                str(self.model_path),
                sess_options=create_ort_session_options(ort_module),
                providers=(
                    list(providers)
                    if providers is not None
                    else _default_ort_providers(ort_module)
                ),
            )
        self.session = session
        inputs = list(self.session.get_inputs())
        if len(inputs) != 1:
            raise ValueError(f"Expected one NanoDet input, found {len(inputs)}")
        self.input_name = str(inputs[0].name)
        shape = getattr(inputs[0], "shape", None)
        if isinstance(shape, Sequence) and len(shape) == 4:
            shape_height, shape_width = shape[2], shape[3]
            if isinstance(shape_width, int) and shape_width > 0:
                self.input_width = shape_width
            if isinstance(shape_height, int) and shape_height > 0:
                self.input_height = shape_height

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("NanoDet expects a BGR color numpy frame")
        frame = frame[:, :, :3]
        source_height, source_width = frame.shape[:2]
        if source_width <= 0 or source_height <= 0:
            raise ValueError("frame cannot be empty")
        scale = min(
            self.input_width / source_width,
            self.input_height / source_height,
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = _resize_image(frame, (resized_width, resized_height)).astype(np.float32)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized
        canvas = (canvas - self._MEAN) / self._STD
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor, LetterboxTransform(
            scale=scale,
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            source_width=source_width,
            source_height=source_height,
            input_width=self.input_width,
            input_height=self.input_height,
        )

    @staticmethod
    def _flatten_output(tensor: np.ndarray, feature_count: int) -> np.ndarray | None:
        values = np.asarray(tensor)
        if values.ndim >= 1 and values.shape[0] == 1:
            values = values[0]
        if values.ndim == 2:
            if values.shape[-1] == feature_count:
                return values.reshape(-1, feature_count)
            if values.shape[0] == feature_count:
                return values.reshape(feature_count, -1).T
        if values.ndim == 3:
            if values.shape[0] == feature_count:
                return values.transpose(1, 2, 0).reshape(-1, feature_count)
            if values.shape[-1] == feature_count:
                return values.reshape(-1, feature_count)
        return None

    @staticmethod
    def _sigmoid_if_needed(values: np.ndarray) -> np.ndarray:
        if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
            clipped = np.clip(values, -80.0, 80.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        return values

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        transform: LetterboxTransform,
    ) -> list[Detection]:
        """Decode unordered NanoDet class/DFL tensors into person boxes."""

        classification: dict[int, np.ndarray] = {}
        distribution: dict[int, np.ndarray] = {}
        distribution_features = 4 * (self.reg_max + 1)
        expected_counts = {
            (self.input_width // stride) * (self.input_height // stride): stride
            for stride in self.strides
        }
        for raw_output in outputs:
            class_values = self._flatten_output(np.asarray(raw_output), 80)
            if class_values is not None and len(class_values) in expected_counts:
                classification[len(class_values)] = class_values
                continue
            box_values = self._flatten_output(
                np.asarray(raw_output),
                distribution_features,
            )
            if box_values is not None and len(box_values) in expected_counts:
                distribution[len(box_values)] = box_values

        missing = [
            count
            for count in expected_counts
            if count not in classification or count not in distribution
        ]
        if missing:
            shapes = [tuple(np.asarray(output).shape) for output in outputs]
            raise ValueError(
                "NanoDet outputs could not be paired for all strides; "
                f"missing spatial counts {missing}, output shapes={shapes}"
            )

        decoded_boxes: list[BBox] = []
        decoded_scores: list[float] = []
        projection = np.arange(self.reg_max + 1, dtype=np.float32)
        for count, stride in sorted(expected_counts.items(), key=lambda item: item[1]):
            scores = self._sigmoid_if_needed(classification[count])[:, 0]
            selected = np.flatnonzero(scores >= self.confidence_threshold)
            if selected.size == 0:
                continue
            grid_width = self.input_width // stride
            box_logits = distribution[count][selected].reshape(-1, 4, self.reg_max + 1)
            shifted = box_logits - np.max(box_logits, axis=2, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= np.sum(probabilities, axis=2, keepdims=True)
            distances = np.sum(probabilities * projection, axis=2) * stride

            for local_index, prior_index in enumerate(selected.tolist()):
                grid_x = prior_index % grid_width
                grid_y = prior_index // grid_width
                center_x = (grid_x + 0.5) * stride
                center_y = (grid_y + 0.5) * stride
                left, top, right, bottom = distances[local_index]
                restored = transform.restore(
                    (
                        center_x - float(left),
                        center_y - float(top),
                        center_x + float(right),
                        center_y + float(bottom),
                    )
                )
                if area(restored) <= 0.0:
                    continue
                decoded_boxes.append(restored)
                decoded_scores.append(float(scores[prior_index]))

        keep = non_max_suppression(
            decoded_boxes,
            decoded_scores,
            self.iou_threshold,
        )
        keep = sorted(keep, key=lambda index: (-decoded_scores[index], index))[
            : self.max_detections
        ]
        return [
            Detection(
                bbox=decoded_boxes[index],
                score=decoded_scores[index],
                class_id=0,
                label="person",
            )
            for index in keep
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, transform = self.preprocess(frame)
        with self._run_lock:
            outputs = self.session.run(None, {self.input_name: tensor})
        return self.decode([np.asarray(output) for output in outputs], transform)


PersonDetector = NanoDetPersonDetector
