"""Video-source normalization and a reconnecting latest-frame capture worker.

The module deliberately has no import-time OpenCV dependency.  ``cv2`` is
loaded only by the default capture factory, which keeps unit tests and tools
that only inspect configuration lightweight.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit


SourceKind = Literal["ip_webcam", "webcam", "file", "stream"]


class CaptureLike(Protocol):
    """Small subset of ``cv2.VideoCapture`` used by :class:`CaptureWorker`."""

    def isOpened(self) -> bool:  # noqa: N802 - follows OpenCV's public API
        ...

    def read(self) -> tuple[bool, Any]:
        ...

    def release(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """A normalized value suitable for ``cv2.VideoCapture``."""

    value: str | int
    kind: SourceKind
    label: str
    reconnect: bool

    @property
    def capture_value(self) -> str | int:
        return self.value


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One captured frame plus ordering and timing metadata."""

    frame: Any
    sequence: int
    captured_at: float
    monotonic_at: float
    source: SourceSpec


def normalize_ip_webcam_url(value: str, *, default_scheme: str = "http") -> str:
    """Return an Android IP Webcam MJPEG endpoint ending in ``/video``.

    Accepted inputs include ``192.168.1.20:8080``, a base HTTP URL, or an
    already complete ``/video`` URL.  Authentication, port and query values are
    retained, while any supplied path is normalized to the application's MJPEG
    endpoint.
    """

    text = str(value).strip()
    if not text:
        raise ValueError("IP Webcam URL must not be empty")

    if text.startswith("//"):
        text = f"{default_scheme}:{text}"
    elif "://" not in text:
        text = f"{default_scheme}://{text}"

    try:
        parts = urlsplit(text)
        hostname = parts.hostname
        # Accessing ``port`` also validates malformed/non-numeric ports.
        _ = parts.port
    except ValueError as exc:
        raise ValueError(f"Invalid IP Webcam URL: {value!r}") from exc

    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("IP Webcam URL must use http:// or https://")
    if not hostname or not parts.netloc:
        raise ValueError(f"Invalid IP Webcam URL: {value!r}")

    return urlunsplit((scheme, parts.netloc, "/video", parts.query, ""))


def redact_source_label(value: str | int) -> str:
    """Return a display-safe source label without credentials or query secrets."""

    if isinstance(value, int):
        return f"Webcam {value}"
    text = str(value)
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return "Nguồn mạng"
    if not hostname:
        return "Nguồn mạng"
    safe_host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    netloc = f"{safe_host}:{port}" if port is not None else safe_host
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))


def parse_source(
    value: SourceSpec | str | int | os.PathLike[str],
    *,
    ip_webcam: bool | None = None,
) -> SourceSpec:
    """Normalize a webcam index, video path, or network-stream URL.

    HTTP(S) inputs default to the Android IP Webcam convention and therefore
    receive a ``/video`` path.  Pass ``ip_webcam=False`` to preserve a generic
    HTTP(S) stream URL exactly as supplied.
    """

    if isinstance(value, SourceSpec):
        return value

    if isinstance(value, int):
        if value < 0:
            raise ValueError("Webcam index must be zero or greater")
        return SourceSpec(value=value, kind="webcam", label=f"Webcam {value}", reconnect=True)

    if isinstance(value, Path):
        text = str(value)
    else:
        text = str(value).strip()
    if not text:
        raise ValueError("Video source must not be empty")

    if text.isdecimal():
        return parse_source(int(text))
    if text.startswith("-") and text[1:].isdecimal():
        raise ValueError("Webcam index must be zero or greater")

    candidate = text if "://" in text else ""
    if candidate:
        parts = urlsplit(candidate)
        scheme = parts.scheme.lower()
        if scheme in {"http", "https"}:
            use_ip_webcam = ip_webcam is not False
            normalized = normalize_ip_webcam_url(text) if use_ip_webcam else text
            return SourceSpec(
                value=normalized,
                kind="ip_webcam" if use_ip_webcam else "stream",
                label=redact_source_label(normalized),
                reconnect=True,
            )
        if scheme in {"rtsp", "rtmp"} and parts.netloc:
            return SourceSpec(
                value=text,
                kind="stream",
                label=redact_source_label(text),
                reconnect=True,
            )
        raise ValueError(f"Unsupported video stream URL: {value!r}")

    expanded = os.path.expandvars(os.path.expanduser(text))
    return SourceSpec(value=expanded, kind="file", label=expanded, reconnect=False)


def _default_capture_factory(source: str | int) -> CaptureLike:
    import cv2

    # Open/read timeout values are open-only parameters for several OpenCV
    # backends.  Passing them to ``VideoCapture`` prevents a dead network URL
    # from freezing the Streamlit run before ``CaptureWorker`` can enforce its
    # own first-frame deadline.  Older/fake OpenCV builds may not support the
    # overload, so retain the one-argument fallback.
    open_parameters: list[int] = []
    for property_name, property_value in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
        ("CAP_PROP_READ_TIMEOUT_MSEC", 5000),
    ):
        property_id = getattr(cv2, property_name, None)
        if property_id is not None:
            open_parameters.extend((int(property_id), property_value))
    try:
        capture = (
            cv2.VideoCapture(source, cv2.CAP_ANY, open_parameters)
            if open_parameters
            else cv2.VideoCapture(source)
        )
    except (AttributeError, TypeError, cv2.error):
        capture = cv2.VideoCapture(source)
    # Phone recordings commonly store portrait orientation as container
    # metadata while the encoded frames remain landscape.  OpenCV leaves that
    # metadata unapplied by default, which turns faces sideways and makes
    # YuNet miss them.  Enable FFmpeg's automatic rotation when the installed
    # OpenCV build exposes the property; unsupported live backends simply
    # return False from ``set`` and continue unchanged.
    capture_properties = ((getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None), 1),)
    for property_id, property_value in capture_properties:
        if property_id is None:
            continue
        try:
            capture.set(property_id, property_value)
        except (AttributeError, TypeError):
            pass
    return capture


_QUEUE_STOP = object()


class CaptureWorker:
    """Continuously capture frames without allowing a stale-frame backlog.

    Only the newest frame is retained.  Live sources reconnect with exponential
    backoff; local files stop normally at EOF.  ``stop()`` releases the active
    capture to unblock a pending OpenCV read before joining the worker thread.
    """

    def __init__(
        self,
        source: SourceSpec | str | int | os.PathLike[str],
        *,
        capture_factory: Callable[[str | int], CaptureLike] | None = None,
        queue_size: int = 1,
        initial_backoff: float = 0.25,
        max_backoff: float = 5.0,
        backoff_multiplier: float = 2.0,
        startup_timeout: float | None = 15.0,
        name: str = "classroom-capture",
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if initial_backoff < 0 or max_backoff < initial_backoff:
            raise ValueError("Reconnect backoff values are invalid")
        if backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if startup_timeout is not None and startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive or None")

        self.source = parse_source(source)
        self._capture_factory = capture_factory or _default_capture_factory
        self._frames: queue.Queue[FramePacket | object] = queue.Queue(maxsize=queue_size)
        self._initial_backoff = float(initial_backoff)
        self._max_backoff = float(max_backoff)
        self._backoff_multiplier = float(backoff_multiplier)
        self._startup_timeout = (
            None if startup_timeout is None else float(startup_timeout)
        )
        self._name = name

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._capture: CaptureLike | None = None
        self._latest: FramePacket | None = None
        self._sequence = 0
        self._connected = False
        self._last_error: str | None = None
        self._reconnect_count = 0

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    @property
    def reconnect_count(self) -> int:
        with self._state_lock:
            return self._reconnect_count

    @property
    def sequence(self) -> int:
        with self._state_lock:
            return self._sequence

    @property
    def latest(self) -> FramePacket | None:
        with self._state_lock:
            return self._latest

    def latest_frame(self, *, copy: bool = False) -> Any | None:
        packet = self.latest
        if packet is None:
            return None
        frame = packet.frame
        if copy and hasattr(frame, "copy"):
            return frame.copy()
        return frame

    def start(self) -> "CaptureWorker":
        with self._state_lock:
            if self.running:
                return self
            self._stop_event.clear()
            self._connected = False
            self._last_error = None
            self._latest = None
            self._sequence = 0
            self._reconnect_count = 0
            self._drain_frame_queue()
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        self._release_active_capture()
        self._wake_consumers()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._state_lock:
            self._connected = False

    def get_latest(self, timeout: float | None = None) -> FramePacket | None:
        """Return the newest queued frame, or ``None`` when stopped/timed out."""

        try:
            item = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _QUEUE_STOP:
            return None
        assert isinstance(item, FramePacket)

        # A larger configured queue still has latest-frame semantics.
        newest = item
        while True:
            try:
                candidate = self._frames.get_nowait()
            except queue.Empty:
                break
            if candidate is _QUEUE_STOP:
                break
            assert isinstance(candidate, FramePacket)
            newest = candidate
        return newest

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        timeout: float | None = None,
    ) -> FramePacket | None:
        """Wait until a frame newer than ``after_sequence`` is available."""

        latest = self.latest
        if latest is not None and latest.sequence > after_sequence:
            return latest

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self._stop_event.is_set():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            packet = self.get_latest(timeout=remaining)
            if packet is not None and packet.sequence > after_sequence:
                return packet
            if deadline is not None and time.monotonic() >= deadline:
                return None
        return None

    def _run(self) -> None:
        backoff = self._initial_backoff
        started = time.monotonic()
        try:
            while not self._stop_event.is_set():
                capture: CaptureLike | None = None
                received_frame = False
                try:
                    capture = self._capture_factory(self.source.capture_value)
                    self._set_active_capture(capture)
                    if not capture.isOpened():
                        raise RuntimeError(f"Could not open video source: {self.source.label}")

                    self._set_connection_state(connected=True, error=None)
                    while not self._stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok or frame is None:
                            # Reaching EOF after at least one valid frame is the
                            # expected completion path for an uploaded/local file.
                            # A stream dropping frames (or an unreadable file that
                            # never yielded a frame) is still surfaced as an error.
                            if self.source.kind == "file" and received_frame:
                                self._set_connection_state(connected=False, error=None)
                                break
                            raise RuntimeError(f"Video source stopped returning frames: {self.source.label}")
                        received_frame = True
                        backoff = self._initial_backoff
                        self._publish(frame)
                except Exception as exc:
                    if not self._stop_event.is_set():
                        self._set_connection_state(
                            connected=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                finally:
                    if capture is not None:
                        self._release_active_capture(expected=capture)
                    self._set_connection_state(connected=False)

                if self._stop_event.is_set() or not self.source.reconnect:
                    break
                if (
                    self._startup_timeout is not None
                    and self.sequence == 0
                    and time.monotonic() - started >= self._startup_timeout
                ):
                    self._set_connection_state(
                        connected=False,
                        error=(
                            self.last_error
                            or f"Timed out waiting for the first frame: {self.source.label}"
                        ),
                    )
                    break

                with self._state_lock:
                    self._reconnect_count += 1
                if self._stop_event.wait(backoff):
                    break
                if received_frame:
                    backoff = self._initial_backoff
                else:
                    backoff = min(
                        self._max_backoff,
                        max(self._initial_backoff, backoff * self._backoff_multiplier),
                    )
        finally:
            self._release_active_capture()
            self._set_connection_state(connected=False)
            self._wake_consumers()

    def _publish(self, frame: Any) -> None:
        now_monotonic = time.monotonic()
        with self._state_lock:
            self._sequence += 1
            packet = FramePacket(
                frame=frame,
                sequence=self._sequence,
                captured_at=time.time(),
                monotonic_at=now_monotonic,
                source=self.source,
            )
            self._latest = packet

        # A local video is a finite deterministic input.  Apply backpressure so
        # the inference worker processes every frame instead of letting OpenCV
        # decode the whole file into a one-frame queue before inference starts.
        # Live cameras keep the low-latency newest-frame behavior below.
        if self.source.kind == "file":
            while not self._stop_event.is_set():
                try:
                    self._frames.put(packet, timeout=0.10)
                    return
                except queue.Full:
                    continue
            return

        while True:
            try:
                self._frames.put_nowait(packet)
                break
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass

    def _set_connection_state(
        self,
        *,
        connected: bool,
        error: str | None | object = _QUEUE_STOP,
    ) -> None:
        with self._state_lock:
            self._connected = connected
            if error is not _QUEUE_STOP:
                self._last_error = error if isinstance(error, str) else None

    def _set_active_capture(self, capture: CaptureLike) -> None:
        with self._capture_lock:
            self._capture = capture

    def _release_active_capture(self, *, expected: CaptureLike | None = None) -> None:
        with self._capture_lock:
            if expected is not None and self._capture is not expected:
                return
            capture = self._capture
            self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                # Stopping must stay best-effort even for a broken camera driver.
                pass

    def _drain_frame_queue(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _wake_consumers(self) -> None:
        if self.source.kind == "file" and not self._stop_event.is_set():
            # Preserve the final queued video frame at natural EOF.  The
            # consumer will take it and then observe that the worker stopped.
            # If the queue is already empty, a sentinel avoids one poll delay.
            try:
                self._frames.put_nowait(_QUEUE_STOP)
            except queue.Full:
                pass
            return
        while True:
            try:
                self._frames.put_nowait(_QUEUE_STOP)
                return
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    return

    def __enter__(self) -> "CaptureWorker":
        return self.start()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop()
