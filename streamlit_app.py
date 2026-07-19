"""Streamlit 1.59 UI for the classroom helmet-compliance demonstration."""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classroom_demo.ui_helpers import (  # noqa: E402
    PipelineAdapter,
    PipelineCallError,
    PipelineUnavailable,
    UISnapshot,
    connection_view,
    dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes,
    events_dataframe,
    filter_events,
    frame_to_jpeg_bytes,
    inspect_demo_readiness,
    materialize_uploaded_video,
    people_dataframe,
    students_dataframe,
)
from src.classroom_demo.sources import CaptureWorker, redact_source_label  # noqa: E402


st.set_page_config(
    page_title="Giám sát mũ bảo hiểm lớp học",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


SOURCE_OPTIONS = ("IP Webcam", "Webcam", "Video")
STATUS_OPTIONS = ("Đội đúng", "Đội sai", "Không mũ", "Chưa rõ")


def initialize_session_state() -> None:
    """Initialize all per-tab, per-browser state in one predictable place."""

    defaults: dict[str, Any] = {
        "monitor_running": False,
        "runtime_message": "",
        "runtime_level": "info",
        "active_config": {},
        "latest_frame": None,
        "latest_frame_channels": "BGR",
        "latest_stats": {},
        "latest_people": [],
        "latest_events": [],
        "latest_updated_at": None,
        "poll_error_count": 0,
        "history_records": [],
        "history_loaded": False,
        "history_message": "",
        "last_event_count": 0,
        "registration_result": None,
        "connection_code": "stopped",
        "source_kind": "IP Webcam",
        "ip_webcam_url": "http://192.168.1.100:8080/video",
        "webcam_index": 0,
        "model_path": "models/best.onnx",
        "device": "auto",
        "detection_confidence": 0.35,
        "face_match_threshold": 0.50,
        "face_match_margin": 0.10,
        "save_evidence": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource
def get_pipeline_adapter(project_root: str) -> PipelineAdapter:
    """Create one shared, heavyweight pipeline bridge and reuse it on reruns."""

    return PipelineAdapter(project_root)


@st.cache_data(ttl=10, max_entries=4)
def get_demo_readiness(project_root: str) -> dict[str, Any]:
    """Cache the cheap filesystem readiness check without caching model objects."""

    return inspect_demo_readiness(project_root)


def stat_value(stats: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key not in stats or stats[key] is None:
            continue
        try:
            return float(stats[key])
        except (TypeError, ValueError):
            continue
    return float(default)


def integer_metric(value: float) -> str:
    return f"{max(0, int(round(value))):,}".replace(",", ".")


def format_updated_at(value: Any) -> str:
    if not isinstance(value, datetime):
        return "Chưa có dữ liệu"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    local = value.astimezone(timezone(timedelta(hours=7)))
    return local.strftime("%H:%M:%S · %d/%m/%Y")


def show_runtime_message() -> None:
    message = str(st.session_state.runtime_message).strip()
    if not message:
        return
    level = st.session_state.runtime_level
    if level == "error":
        st.error(message, icon=":material/error:")
    elif level == "warning":
        st.warning(message, icon=":material/warning:")
    elif level == "success":
        st.success(message, icon=":material/check_circle:")
    else:
        st.info(message, icon=":material/info:")


def build_source_config(video_upload: Any) -> dict[str, Any]:
    source_kind = st.session_state.source_kind
    if source_kind == "IP Webcam":
        source = str(st.session_state.ip_webcam_url).strip()
        if not source:
            raise ValueError("Hãy nhập địa chỉ IP Webcam.")
        if not re.match(r"^https?://", source, flags=re.IGNORECASE):
            source = f"http://{source}"
        source_type = "ip_webcam"
        source_name = redact_source_label(source)
    elif source_kind == "Webcam":
        source = int(st.session_state.webcam_index)
        source_type = "webcam"
        source_name = f"Webcam {source}"
    else:
        if video_upload is None:
            raise ValueError("Hãy chọn một tệp video trước khi bắt đầu.")
        source_path = materialize_uploaded_video(
            video_upload,
            PROJECT_ROOT / "outputs" / "classroom" / "uploads",
        )
        source = str(source_path)
        source_type = "video"
        source_name = str(getattr(video_upload, "name", source_path.name))

    raw_model_path = Path(str(st.session_state.model_path).strip())
    model_path = raw_model_path if raw_model_path.is_absolute() else PROJECT_ROOT / raw_model_path
    selected_device = st.session_state.device
    providers = {
        "auto": None,
        "dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
        "cpu": ("CPUExecutionProvider",),
    }[selected_device]
    return {
        "source_kind": source_type,
        "source_type": source_type,
        "source": source,
        "source_name": source_name,
        "ip_webcam_url": source if source_type == "ip_webcam" else "",
        "model_path": str(model_path),
        "helmet_model": str(model_path),
        "project_root": str(PROJECT_ROOT),
        "device": selected_device,
        "providers": providers,
        "detection_confidence": float(st.session_state.detection_confidence),
        "conf": float(st.session_state.detection_confidence),
        "person_confidence": float(st.session_state.detection_confidence),
        "helmet_confidence": float(st.session_state.detection_confidence),
        "face_match_threshold": float(st.session_state.face_match_threshold),
        "face_threshold": float(st.session_state.face_match_threshold),
        "cosine_threshold": float(st.session_state.face_match_threshold),
        "face_match_margin": float(st.session_state.face_match_margin),
        "cosine_margin": float(st.session_state.face_match_margin),
        "source_startup_timeout": 15.0,
        "save_evidence": bool(st.session_state.save_evidence),
        "save_event_snapshots": bool(st.session_state.save_evidence),
        "output_dir": str(PROJECT_ROOT / "outputs" / "classroom"),
    }


def probe_video_source(source: str | int, *, timeout: float = 8.0) -> tuple[int, int]:
    """Open a source without loading AI models and verify one real frame."""

    worker = CaptureWorker(source, startup_timeout=timeout, name="source-probe")
    try:
        worker.start()
        packet = worker.wait_for_frame(timeout=timeout)
        if packet is None:
            detail = worker.last_error or "không nhận được khung hình đầu tiên"
            raise RuntimeError(detail)
        shape = getattr(packet.frame, "shape", ())
        if len(shape) < 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
            raise RuntimeError("nguồn trả về khung hình không hợp lệ")
        return int(shape[1]), int(shape[0])
    finally:
        worker.stop(timeout=2.0)


def render_sidebar(adapter: PipelineAdapter) -> tuple[bool, bool, Any]:
    running = bool(st.session_state.monitor_running)
    readiness = get_demo_readiness(str(PROJECT_ROOT))
    with st.sidebar:
        if readiness["ready"]:
            st.badge(
                "Kỹ thuật sẵn sàng",
                icon=":material/check_circle:",
                color="green" if not readiness["warnings"] else "orange",
            )
        else:
            st.badge("Chưa sẵn sàng", icon=":material/error:", color="red")
        st.caption(
            f"Sinh viên đủ ảnh: {readiness['ready_student_count']} / "
            f"{readiness['student_count']} · Model: {readiness['model_count']} / 4"
        )
        with st.expander("Kiểm tra sẵn sàng", icon=":material/fact_check:"):
            for message in readiness["errors"]:
                st.error(message, icon=":material/error:")
            for message in readiness["warnings"]:
                st.warning(message, icon=":material/warning:")
            if not readiness["errors"] and not readiness["warnings"]:
                st.success("Model, roster và ảnh tham chiếu đều đạt yêu cầu.")

        st.header("Nguồn video")
        st.segmented_control(
            "Loại nguồn",
            SOURCE_OPTIONS,
            key="source_kind",
            required=True,
            disabled=running,
            width="stretch",
        )

        video_upload = None
        if st.session_state.source_kind == "IP Webcam":
            st.text_input(
                "Địa chỉ IP Webcam",
                key="ip_webcam_url",
                placeholder="http://192.168.1.100:8080/video",
                help="Điện thoại và máy tính phải truy cập được lẫn nhau trên cùng mạng.",
                disabled=running,
                width="stretch",
            )
        elif st.session_state.source_kind == "Webcam":
            st.number_input(
                "Chỉ số webcam",
                min_value=0,
                max_value=10,
                step=1,
                key="webcam_index",
                disabled=running,
                width="stretch",
            )
        else:
            video_upload = st.file_uploader(
                "Chọn video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="video_source_upload",
                max_upload_size=512,
                disabled=running,
                width="stretch",
            )

        with st.expander("Cấu hình nhận diện", icon=":material/tune:"):
            st.text_input(
                "Đường dẫn model",
                key="model_path",
                disabled=running,
                width="stretch",
            )
            st.selectbox(
                "Thiết bị xử lý",
                options=["auto", "dml", "cpu"],
                format_func=lambda value: {
                    "auto": "Tự động",
                    "dml": "GPU DirectML",
                    "cpu": "CPU",
                }[value],
                key="device",
                disabled=running,
                width="stretch",
            )
            st.slider(
                "Ngưỡng phát hiện",
                min_value=0.10,
                max_value=0.90,
                step=0.05,
                key="detection_confidence",
                disabled=running,
                width="stretch",
            )
            st.slider(
                "Ngưỡng nhận diện khuôn mặt",
                min_value=0.48,
                max_value=0.90,
                step=0.01,
                key="face_match_threshold",
                disabled=running,
                width="stretch",
                help="Không nên hạ dưới 0,50; bằng chứng yếu sẽ giữ nhãn Chưa rõ.",
            )
            st.slider(
                "Biên phân biệt hai người gần nhất",
                min_value=0.05,
                max_value=0.30,
                step=0.01,
                key="face_match_margin",
                disabled=running,
                width="stretch",
                help="Mặc định 0,10 để tránh nhận nhầm các khuôn mặt gần giống nhau.",
            )
            st.toggle(
                "Lưu ảnh vi phạm",
                key="save_evidence",
                disabled=running,
                width="stretch",
            )

        if st.button(
            "Kiểm tra nguồn video",
            icon=":material/network_check:",
            disabled=running,
            width="stretch",
        ):
            try:
                probe_config = build_source_config(video_upload)
                with st.spinner("Đang kiểm tra khung hình đầu tiên…", show_time=True):
                    width, height = probe_video_source(probe_config["source"])
                st.session_state.runtime_message = (
                    f"Nguồn video hoạt động: nhận được khung hình {width}×{height}."
                )
                st.session_state.runtime_level = "success"
            except (OSError, RuntimeError, ValueError) as exc:
                st.session_state.runtime_message = f"Không thể mở nguồn video: {exc}"
                st.session_state.runtime_level = "error"

        with st.container(horizontal=True, horizontal_alignment="distribute"):
            start_clicked = st.button(
                "Bắt đầu",
                type="primary",
                icon=":material/play_arrow:",
                disabled=running or not readiness["ready"],
                width="stretch",
            )
            stop_clicked = st.button(
                "Dừng",
                icon=":material/stop:",
                disabled=not running,
                width="stretch",
            )

        if running:
            st.badge("Đang kết nối/giám sát", icon=":material/videocam:", color="blue")
        elif adapter.last_error:
            st.badge("Backend cần kiểm tra", icon=":material/warning:", color="orange")
        else:
            st.badge("Đã dừng", icon=":material/pause_circle:", color="gray")
        st.caption(f"Backend: {adapter.backend_name}")

        if adapter.last_error and st.button(
            "Thử nạp lại backend",
            icon=":material/refresh:",
            type="tertiary",
            width="stretch",
        ):
            adapter.reset_backend()
            st.session_state.monitor_running = False
            st.session_state.runtime_message = "Đã xóa backend cũ; hệ thống sẽ nạp lại ở lần gọi tiếp theo."
            st.session_state.runtime_level = "info"
            st.rerun()

        st.caption("Dữ liệu khuôn mặt chỉ dùng cho demo nội bộ khi đã có sự đồng ý.")
    return start_clicked, stop_clicked, video_upload


def handle_source_controls(
    adapter: PipelineAdapter,
    *,
    start_clicked: bool,
    stop_clicked: bool,
    video_upload: Any,
) -> None:
    if stop_clicked:
        try:
            adapter.stop()
            st.session_state.runtime_message = "Đã dừng nguồn video an toàn."
            st.session_state.runtime_level = "success"
        except PipelineCallError as exc:
            st.session_state.runtime_message = str(exc)
            st.session_state.runtime_level = "error"
        finally:
            st.session_state.monitor_running = False
            st.session_state.history_loaded = False

    if start_clicked:
        try:
            config = build_source_config(video_upload)
            with st.spinner("Đang khởi tạo camera và mô hình…", show_time=True):
                adapter.start(config)
            st.session_state.active_config = config
            st.session_state.monitor_running = True
            st.session_state.poll_error_count = 0
            st.session_state.runtime_message = (
                f"Đang kết nối và chờ khung hình đầu tiên: {config['source_name']}"
            )
            st.session_state.runtime_level = "info"
        except (ValueError, PipelineUnavailable, PipelineCallError) as exc:
            st.session_state.monitor_running = False
            st.session_state.runtime_message = str(exc)
            st.session_state.runtime_level = "error"


def people_column_config() -> dict[str, Any]:
    return {
        "MSSV": st.column_config.TextColumn("MSSV", pinned=True),
        "Tin cậy khuôn mặt": st.column_config.NumberColumn(
            "Tin cậy khuôn mặt", format="percent"
        ),
        "Tin cậy detector mũ": st.column_config.NumberColumn(
            "Tin cậy detector mũ",
            format="percent",
            help="Điểm tin cậy của hộp helmet/no_helmet, không phải xác suất Đội sai.",
        ),
    }


def events_column_config() -> dict[str, Any]:
    return {
        "Thời gian": st.column_config.DatetimeColumn(
            "Thời gian", format="DD/MM/YYYY HH:mm:ss"
        ),
        "MSSV": st.column_config.TextColumn("MSSV", pinned=True),
        "Tin cậy": st.column_config.NumberColumn("Tin cậy", format="percent"),
    }


def cache_snapshot(snapshot: UISnapshot) -> None:
    event_count = int(snapshot.stats.get("event_count") or 0)
    if event_count > int(st.session_state.last_event_count or 0):
        st.session_state.history_loaded = False
    st.session_state.last_event_count = event_count
    st.session_state.latest_frame = snapshot.frame
    st.session_state.latest_frame_channels = snapshot.channels
    st.session_state.latest_stats = snapshot.stats
    st.session_state.latest_people = snapshot.people
    st.session_state.latest_events = snapshot.events
    st.session_state.latest_updated_at = snapshot.updated_at


def render_live_contents() -> None:
    stats = dict(st.session_state.latest_stats or {})
    people = people_dataframe(st.session_state.latest_people)
    events = events_dataframe(st.session_state.latest_events)

    active_people = stat_value(
        stats,
        "active_people",
        "active_count",
        "person_count",
        "people_count",
        "active_rider_count",
        default=float(len(people)),
    )
    identified_people = stat_value(
        stats,
        "identified_people",
        "identified_count",
        "recognized_count",
        default=float((people["MSSV"].astype(str).str.strip() != "").sum()) if not people.empty else 0,
    )
    violations = stat_value(stats, "current_violations", "violation_count", default=-1)
    if violations < 0:
        violations = (
            float(people["Trạng thái mũ"].isin(["Đội sai", "Không mũ"]).sum())
            if not people.empty
            else 0.0
        )
    fps = stat_value(stats, "fps", "processing_fps", "inference_fps", default=0.0)
    unknown_people = stat_value(
        stats,
        "unknown_identity_count",
        default=max(0.0, active_people - identified_people),
    )

    connection = connection_view(
        stats,
        monitor_running=bool(st.session_state.monitor_running),
    )
    if connection["code"] == "live" and st.session_state.connection_code != "live":
        st.toast("Camera và mô hình đã sẵn sàng", icon=":material/check_circle:")
    st.session_state.connection_code = connection["code"]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(
            connection["label"],
            icon=connection["icon"],
            color=connection["color"],
        )
        if connection["message"]:
            st.caption(connection["message"])

    with st.container(horizontal=True):
        st.metric("Người trong khung", integer_metric(active_people), border=True)
        st.metric("Đã nhận diện", integer_metric(identified_people), border=True)
        st.metric("Chưa xác định", integer_metric(unknown_people), border=True)
        st.metric("Vi phạm hiện tại", integer_metric(violations), border=True)
        st.metric("Tốc độ xử lý", f"{fps:.1f} FPS", border=True)

    video_column, people_column = st.columns([1.7, 1.0], vertical_alignment="top")
    with video_column:
        with st.container(border=True):
            st.subheader("Camera trực tiếp")
            if st.session_state.latest_frame is not None:
                st.image(
                    st.session_state.latest_frame,
                    channels=st.session_state.latest_frame_channels,
                    width="stretch",
                    output_format="JPEG",
                )
                st.caption(
                    f"Cập nhật: {format_updated_at(st.session_state.latest_updated_at)}"
                )
            elif st.session_state.monitor_running:
                st.info(
                    "Đang chờ khung hình đầu tiên từ camera…",
                    icon=":material/hourglass_top:",
                )
            else:
                st.info(
                    "Chọn nguồn ở thanh bên và nhấn **Bắt đầu**.",
                    icon=":material/videocam_off:",
                )

    with people_column:
        with st.container(border=True):
            st.subheader("Người đang xuất hiện")
            st.dataframe(
                people,
                width="stretch",
                height=390,
                hide_index=True,
                column_order=[
                    "MSSV",
                    "Họ và tên",
                    "Trạng thái mũ",
                    "Tin cậy khuôn mặt",
                    "Tin cậy detector mũ",
                    "Giải thích",
                ],
                column_config=people_column_config(),
                placeholder="Chưa phát hiện người trong khung hình.",
            )

    with st.container(border=True):
        st.subheader("Sự kiện gần đây")
        st.dataframe(
            events.head(10),
            width="stretch",
            height=250,
            hide_index=True,
            column_order=[
                "Thời gian",
                "MSSV",
                "Họ và tên",
                "Trạng thái mũ",
                "Tin cậy",
            ],
            column_config=events_column_config(),
            placeholder="Chưa có sự kiện được xác nhận.",
        )


@st.fragment(run_every=0.5)
def live_monitor_fragment(adapter: PipelineAdapter) -> None:
    """Poll only the live dashboard twice per second, without rerunning the app."""

    if st.session_state.monitor_running:
        try:
            snapshot = adapter.snapshot()
            cache_snapshot(snapshot)
            st.session_state.poll_error_count = 0
            backend_state = str(snapshot.stats.get("state", "")).lower()
            backend_error = str(snapshot.stats.get("last_error") or snapshot.message or "").strip()
            if backend_state == "failed":
                st.session_state.monitor_running = False
                st.session_state.runtime_message = (
                    backend_error or "Pipeline đã dừng do lỗi xử lý."
                )
                st.session_state.runtime_level = "error"
                st.rerun()
            elif backend_state == "stopped" and not snapshot.stats.get("running", False):
                st.session_state.monitor_running = False
                st.session_state.runtime_message = "Nguồn video đã kết thúc."
                st.session_state.runtime_level = "success"
                st.rerun()
            elif backend_error:
                st.warning(backend_error, icon=":material/warning:")
        except (PipelineUnavailable, PipelineCallError) as exc:
            st.session_state.poll_error_count += 1
            st.error(str(exc), icon=":material/sync_problem:")
            st.caption("Hệ thống sẽ tiếp tục thử ở lần làm mới tiếp theo.")
    render_live_contents()


def render_registration(adapter: PipelineAdapter) -> None:
    st.subheader("Đăng ký sinh viên")
    st.caption(
        "Dùng nhiều ảnh rõ mặt hoặc khung hình camera hiện tại. "
        "Ảnh đa góc giúp nhận diện ổn định hơn trong phòng đông người."
    )

    form_column, preview_column = st.columns([1.15, 0.85], vertical_alignment="top")
    with form_column:
        with st.form("student_registration", clear_on_submit=False, width="stretch"):
            student_id = st.text_input(
                "Mã số sinh viên",
                placeholder="Ví dụ: CE182206",
                key="registration_student_id",
                width="stretch",
            )
            full_name = st.text_input(
                "Họ và tên",
                placeholder="Nguyễn Thị Bích Tuyền",
                key="registration_full_name",
                width="stretch",
            )
            uploaded_images = st.file_uploader(
                "Ảnh khuôn mặt",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                max_upload_size=20,
                key="registration_images",
                width="stretch",
            )
            use_current_frame = st.checkbox(
                "Dùng thêm khung hình camera hiện tại",
                disabled=st.session_state.latest_frame is None,
                key="registration_use_current_frame",
            )
            has_consent = st.checkbox(
                "Đã có sự đồng ý sử dụng ảnh cho mục đích demo nội bộ",
                key="registration_consent",
            )
            submitted = st.form_submit_button(
                "Lưu sinh viên",
                type="primary",
                icon=":material/person_add:",
                width="stretch",
            )

        if submitted:
            clean_id = student_id.strip().upper()
            clean_name = " ".join(full_name.split())
            errors: list[str] = []
            if not re.fullmatch(r"[A-Z0-9_-]{4,32}", clean_id):
                errors.append("MSSV phải gồm 4–32 ký tự chữ, số, gạch nối hoặc gạch dưới.")
            if len(clean_name) < 2:
                errors.append("Hãy nhập họ và tên sinh viên.")
            if not has_consent:
                errors.append("Cần xác nhận sự đồng ý trước khi lưu dữ liệu khuôn mặt.")

            images = [item.getvalue() for item in (uploaded_images or [])]
            if use_current_frame and st.session_state.latest_frame is not None:
                try:
                    images.append(
                        frame_to_jpeg_bytes(
                            st.session_state.latest_frame,
                            channels=st.session_state.latest_frame_channels,
                        )
                    )
                except (OSError, ValueError) as exc:
                    errors.append(f"Không thể dùng khung hình hiện tại: {exc}")
            if not images:
                errors.append("Hãy tải ít nhất một ảnh hoặc dùng khung hình hiện tại.")

            if errors:
                for error in errors:
                    st.error(error, icon=":material/error:")
            else:
                try:
                    with st.spinner("Đang tạo dữ liệu nhận diện…", show_time=True):
                        result = adapter.register_student(
                            student_id=clean_id,
                            full_name=clean_name,
                            images=images,
                        )
                    st.session_state.registration_result = result
                    get_demo_readiness.clear()
                    st.success(
                        f"Đã đăng ký {clean_id} · {clean_name} với {len(images)} ảnh.",
                        icon=":material/check_circle:",
                    )
                    st.toast("Đăng ký sinh viên thành công", icon=":material/person_check:")
                except (PipelineUnavailable, PipelineCallError) as exc:
                    st.error(str(exc), icon=":material/error:")

    with preview_column:
        with st.container(border=True):
            st.subheader("Khung hình dùng để đăng ký")
            if st.session_state.latest_frame is None:
                st.info(
                    "Chưa có khung hình. Bạn vẫn có thể tải ảnh từ máy tính.",
                    icon=":material/add_photo_alternate:",
                )
            else:
                st.image(
                    st.session_state.latest_frame,
                    channels=st.session_state.latest_frame_channels,
                    width="stretch",
                    output_format="JPEG",
                    caption="Khung hình hiện tại; nên bảo đảm chỉ có một khuôn mặt rõ để đăng ký.",
                )
        if st.session_state.registration_result:
            with st.expander("Kết quả đăng ký", icon=":material/task_alt:"):
                st.json(st.session_state.registration_result)

    st.subheader("Danh sách đã đăng ký")
    try:
        roster = students_dataframe(adapter.list_students())
        st.dataframe(
            roster,
            width="stretch",
            height=280,
            hide_index=True,
            placeholder="Chưa có sinh viên trong danh sách.",
        )
    except (PipelineUnavailable, PipelineCallError) as exc:
        st.warning(str(exc), icon=":material/warning:")


def load_history(adapter: PipelineAdapter) -> None:
    try:
        st.session_state.history_records = adapter.history(limit=5000)
        st.session_state.history_loaded = True
        st.session_state.history_message = ""
    except (PipelineUnavailable, PipelineCallError) as exc:
        fallback = list(st.session_state.latest_events or [])
        st.session_state.history_records = fallback
        st.session_state.history_loaded = bool(fallback)
        st.session_state.history_message = str(exc)


def render_history(adapter: PipelineAdapter) -> None:
    st.subheader("Lịch sử & báo cáo")
    st.caption("Lọc sự kiện đã lưu, sau đó tải báo cáo dùng được với Excel.")

    with st.container(horizontal=True, horizontal_alignment="right"):
        refresh_clicked = st.button(
            "Làm mới dữ liệu",
            icon=":material/refresh:",
            key="refresh_history",
            width="content",
        )
    if refresh_clicked or not st.session_state.history_loaded:
        load_history(adapter)
    if st.session_state.history_message:
        st.warning(
            f"Đang hiển thị dữ liệu gần nhất còn trong phiên. {st.session_state.history_message}",
            icon=":material/database_off:",
        )

    history = events_dataframe(st.session_state.history_records)
    filter_columns = st.columns([1.2, 1.0, 1.2], vertical_alignment="bottom")
    with filter_columns[0]:
        query = st.text_input(
            "Tìm theo MSSV hoặc họ tên",
            placeholder="Nhập MSSV hoặc tên…",
            key="history_query",
            width="stretch",
        )
    with filter_columns[1]:
        statuses = st.multiselect(
            "Trạng thái mũ",
            options=STATUS_OPTIONS,
            default=["Đội sai", "Không mũ"],
            key="history_statuses",
            width="stretch",
        )
    with filter_columns[2]:
        selected_dates = st.date_input(
            "Khoảng ngày",
            value=(date.today() - timedelta(days=30), date.today()),
            key="history_dates",
            width="stretch",
        )

    start_date: date | None = None
    end_date: date | None = None
    if isinstance(selected_dates, (tuple, list)):
        if len(selected_dates) >= 1:
            start_date = selected_dates[0]
        if len(selected_dates) >= 2:
            end_date = selected_dates[1]
    elif isinstance(selected_dates, date):
        start_date = end_date = selected_dates

    filtered = filter_events(
        history,
        student_query=query,
        statuses=statuses,
        start_date=start_date,
        end_date=end_date,
    )
    st.caption(f"Hiển thị {len(filtered):,} / {len(history):,} sự kiện".replace(",", "."))
    st.dataframe(
        filtered,
        width="stretch",
        height=480,
        hide_index=True,
        column_config=events_column_config(),
        placeholder="Không có sự kiện phù hợp với bộ lọc.",
    )

    evidence_options: list[tuple[str, Path]] = []
    evidence_root = (PROJECT_ROOT / "outputs" / "classroom").resolve()
    if not filtered.empty and "Ảnh minh chứng" in filtered.columns:
        for row_index, row in filtered.iterrows():
            raw_path = str(row.get("Ảnh minh chứng") or "").strip()
            if not raw_path:
                continue
            try:
                evidence_path = Path(raw_path).resolve()
            except OSError:
                continue
            if not evidence_path.is_relative_to(evidence_root) or not evidence_path.is_file():
                continue
            label = (
                f"{row.get('MSSV') or 'Chưa rõ'} · "
                f"{row.get('Trạng thái mũ') or ''} · dòng {row_index + 1}"
            )
            evidence_options.append((label, evidence_path))
    if evidence_options:
        selected_label = st.selectbox(
            "Xem ảnh minh chứng",
            options=[label for label, _path in evidence_options],
            key="history_evidence_selection",
            width="stretch",
        )
        selected_path = dict(evidence_options)[selected_label]
        st.image(
            str(selected_path),
            caption=selected_label,
            width="stretch",
        )

    csv_data = dataframe_to_csv_bytes(filtered)
    xlsx_data: bytes | None = None
    xlsx_error = ""
    try:
        xlsx_data = dataframe_to_xlsx_bytes(filtered)
    except RuntimeError as exc:
        xlsx_error = str(exc)

    stamp = date.today().isoformat()
    with st.container(horizontal=True, horizontal_alignment="left"):
        st.download_button(
            "Tải CSV",
            data=csv_data,
            file_name=f"lich_su_mu_bao_hiem_{stamp}.csv",
            mime="text/csv; charset=utf-8",
            icon=":material/download:",
            disabled=filtered.empty,
            on_click="ignore",
            width="stretch",
        )
        st.download_button(
            "Tải Excel",
            data=xlsx_data or b"",
            file_name=f"lich_su_mu_bao_hiem_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/table_view:",
            disabled=filtered.empty or xlsx_data is None,
            on_click="ignore",
            width="stretch",
        )
    if xlsx_error:
        st.caption(xlsx_error)


def sync_session_from_backend(adapter: PipelineAdapter) -> None:
    """Reattach a refreshed browser session to an already running backend."""

    try:
        snapshot = adapter.snapshot()
    except (PipelineUnavailable, PipelineCallError):
        return
    state = str(snapshot.stats.get("state") or "").strip().casefold()
    backend_running = bool(snapshot.stats.get("running")) or state in {
        "starting",
        "running",
        "reconnecting",
    }
    if backend_running:
        st.session_state.monitor_running = True
        cache_snapshot(snapshot)
    elif state in {"stopped", "failed"}:
        st.session_state.monitor_running = False


initialize_session_state()
adapter = get_pipeline_adapter(str(PROJECT_ROOT))
sync_session_from_backend(adapter)
start_clicked, stop_clicked, uploaded_video = render_sidebar(adapter)
handle_source_controls(
    adapter,
    start_clicked=start_clicked,
    stop_clicked=stop_clicked,
    video_upload=uploaded_video,
)

with st.container(horizontal=True, vertical_alignment="center"):
    st.title("Giám sát mũ bảo hiểm lớp học")
    if st.session_state.monitor_running:
        st.badge("Đang chạy", icon=":material/fiber_manual_record:", color="blue")
    else:
        st.badge("Sẵn sàng", icon=":material/health_and_safety:", color="blue")
st.caption(
    "Nhận diện MSSV và theo dõi bốn trạng thái: đội đúng, đội sai, không mũ, chưa rõ. "
    "Kết quả là công cụ hỗ trợ demo và cần người xem xác nhận."
)
show_runtime_message()

monitor_tab, registration_tab, history_tab = st.tabs(
    [
        ":material/videocam: Giám sát trực tiếp",
        ":material/person_add: Đăng ký sinh viên",
        ":material/history: Lịch sử & báo cáo",
    ],
    key="main_workspace_tabs",
    on_change="rerun",
    width="stretch",
)

if monitor_tab.open:
    with monitor_tab:
        live_monitor_fragment(adapter)
if registration_tab.open:
    with registration_tab:
        render_registration(adapter)
if history_tab.open:
    with history_tab:
        render_history(adapter)
