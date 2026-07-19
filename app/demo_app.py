"""Streamlit interface for the merged end-to-end demonstration."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import HelmetViolationProcessor  # noqa: E402


st.set_page_config(
    page_title="Helmet Violation Detection",
    page_icon="🪖",
    layout="wide",
)
st.title("Hệ thống phát hiện vi phạm quy định đội mũ bảo hiểm")
st.caption(
    "YOLOv8s · ByteTrack · rider–head association · temporal confirmation · "
    "SQLite · Streamlit"
)

with st.sidebar:
    st.header("Cấu hình")
    model_path = st.text_input("Đường dẫn model", "models/best.pt")
    confidence = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("NMS IoU", 0.20, 0.90, 0.45, 0.05)
    imgsz = st.select_slider("Inference image size", options=[416, 512, 640, 768], value=640)
    device = st.selectbox("Thiết bị", ["auto", "cpu", "0"], index=0)

    st.subheader("Xác nhận theo thời gian")
    history_size = st.slider("Số vote gần nhất", 4, 30, 12)
    min_votes = st.slider("Vote tối thiểu để ổn định", 2, 15, 4)
    vote_ratio = st.slider("Tỷ lệ majority tối thiểu", 0.50, 1.00, 0.60, 0.05)
    event_hits = st.slider("Frame ổn định để tạo event", 1, 10, 3)

    st.subheader("Tính năng tùy chọn")
    enable_ocr = st.checkbox("Bật PaddleOCR thử nghiệm", value=False)
    enable_email = st.checkbox("Bật SMTP email", value=False)
    st.warning(
        "Model hiện tại không có lớp biển số. OCR chỉ chạy trên vùng dưới của "
        "rider và có thể trả UNREAD; không xem đây là OCR đã được đánh giá."
    )

uploaded = st.file_uploader(
    "Tải video MP4/AVI/MOV/MKV",
    type=["mp4", "avi", "mov", "mkv"],
)
run_button = st.button(
    "Chạy demo end-to-end",
    type="primary",
    disabled=uploaded is None,
)

frame_box = st.empty()
metric_columns = st.columns(5)
progress = st.progress(0.0)
status = st.empty()

if run_button and uploaded is not None:
    resolved_model = PROJECT_ROOT / model_path
    if not resolved_model.exists():
        st.error(f"Không tìm thấy model: {resolved_model}")
        st.stop()

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(uploaded.getbuffer())
        input_path = Path(temporary_file.name)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "streamlit_runs" / run_id
    output_video = output_dir / "videos" / "streamlit_result.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    processor: HelmetViolationProcessor | None = None
    try:
        processor = HelmetViolationProcessor(
            model_path=str(resolved_model),
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            device=device,
            history_size=history_size,
            min_votes=min_votes,
            vote_ratio=vote_ratio,
            min_hits=event_hits,
            enable_ocr=enable_ocr,
            enable_email=enable_email,
            output_dir=output_dir,
            source_name=uploaded.name,
        )

        def update_ui(frame, stats):
            frame_box.image(frame, channels="BGR", use_container_width=True)
            total = stats["total_frames"]
            if total > 0:
                progress.progress(min(1.0, (stats["frame_index"] + 1) / total))
            metric_columns[0].metric("Frame", stats["frame_index"])
            metric_columns[1].metric("FPS xử lý", f"{stats['processing_fps']:.1f}")
            metric_columns[2].metric("Rider đang theo dõi", stats["active_rider_count"])
            metric_columns[3].metric("Vi phạm", stats["violation_count"])
            metric_columns[4].metric("Event mới", len(stats.get("new_events", [])))

        status.info("Đang xử lý video. Giữ tab này mở.")
        summary = processor.process_source(
            str(input_path),
            output_video=output_video,
            display=False,
            frame_callback=update_ui,
            callback_every=3,
        )
        violations = processor.logger.recent(limit=200)
        status.success("Đã xử lý xong.")

        st.subheader("Tóm tắt lần chạy")
        st.json(summary)
        st.subheader("Danh sách vi phạm")
        if violations:
            st.dataframe(pd.DataFrame(violations), use_container_width=True)
            st.write("Thư mục evidence:", str(output_dir / "violations"))
        else:
            st.warning(
                "Không tìm thấy vi phạm đã xác nhận. Có thể thử giảm confidence "
                "hoặc dùng video có rider không đội mũ rõ hơn."
            )

        if output_video.exists():
            st.video(str(output_video))
        csv_path = Path(summary["csv_path"])
        if csv_path.exists():
            st.download_button(
                "Tải violations.csv",
                data=csv_path.read_bytes(),
                file_name="violations.csv",
                mime="text/csv",
            )
    except Exception as exc:
        status.error(f"Demo lỗi: {type(exc).__name__}: {exc}")
        st.exception(exc)
    finally:
        if processor is not None:
            processor.close()
        input_path.unlink(missing_ok=True)
