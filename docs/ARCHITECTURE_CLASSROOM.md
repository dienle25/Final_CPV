# Kiến trúc bản demo lớp học

```text
IP Webcam / webcam / video
          |
          v
Capture worker (latest-frame, reconnect, backoff)
          |
          +------------------------------+
          |                              |
          v                              v
NanoDet COCO: person             YuNet + SFace
          |                      face + embedding
          +---------------+--------------+
                          v
                 gán face vào person
                          |
                          v
           YOLO helmet/no_helmet (ONNX)
                          |
                          v
       gán head box + heuristic đội đúng/sai
                          |
                          v
            IoU tracking + temporal voting
                          |
              +-----------+------------+
              |                        |
              v                        v
     frame có khung/nhãn        SQLite + JPG snapshot
              |                        |
              +-----------+------------+
                          v
          Streamlit + lịch sử + CSV/XLSX
```

## Quyết định kỹ thuật

- Runtime chính dùng ONNX Runtime DirectML trên Windows để tận dụng GTX 1650 mà
  không cần mang theo PyTorch/CUDA nặng.
- Capture và inference chạy nền; giao diện chỉ đọc snapshot trạng thái. Hàng đợi
  chỉ giữ frame mới nhất để độ trễ không tăng dần khi xử lý chậm hơn camera.
- Phiên DirectML chạy tuần tự và có khóa vì một session DirectML không được gọi
  đồng thời từ nhiều thread.
- ID chỉ được công bố khi độ tương đồng SFace đạt ngưỡng và hơn ứng viên thứ hai
  một khoảng an toàn. Kết quả còn được bỏ phiếu qua nhiều frame.
- Mỗi event được ghi trong giao dịch SQLite và có cooldown để không tạo hàng
  chục bản ghi giống nhau cho cùng người/trạng thái.
- Ảnh bằng chứng, database và báo cáo đều nằm dưới `outputs/classroom`, nên có
  thể sao lưu hoặc di chuyển cả thư mục project như một đơn vị.

## Ranh giới với pipeline xe máy cũ

Các file `src/detect.py`, `src/association.py`, `src/violation_logger.py` và
`app/demo_app.py` được giữ để tham khảo/kế thừa bài toán ngoài đường. Bản lớp học
chạy từ `streamlit_app.py` và package `src/classroom_demo`; nó không dùng lớp
`rider` của checkpoint để thay cho người trong phòng.
