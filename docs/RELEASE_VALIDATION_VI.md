# Biên bản bàn giao và xác minh demo lớp học

Ngày xác minh: **19/07/2026**  
Môi trường: Windows 11, Python 3.12.13, Ryzen 7 4800H, RAM 8 GB,
NVIDIA GTX 1650 4 GB, ONNX Runtime DirectML.

## Kết quả bàn giao

Project đã có luồng demo hoàn chỉnh:

1. nhận video từ IP Webcam, webcam máy tính hoặc video tải lên;
2. phát hiện người và khuôn mặt;
3. nhận diện MSSV/họ tên từ gallery sinh viên;
4. theo dõi từng người giữa các khung hình;
5. hiển thị **Đội đúng / Đội sai / Không mũ / Chưa rõ**;
6. lưu lịch sử SQLite, chỉ lưu JPG cho **Đội sai/Không mũ** khi bật tùy chọn;
7. lọc lịch sử, tải CSV/XLSX;
8. đăng ký sinh viên mới từ ảnh tải lên hoặc khung hình hiện tại;
9. giao diện Streamlit tiếng Việt và script cài/chạy một lần trên Windows.

Bản hardening dùng mặc định cosine **0,50**, top-2 margin **0,10**, xác nhận
nhiều frame và xóa danh tính cũ sau các lần từ chối liên tiếp. Danh tính có ít
hơn 5 ảnh tham chiếu chịu ngưỡng tối thiểu 0,55. Khi bằng chứng khuôn mặt hoặc mũ
mâu thuẫn, kết quả là **Chưa rõ** (fail-closed), không ép chọn MSSV/trạng thái.

Danh sách có sẵn:

| MSSV | Họ và tên | Ảnh | Readiness |
|---|---|---:|---|
| CE182206 | Nguyễn Thị Bích Tuyền | 1 | Chưa sẵn sàng |
| CE190579 | Lê Thanh Điền | 21 | Sẵn sàng thử riêng |
| CE190625 | Lữ Phú Quý | 1 | Chưa sẵn sàng |
| CE191641 | Nguyễn Hiếu Thành | 1 | Chưa sẵn sàng |
| CE190256 | Trần Khoa Đăng | 1 | Chưa sẵn sàng |

## Xác minh kỹ thuật

- pytest: **52/52 đạt**.
- Streamlit AppTest: **0 exception**; tab đăng ký hiển thị đủ 5 sinh viên ngay
  cả khi chưa bật camera.
- Preflight: **0 lỗi, 4 cảnh báo**. CE190579 có 21 ảnh tham chiếu; bốn
  sinh viên còn lại hiện có 1 ảnh/người.
- pip check, compileall và phân tích cú pháp PowerShell: đạt.
- DirectML và CPU provider đều được phát hiện.
- SHA-256 của best.pt, best.onnx, NanoDet, YuNet và SFace khớp manifest.
- Bốn model ONNX đều nạp được bằng ONNX Runtime.
- Smoke test suy luận thật trên montage năm ảnh: phát hiện đủ năm khuôn mặt/người
  và SFace trả đúng cả năm MSSV.
- Smoke test end-to-end 30 khung hình trên montage: xử lý đủ 30 khung hình,
  khoảng **5,25 FPS** trên máy đích. Đây là kiểm tra tích hợp, không phải đo độ
  chính xác khoa học trong phòng thật.
- Video riêng của CE190579 đã được kiểm thử bằng YuNet/SFace thật trên 56 mẫu:
  phát hiện mặt **56/56**, xếp hạng nhất đúng **56/56**, không có MSSV sai.
  Cả ngưỡng cũ `0,42 / margin 0,05` và ngưỡng an toàn hiện tại
  `0,50 / margin 0,10` đều nhận đúng 56/56.
- Luồng đầy đủ trên video test CE190579 với DirectML: 56/56 khung có người và
  mặt; 4 quan sát đầu giữ **Chưa rõ**, sau đó 52/52 quan sát mang nhãn
  CE190579, nhận ổn định lần đầu tại khoảng 1,34 giây và không có ID sai.
  Trạng thái gồm 51 **Không mũ** và 5 **Chưa rõ**; tốc độ hiệu dụng khoảng
  **3,91 FPS** (trung vị 175 ms, P95 271 ms mỗi khung).
- Đã sửa luồng đọc video điện thoại để tự áp dụng metadata xoay. Hai clip
  CE190579 có metadata 270° và nay được đưa về khung đứng 848×480 trước suy luận.
- Start → Stop → Start lại, đổi nguồn/thiết bị/ngưỡng, EOF video, lưu lịch sử
  qua lần mở app mới và bật/tắt ảnh minh chứng đều đã được kiểm thử.
- `CHECK_DEMO.bat` cung cấp đường chạy preflight một lần nhấp; giao diện phân
  biệt **đang kết nối**, **đang nhận hình**, **frame cũ/đang reconnect** và
  **kết nối thất bại**, tránh báo “live” trước frame đầu tiên.

Giao diện được xây theo Streamlit 1.59.2 với cache model, fragment cập nhật live,
theme native và worker nền để tránh khóa thao tác.

## Giới hạn cần nói đúng khi demo

- Checkpoint mũ gốc chỉ có ba lớp helmet, no_helmet, rider; không có lớp
  “đội sai”. Nhãn **Đội sai** hiện là heuristic hình học giữa mũ và vùng
  mặt/đầu. Hệ thống **không kiểm tra dây quai** và không thể xác nhận chắc chắn
  việc cài quai từ camera xa. Trường hợp biên/mâu thuẫn phải giữ **Chưa rõ**.
- CE190579 đã có 1 ảnh thẻ và 20 khung video đa góc. Bốn sinh viên còn lại vẫn
  chỉ có một ảnh chân dung; trước buổi demo nên đăng ký **6–15 ảnh/người**:
  chính diện, nghiêng trái/phải, ánh sáng và khoảng cách gần giống phòng demo.
- Một camera góc rộng khó giữ khuôn mặt đủ 80–100 px cho cả 20 người. Kịch bản
  ổn định là **4 lượt, mỗi lượt 4–6 người**; nếu bắt buộc 20 người cùng lúc,
  dùng 1080p, ánh sáng tốt và ưu tiên hai camera.
- Đây là công cụ hỗ trợ demo, không dùng làm căn cứ kỷ luật hoặc quyết định tự
  động.

## Chạy nhanh trên máy demo

Từ PowerShell tại thư mục project:

    powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
    .\CHECK_DEMO.bat
    .\RUN_CLASSROOM_DEMO.bat

Trên IP Webcam, chọn khoảng **1280×720, 10–15 FPS**, rồi nhập URL dạng:

    http://<IP-điện-thoại>:8080/video

Laptop và điện thoại phải truy cập được nhau trên cùng mạng. Hãy thử đầy đủ
camera, vị trí đứng, ánh sáng và năm người thật trước ngày trình bày.

## Quyền riêng tư và phát hành

ZIP này là gói **demo riêng tư** và có 25 ảnh tham chiếu khuôn mặt cùng MSSV
(21 ảnh CE190579 và 1 ảnh cho mỗi sinh viên còn lại). Không đăng công khai.
Quyền phân phối code/checkpoint gốc chưa được xác lập; chỉ đưa repo lên GitHub
Private theo GITHUB_UPLOAD.md cho đến khi hoàn tất giấy phép và sự đồng ý dữ
liệu.
