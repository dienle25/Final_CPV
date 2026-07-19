# Xác minh nhận diện CE190579 – Lê Thanh Điền

Ngày kiểm thử: **19/07/2026**  
Máy kiểm thử: Windows 11, Ryzen 7 4800H, RAM 8 GB, ONNX Runtime DirectML.

## Dữ liệu đã dùng

- Video đăng ký: `CE190579_enroll.mp4`, 34,73 giây, 848×480, 29,77 FPS.
- Video test: `CE190579_test.mp4`, 18,71 giây, 848×480, 29,77 FPS.
- Hai video mang metadata xoay 270°. Luồng đọc camera/video đã được sửa để tự
  áp dụng metadata và đưa ảnh về khung đứng 848×480 trước khi suy luận.
- Gallery hiện có **21 ảnh** cho CE190579: 1 ảnh thẻ ban đầu và 20 khung hình
  đa góc lấy từ video đăng ký. YuNet tìm đúng một khuôn mặt trong cả 20 khung;
  điểm phát hiện thấp nhất là 0,852 ở ngưỡng 0,80.

## Kết quả nhận diện khuôn mặt

Video test được lấy mẫu 56 khung, xấp xỉ 3 FPS, và chạy bằng YuNet/SFace thật:

| Chỉ số | Kết quả |
|---|---:|
| Phát hiện được mặt | 56/56 |
| Xếp hạng nhất là CE190579 | 56/56 |
| MSSV sai | 0 |
| Nhận đúng ở ngưỡng 0,42 / margin 0,05 | 56/56 |
| Nhận đúng ở ngưỡng chặt 0,50 / margin 0,10 | 56/56 |
| Cosine similarity nhỏ nhất / trung vị | 0,5780 / 0,7165 |
| Margin nhỏ nhất / trung vị | 0,3438 / 0,5083 |

Ngưỡng vận hành mặc định của bản hardening là **0,50 / margin 0,10**. Không
dùng lại 0,42 / 0,05 cho demo, dù bộ video CE190579 vẫn đạt ở mức đó. Mặt live
còn phải vượt kiểm tra chất lượng/kích thước và nhiều phiếu đồng thuận.

## Kết quả luồng đầy đủ

Pipeline người + mặt + nhận diện + mũ + theo dõi được chạy bằng DirectML trên
cùng 56 khung:

- Có người: **56/56**; có mặt: **56/56**.
- Bốn quan sát đầu hiển thị **Chưa rõ** theo cơ chế bỏ phiếu chống đoán vội.
- Sau giai đoạn xác nhận: **52/52** mang nhãn `CE190579`; không có ID sai.
- Nhận ổn định lần đầu tại khoảng **1,34 giây**.
- Trạng thái mũ: **51 Không mũ**, **5 Chưa rõ**.
- Tốc độ hiệu dụng: khoảng **3,91 FPS**; thời gian xử lý trung vị 175 ms,
  P95 271 ms trên máy kiểm thử.

Toàn project đồng thời đạt **52/52 test**, `pip check` và `compileall`. Preflight
đạt **0 lỗi, 4 cảnh báo**; bốn cảnh báo là bốn sinh viên khác vẫn chỉ có một
ảnh tham chiếu.

## Cách thử ngay

1. Đóng phiên app cũ bằng `Ctrl+C` để bỏ gallery đang nằm trong bộ nhớ.
2. Mở PowerShell tại đúng thư mục project đã giải nén.
3. Chạy `./CHECK_DEMO.bat`; chỉ khi không có lỗi mới chạy
   `./RUN_CLASSROOM_DEMO.bat`.
4. Trong app, chọn video tải lên hoặc IP Webcam, giữ thiết bị đúng chiều và để
   khuôn mặt có kích thước ít nhất tương đương video test (trung vị khoảng
   105 px chiều cao).
5. Chờ khoảng 1–2 giây để nhãn ổn định; không giảm ngưỡng để ép hệ thống đoán.

## Giới hạn cần nói rõ

- Kết quả trên là với đúng video test người dùng đã cung cấp. Bộ 20 ảnh đăng ký
  được chọn sau khi phân tích bộ test này, vì vậy số 56/56 có thể lạc quan hơn
  một video hoàn toàn mới.
- Chưa có video chuyên dụng của người lạ/không đăng ký nên chưa thể đo tỷ lệ
  nhận nhầm ngoài danh sách theo đúng điều kiện phòng. Một phép quét proxy trên
  458 khuôn mặt từ ảnh dataset mũ không chấp nhận nhầm trường hợp nào ở ngưỡng
  0,50 / margin 0,10, nhưng không thay thế clip người lạ thật. Khi không đủ chắc
  chắn, hệ thống phải giữ nhãn **Chưa rõ**.
- Video CE190579 hiện không có các trường hợp đội đúng và đội sai, nên clip này
  chỉ xác minh trạng thái **Không mũ**. Nhãn **Đội sai** của project là heuristic
  hình học, không kiểm tra được dây quai và không thể xác nhận chắc chắn việc
  cài quai từ camera xa.
- Không thể cam kết tuyệt đối không sai trong mọi ánh sáng, góc mặt và khoảng
  cách. Cần rehearsal tại đúng phòng, đúng vị trí điện thoại trước khi demo.
