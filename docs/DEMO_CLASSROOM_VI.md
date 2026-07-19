# Runbook demo trong phòng

Tài liệu này dành cho buổi trình chiếu bằng laptop Windows và điện thoại Android
chạy **IP Webcam**. Hãy chạy thử toàn bộ luồng ít nhất một ngày trước buổi demo
và dùng tiêu chí Go/No-Go tại [`DEMO_READINESS_VI.md`](DEMO_READINESS_VI.md).

## 1. Cấu hình khuyến nghị

### Laptop

- Cắm sạc và chọn chế độ nguồn **Best performance**.
- Đóng game, trình duyệt nhiều tab và ứng dụng họp trực tuyến; máy chỉ có 8 GB RAM.
- NVIDIA GTX 1650 chạy các model ONNX qua DirectML. Không cần cài CUDA Toolkit.
- Giữ trống ít nhất 3 GB ổ C cho môi trường Python, ảnh và cơ sở dữ liệu.

### Điện thoại/IP Webcam

- Đặt điện thoại nằm ngang, camera sau, cố định bằng tripod.
- Bắt đầu với 1280×720, 15 FPS, chất lượng JPEG khoảng 70-80%.
- Tắt autofocus liên tục nếu hình bị “thở”; chạm lấy nét vào hàng người giữa.
- Cắm sạc, tắt tiết kiệm pin cho IP Webcam và giữ màn hình/app hoạt động.
- Đặt camera cao khoảng 1,5-1,7 m, hướng gần ngang mặt, ánh sáng chiếu từ phía camera.

Ứng dụng chấp nhận cả địa chỉ gốc như `http://192.168.1.25:8080` và địa chỉ
đầy đủ `http://192.168.1.25:8080/video`; địa chỉ gốc sẽ tự thêm `/video`.

## 2. Kiểm tra mạng trước buổi demo

1. Kết nối laptop và điện thoại vào cùng Wi-Fi.
2. Trong IP Webcam, bấm **Start server** và ghi lại địa chỉ IPv4 hiển thị ở
   cuối màn hình, ví dụ `http://192.168.1.25:8080`.
3. Mở địa chỉ đó bằng trình duyệt trên laptop. Nếu trang IP Webcam không mở,
   ứng dụng Python cũng không thể mở luồng.
4. Nhập địa chỉ vào giao diện demo, bấm **Kiểm tra nguồn video**; khi nhận được
   kích thước khung hình thật mới bấm **Bắt đầu**.
5. Chỉ coi nguồn đã hoạt động khi giao diện chuyển sang **Camera và mô hình
   đang trực tiếp** và số frame tăng. **Đang chờ khung hình đầu tiên** chỉ có
   nghĩa worker đã được mở.

Nhiều mạng trường bật AP/client isolation: các máy có Internet nhưng không được
phép kết nối trực tiếp với nhau. Chuẩn bị một trong hai phương án dự phòng:

- bật hotspot trên một điện thoại khác rồi nối cả laptop và điện thoại camera;
- dùng router/travel router riêng không cần Internet.

Không cần Internet để nhận diện; toàn bộ suy luận và lưu lịch sử diễn ra cục bộ.

## 3. Đăng ký khuôn mặt

Project đã có dữ liệu khởi động cho năm sinh viên:

| MSSV | Họ và tên | Số ảnh hiện có | Readiness |
|---|---|---:|---|
| CE182206 | Nguyễn Thị Bích Tuyền | 1 | Chưa sẵn sàng |
| CE190579 | Lê Thanh Điền | 21 | Sẵn sàng thử riêng |
| CE190625 | Lữ Phú Quý | 1 | Chưa sẵn sàng |
| CE191641 | Nguyễn Hiếu Thành | 1 | Chưa sẵn sàng |
| CE190256 | Trần Khoa Đăng | 1 | Chưa sẵn sàng |

Một ảnh/người không đủ tin cậy cho sân khấu thật. Hiện chỉ CE190579 đã có bộ
đa góc; không nên tuyên bố hệ thống sẵn sàng nhận diện đủ năm người cho đến khi
bốn sinh viên còn lại được bổ sung. Với từng sinh viên, vào
**Đăng ký sinh viên** và lưu 6-15 ảnh:

1. 3-5 ảnh chính diện, nét và đủ sáng.
2. 2 ảnh xoay nhẹ sang trái, 2 ảnh xoay nhẹ sang phải.
3. 1-2 ảnh hơi ngẩng/cúi.
4. 1-2 ảnh có đội loại mũ dùng trong demo nhưng vẫn nhìn rõ mặt.

Chỉ lưu ảnh khi giao diện báo phát hiện đúng một mặt. Sau khi thêm ảnh, bấm
**Nạp lại thư viện khuôn mặt** và thử từng người trong luồng trực tiếp.

Hệ thống từ chối ảnh quá nhỏ, ảnh không nhất quán với MSSV đang bổ sung hoặc
ảnh quá giống một MSSV khác. Không đổi tên thư mục để vượt qua kiểm tra này.

## 4. Ngưỡng nhận diện an toàn

- Giữ cosine threshold mặc định **0,50** và top-2 margin **0,10**.
- Danh tính có ít hơn 5 ảnh tham chiếu chịu ngưỡng nội bộ chặt hơn **0,55**.
- Mặt dùng để nhận diện trực tiếp cần đủ rõ; mục tiêu thực tế là rộng 80-100 px.
- Hệ thống cần nhiều phiếu đồng thuận trước khi hiện MSSV và xóa danh tính cũ
  sau các lần từ chối liên tiếp.
- Không hạ threshold để ép hệ thống nhận một người. Kết quả an toàn khi thiếu
  bằng chứng là **Chưa rõ**, rồi bổ sung dữ liệu/ánh sáng và thử lại.

## 5. Số người và khoảng cách

Pipeline có thể theo dõi nhiều người, nhưng giới hạn thật là số pixel trên mỗi
khuôn mặt. Để nhận diện ổn định, mặt nên rộng ít nhất khoảng 80-100 px.

- Kịch bản chắc chắn nhất: 4-6 người/lượt, đứng cách camera khoảng 2,5-4 m.
- Với 20 người trong phòng: chia thành bốn nhóm năm người để trình diễn lần lượt.
- Nếu bắt buộc thấy cả 20 người cùng lúc: thử 1920×1080 ở 10-15 FPS, bố trí hai
  hàng gần nhau và kiểm tra trước. Không nên hứa nhận diện đủ 20 người từ một
  khung hình rất rộng bằng một điện thoại.

## 6. Bốn trạng thái mũ

- **Đội đúng:** model thấy mũ và hộp mũ nằm đúng vùng đầu/mặt.
- **Không mũ:** model thấy lớp `no_helmet` phù hợp với vùng mặt/đầu.
- **Đội sai:** có mũ gần người nhưng vị trí/độ phủ lệch khỏi vùng đầu dự kiến.
- **Chưa rõ:** bằng chứng chưa đủ hoặc các frame mâu thuẫn.

Checkpoint gốc không có lớp huấn luyện riêng cho “đội sai”. Trạng thái này là
heuristic hình học có làm mượt theo thời gian. Nó phù hợp để minh họa mũ đặt
lệch, đội quá ngửa hoặc cầm sát đầu; nó **không kiểm tra dây quai** và không thể
kết luận dây đã cài đúng ở khoảng cách xa. Khi `helmet`/`no_helmet` mâu thuẫn,
không có mặt phù hợp hoặc vị trí mũ nằm trong vùng biên, hệ thống trả **Chưa rõ**
thay vì chọn lớp có điểm cao hơn một cách cưỡng ép.

## 7. Kịch bản trình bày 5 phút

1. Chạy `CHECK_DEMO.bat`, sau đó mở **Giám sát trực tiếp** và cho thấy readiness.
2. Mời hai sinh viên không mũ vào khung; chỉ MSSV khi nhận diện đã ổn định.
3. Một người đội mũ đúng, một người đặt mũ lệch; chờ 1-2 giây để làm mượt nhãn.
4. Mở **Lịch sử & báo cáo**, cho thấy thời gian, MSSV, trạng thái và ảnh bằng chứng.
5. Lọc theo MSSV/trạng thái rồi tải CSV và Excel.
6. Mở **Đăng ký sinh viên**, lưu thêm một ảnh để chứng minh hệ thống có thể mở
   rộng đến sinh viên mới mà không sửa mã nguồn.

## 8. Xử lý sự cố nhanh

| Hiện tượng | Cách xử lý |
|---|---|
| Không mở được IP Webcam từ laptop | Kiểm tra `/video`, tắt VPN/firewall tạm thời theo chính sách của máy, hoặc dùng hotspot/router dự phòng. |
| Giao diện đứng ở “Đang chờ khung hình đầu tiên” | Chờ tối đa khoảng 15 giây; kiểm tra URL/trình duyệt. Nếu chuyển “Lỗi nguồn hoặc mô hình”, sửa nguồn rồi bấm lại. |
| Video trễ nhiều giây | Giảm còn 1280×720/10-15 FPS, chất lượng 70%, đứng gần access point. |
| DirectML không xuất hiện | Chạy `scripts\preflight.py`, cập nhật driver NVIDIA; ứng dụng vẫn có thể chạy CPU nhưng chậm hơn. |
| Hiện “Chưa rõ” thay vì MSSV | Đây là fail-closed khi bằng chứng yếu/mâu thuẫn; đưa mặt gần hơn, tăng sáng hoặc bổ sung ảnh. Không hạ ngưỡng. |
| Nhận nhầm MSSV | Dừng kịch bản, bổ sung ảnh đúng người, tránh ngược sáng, bảo đảm mặt >80 px và diễn tập lại; không hạ ngưỡng. |
| Nhãn mũ nhấp nháy | Đứng yên 1-2 giây, tăng ánh sáng, không che vùng đầu; tăng cửa sổ làm mượt trong cấu hình nâng cao. |
| Hết RAM | Đóng ứng dụng khác, dùng 720p, giảm số người/lượt; không chạy pipeline PyTorch cũ song song. |
| Port 8501 đã dùng | Dừng Streamlit cũ hoặc chạy PowerShell với một port khác bằng lệnh `.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8502`. |

## 9. Dữ liệu và quyền riêng tư

- Xin sự đồng ý của người tham gia trước khi thu ảnh khuôn mặt.
- Không chiếu hoặc phát tán ảnh/MSSV ngoài mục đích đã thống nhất.
- Dữ liệu nằm trong `data/students` và `outputs/classroom`; không tự gửi lên cloud.
- Sao lưu báo cáo cần thiết rồi xóa dữ liệu sinh trắc học/sự kiện theo thời hạn
  mà nhóm và nhà trường đã thống nhất.
