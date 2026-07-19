# Readiness và tiêu chí Go/No-Go cho demo

Tài liệu này là checklist quyết định có nên trình diễn hay không. Không dùng
việc ứng dụng mở được trang web làm bằng chứng rằng camera, model và dữ liệu đã
sẵn sàng.

## Trạng thái dữ liệu hiện tại

| MSSV | Họ và tên | Ảnh tham chiếu | Kết luận hiện tại |
|---|---|---:|---|
| CE190579 | Lê Thanh Điền | 21 | Có thể diễn tập riêng |
| CE182206 | Nguyễn Thị Bích Tuyền | 1 | Chưa sẵn sàng demo nhận diện |
| CE190625 | Lữ Phú Quý | 1 | Chưa sẵn sàng demo nhận diện |
| CE191641 | Nguyễn Hiếu Thành | 1 | Chưa sẵn sàng demo nhận diện |
| CE190256 | Trần Khoa Đăng | 1 | Chưa sẵn sàng demo nhận diện |

Bốn sinh viên có một ảnh phải được bổ sung 6-15 ảnh/người trong đúng phòng,
ánh sáng, khoảng cách và camera dự kiến. Một ảnh có thể tạo embedding nhưng
không đủ để cam kết nhận diện đa góc.

## Kiểm tra bắt buộc trước buổi demo

1. Mở PowerShell tại đúng thư mục project và chạy:

   ```powershell
   .\CHECK_DEMO.bat
   ```

2. Kết quả phải có **0 lỗi**. Cảnh báo ít ảnh không làm hỏng phần mềm nhưng là
   No-Go cho sinh viên tương ứng.
3. Mở IP Webcam trong trình duyệt laptop, bấm **Kiểm tra nguồn video** trong
   app, sau đó bấm **Bắt đầu** và chờ trạng thái **Camera và mô hình đang trực
   tiếp**. Xác nhận số frame tăng liên tục ít nhất 2 phút.
4. Thử từng sinh viên riêng, rồi thử theo nhóm đúng vị trí sẽ trình diễn. Mỗi
   người cần được nhận đúng qua nhiều góc; không chấp nhận chỉ đúng một lần.
5. Đưa một người chưa đăng ký vào khung. Kết quả mong đợi là **Chưa rõ**, không
   phải MSSV gần nhất. Giữ threshold **0,50**, margin **0,10**.
6. Thử mũ thật ở ba tình huống: đội đúng, không mũ và đặt/đội lệch rõ. Đồng thời
   thử một góc khó để xác nhận hệ thống trả **Chưa rõ** khi bằng chứng mâu thuẫn.
7. Mở lịch sử, xem được ảnh bằng chứng, lọc theo MSSV/trạng thái và tải thử cả
   CSV lẫn Excel.
8. Dừng rồi khởi động lại nguồn một lần; xác nhận app không giữ nhãn MSSV cũ.
9. Chuẩn bị hotspot/router dự phòng và một video test cục bộ đã kiểm tra.

## Quy tắc an toàn của bản hardening

- Nhận diện mặc định dùng cosine **0,50** và khoảng cách top-2 **0,10**; identity
  có ít hơn 5 ảnh tham chiếu dùng ngưỡng tối thiểu **0,55**.
- Mặt nhỏ, mờ, không đủ phiếu; hai ứng viên quá gần nhau; hai track tranh chấp
  cùng MSSV; hoặc nhiều lần từ chối liên tiếp đều dẫn đến **Chưa rõ**.
- `helmet` và `no_helmet` chồng lấn/mâu thuẫn, không gắn được mặt phù hợp hoặc
  hình học mũ nằm ở vùng biên cũng dẫn đến **Chưa rõ**.
- **Đội sai** là heuristic hình học về vị trí/độ phủ của mũ. Hệ thống không có
  model kiểm dây quai và không được trình bày như có khả năng đó.

## Quyết định Go/No-Go

**Go cho demo đủ năm sinh viên** chỉ khi mọi điều sau đều đạt:

- `CHECK_DEMO.bat` không có lỗi;
- cả năm sinh viên có bộ ảnh đa góc và đã qua rehearsal tại phòng;
- người chưa đăng ký không bị gán nhầm MSSV trong clip thử;
- ba tình huống mũ thật đã được thử bằng chính camera/vị trí demo;
- nguồn giữ trạng thái trực tiếp ổn định, lịch sử và xuất file hoạt động;
- có nguồn mạng và video dự phòng.

**Go có điều kiện**: hiện tại có thể trình diễn riêng CE190579 bằng video/luồng
đã kiểm thử, đồng thời nói rõ bốn MSSV còn lại chưa đủ dữ liệu.

**No-Go** nếu thiếu model/dependency, không nhận frame, người lạ bị nhận thành
MSSV, sinh viên mục tiêu còn một ảnh, hoặc chưa thử mũ thật trong phòng.

Không có hệ thống thị giác nào bảo đảm tuyệt đối không sai. Mục tiêu của bản
hardening là từ chối đoán khi thiếu bằng chứng và làm cho trạng thái chưa sẵn
sàng hiển thị rõ trước khi người trình bày bấm bắt đầu.
