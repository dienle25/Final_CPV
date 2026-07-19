# Nhận diện sinh viên và mũ bảo hiểm trong lớp

Bản demo Windows nhận video trực tiếp từ điện thoại chạy **IP Webcam**, nhận
diện khuôn mặt thành MSSV/họ tên, phân loại trạng thái mũ, vẽ khung tiếng Việt,
lưu ảnh bằng chứng và xuất lịch sử CSV/Excel.

## Chức năng đã có

- Nguồn IP Webcam (`http://IP:8080` hoặc `http://IP:8080/video`), webcam USB và video file.
- Tự kết nối lại khi luồng mạng chập chờn; chỉ giữ frame mới nhất để không tăng độ trễ.
- NanoDet phát hiện người trong phòng, không dùng nhầm lớp `rider` của model xe máy.
- YuNet + SFace nhận diện MSSV theo cơ chế an toàn: cosine mặc định **0,50**,
  khoảng cách top-2 tối thiểu **0,10** và xác nhận qua nhiều frame.
- Model mũ ONNX ba lớp gốc `helmet / no_helmet / rider`; runtime lớp học chỉ dùng hai lớp đầu.
- Bốn nhãn hiển thị: **Đội đúng**, **Đội sai**, **Không mũ**, **Chưa rõ**;
  khi bằng chứng mâu thuẫn hoặc chưa đủ, hệ thống giữ **Chưa rõ** thay vì đoán.
- Tracking ID, làm mượt khuôn mặt/trạng thái mũ, cooldown chống ghi trùng sự kiện.
- SQLite, ảnh JPG bằng chứng, bộ lọc lịch sử, CSV UTF-8 BOM và Excel `.xlsx`.
- Giao diện Streamlit tiếng Việt cho giám sát, đăng ký sinh viên và báo cáo.
- ONNX Runtime DirectML tận dụng GTX 1650; tự lùi về CPU nếu DirectML không có.
- Màn hình readiness và preflight sâu kiểm tra Python, dependency, model, suy
  luận ONNX thật, thư viện ảnh và quyền ghi trước giờ demo.

## Dữ liệu đã nhập

| MSSV | Họ và tên | Ảnh tham chiếu |
|---|---|---:|
| CE182206 | Nguyễn Thị Bích Tuyền | 1 |
| CE190579 | Lê Thanh Điền | 21 |
| CE190625 | Lữ Phú Quý | 15 |
| CE191641 | Nguyễn Hiếu Thành | 15 |
| CE190256 | Trần Khoa Đăng | 15 |

CE190579 đã có 1 ảnh thẻ và 20 khung đa góc từ video đăng ký. Bốn sinh viên còn
lại chỉ có 1 ảnh chân dung/người, vì vậy **chưa sẵn sàng cho demo nhận diện năm
người**. Trước demo cần đăng ký thêm 6-15 ảnh cho mỗi người trong chính căn
phòng, ở khoảng cách và ánh sáng sẽ sử dụng. Giao diện readiness sẽ tiếp tục
cảnh báo các MSSV có ít ảnh.

## Cài đặt nhanh trên Windows

Yêu cầu:

- Windows 10/11 64-bit;
- Python 3.11 hoặc 3.12 64-bit từ `python.org` (chọn **Add Python to PATH**);
- khoảng 3 GB dung lượng trống trong lúc cài;
- driver NVIDIA hiện hành nếu muốn dùng DirectML.

Mở PowerShell tại thư mục project:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Script tạo `.venv`, cài đúng bộ phiên bản trong `requirements-classroom.txt` và
chạy preflight. Không cần cài PyTorch, CUDA Toolkit, PaddleOCR hoặc Visual Studio.

Trước mỗi lần diễn tập hoặc trình chiếu, nhấp đúp `CHECK_DEMO.bat` (hoặc chạy
trong PowerShell) và chỉ tiếp tục khi kết quả không có lỗi:

```powershell
.\CHECK_DEMO.bat
```

Khởi động bằng một trong hai cách:

```powershell
.\scripts\run_classroom_demo.ps1
```

hoặc nhấp đúp:

```text
RUN_CLASSROOM_DEMO.bat
```

Giao diện mở tại `http://localhost:8501`.

## Kết nối IP Webcam

1. Kết nối laptop và điện thoại vào cùng Wi-Fi.
2. Mở IP Webcam trên Android, chọn camera sau, 1280×720 và khoảng 15 FPS.
3. Bấm **Start server**; app hiển thị địa chỉ như `http://192.168.1.25:8080`.
4. Mở địa chỉ đó trong trình duyệt laptop để xác nhận hai thiết bị nhìn thấy nhau.
5. Trong sidebar của demo chọn **IP Webcam**, nhập địa chỉ rồi bấm **Bắt đầu**.
6. Chờ trạng thái chuyển sang **Camera và mô hình đang trực tiếp**; dòng
   **Đang chờ khung hình đầu tiên** chưa có nghĩa camera đã gửi được frame.

Mạng trường có thể bật client isolation dù hai thiết bị cùng Wi-Fi. Hãy chuẩn bị
hotspot hoặc router riêng làm phương án dự phòng. Xem runbook chi tiết:
[`docs/DEMO_CLASSROOM_VI.md`](docs/DEMO_CLASSROOM_VI.md).

## Luồng sử dụng

### Giám sát trực tiếp

- Chọn nguồn và thiết bị xử lý ở sidebar. Giữ ngưỡng an toàn mặc định
  **0,50** và margin **0,10**; không hạ ngưỡng để ép hiện MSSV.
- Bấm **Kiểm tra nguồn video** để xác nhận nhận được khung hình thật, sau đó
  bấm **Bắt đầu** và chờ nhãn ổn định qua vài frame.
- Video hiển thị tracking ID, MSSV/họ tên và trạng thái mũ.
- KPI và bảng người hiện tại cập nhật độc lập với phần cấu hình.

Nhận diện dùng nguyên tắc **fail-closed**: mặt quá nhỏ/mờ, điểm hai MSSV quá
gần nhau, hai người cùng tranh chấp một MSSV, hoặc kết quả bị từ chối liên tiếp
sẽ trả về **Chưa rõ**. Đây là hành vi an toàn có chủ ý, không phải lỗi giao diện.

### Đăng ký sinh viên

- Chọn MSSV đã có hoặc nhập MSSV/họ tên mới.
- Tải nhiều ảnh hoặc lấy frame rõ từ camera.
- Hệ thống chỉ nhận ảnh có đúng một khuôn mặt đủ rõ, lưu cục bộ dưới
  `data/students/<MSSV>/` và nạp lại thư viện nhận diện.

### Lịch sử & báo cáo

- Lọc theo thời gian, MSSV và trạng thái.
- Xem ảnh bằng chứng và thời điểm phát hiện.
- Tải CSV có UTF-8 BOM để Excel mở đúng tiếng Việt hoặc tải trực tiếp `.xlsx`.

## Ý nghĩa trạng thái “Đội sai”

Checkpoint được cung cấp chỉ có:

```text
0 helmet
1 no_helmet
2 rider
```

Nó không được huấn luyện lớp `incorrect_helmet`. Vì vậy bản demo suy ra **Đội
sai** bằng heuristic hình học về vị trí và độ phủ của hộp mũ so với mặt/vùng
đầu, rồi làm mượt theo thời gian. Cách này minh họa được mũ đội lệch, quá ngửa
hoặc cầm sát đầu, nhưng **không kiểm tra được dây quai đã cài đúng hay chưa**.
Khi hộp `helmet` và `no_helmet` mâu thuẫn hoặc hình học nằm ở vùng biên, hệ thống
giữ **Chưa rõ**. Báo cáo/bảo vệ cần nói đúng giới hạn này; muốn đánh giá khoa
học phải thu và gán nhãn dữ liệu ba lớp trong phòng thật rồi huấn luyện model
head-level mới.

## 20 người trong phòng

Code không đặt giới hạn cứng 5 người, nhưng độ chính xác khuôn mặt phụ thuộc số
pixel. Một camera điện thoại khung rất rộng thường không giữ được mặt rộng
80-100 px cho cả 20 người. Kịch bản chắc chắn cho demo là bốn lượt, mỗi lượt
4-6 người. Nếu buộc hiển thị 20 người một lúc, dùng 1080p/10-15 FPS, ánh sáng
tốt và thử trước đúng vị trí; hai camera sẽ đáng tin cậy hơn một camera.

## Dữ liệu đầu ra

Runtime tạo dữ liệu theo phiên dưới `outputs/classroom/`:

```text
outputs/classroom/
├── db/
│   └── events.db
├── snapshots/
│   └── ...jpg                  chỉ cho Đội sai/Không mũ khi bật lưu ảnh
└── exports/
    ├── classroom_events.csv   khi gọi export từ backend
    └── classroom_events.xlsx
```

Nút tải trong giao diện trả tệp `lich_su_mu_bao_hiem_<ngày>.csv/.xlsx` qua
trình duyệt, nên chúng nằm trong thư mục Downloads do trình duyệt chọn chứ
không tự xuất hiện dưới `outputs/classroom/exports`.

Ứng dụng không xóa lịch sử khi khởi động lại. Muốn xóa phải thao tác có chủ ý
và nên sao lưu báo cáo trước.

## Kiểm tra

```powershell
.\CHECK_DEMO.bat
.\.venv\Scripts\python.exe -m pytest -q
```

Preflight chấp nhận chạy tiếp khi chỉ còn cảnh báo dữ liệu, nhưng cảnh báo “mới
có 1 ảnh” vẫn có nghĩa sinh viên đó chưa đủ dữ liệu cho kịch bản demo đáng tin.
Preflight không chấp nhận thiếu model, dependency, checksum sai, model ONNX
không nạp/suy luận được hoặc thiếu quyền ghi. Bộ
pytest kiểm tra URL IP Webcam, capture/reconnect, SQLite, CSV/XLSX, hình học,
tracker, face matcher và các adapter detector bằng session giả có kiểm soát.
Trước khi bàn giao, bốn model ONNX đã được chạy suy luận thật trên năm ảnh mẫu;
đây là bước xác minh release, không phải benchmark độ chính xác khoa học.

## Cấu trúc chính

```text
streamlit_app.py                 giao diện lớp học mới
src/classroom_demo/
  detectors.py                  ONNX helmet/person + YuNet
  face_recognition.py           SFace, roster và gallery
  tracking.py                   tracker nhẹ
  helmet_status.py              gán head + bốn trạng thái
  sources.py                    IP Webcam/webcam/video + reconnect
  pipeline.py                   worker suy luận và snapshot UI
  storage.py                    SQLite/JPG/CSV/XLSX
  ui_helpers.py                 adapter và helper Streamlit
data/students.csv               danh sách MSSV
data/students/<MSSV>/           ảnh đăng ký cục bộ
models/                         model ONNX, checkpoint gốc và license
scripts/preflight.py            kiểm tra trước demo
CHECK_DEMO.bat                  kiểm tra readiness một lần nhấp
scripts/setup_windows.ps1       cài môi trường
scripts/run_classroom_demo.ps1  chạy giao diện
docs/DEMO_CLASSROOM_VI.md       runbook buổi trình chiếu
docs/DEMO_READINESS_VI.md       tiêu chí Go/No-Go trước khi trình chiếu
```

Pipeline xe máy cũ trong `src/detect.py` và `app/demo_app.py` vẫn được giữ để
tham khảo, nhưng không phải entrypoint của demo lớp học.

## Quyền riêng tư và giấy phép

Ảnh khuôn mặt và MSSV là dữ liệu cá nhân/sinh trắc học: cần sự đồng ý của người
tham gia, không đưa lên Git công khai và xóa theo thời hạn đã thống nhất. `.gitignore`
đã chặn mặc định các ảnh đăng ký mới và lịch sử runtime.

Nguồn/license model được ghi trong `NOTICE.md`, các tệp license nằm cạnh model
và checksum nằm trong `models/MODEL_MANIFEST.json`. Quyền phân phối checkpoint
mũ/dataset gốc vẫn cần chủ project xác nhận trước khi xuất bản công khai.
