# Đưa mã nguồn lên GitHub an toàn

> **Chỉ dùng repository Private ở trạng thái hiện tại.** Project chưa có
> software license cấp cao nhất và quyền phân phối checkpoint `best.pt`/
> `best.onnx` chưa được xác lập. 25 ảnh tham chiếu của năm sinh viên cùng MSSV
> là dữ liệu cá nhân/sinh trắc học, tuyệt đối không push nếu chưa có sự đồng ý
> phù hợp.

Entrypoint demo lớp học là `streamlit_app.py`. ZIP bàn giao riêng tư cố ý chứa
25 ảnh tham chiếu để chạy demo; Git lại cố ý bỏ qua các ảnh này bằng
`.gitignore`.

## 1. Tạo repository

1. Trên GitHub, chọn **New repository**.
2. Đặt tên, ví dụ `helmet-classroom-demo`.
3. Chọn **Private**.
4. Không tạo README hoặc `.gitignore` từ GitHub vì project đã có hai tệp này.
5. Không chọn license cho đến khi chủ sở hữu xác nhận quyền với toàn bộ code và
   checkpoint.

## 2. Kiểm tra trước khi commit

Mở PowerShell tại thư mục project:

```powershell
.\.venv\Scripts\python.exe scripts\preflight.py
.\.venv\Scripts\python.exe -m pytest -q
git --version
Get-ChildItem -Recurse -File | Where-Object Length -GT 100MB
```

Lệnh cuối không được trả về tệp nào. Sau `git add`, luôn kiểm tra cả tệp bị bỏ
qua và xác nhận không có ảnh mặt, database, snapshot, video tập dượt hoặc secret:

```powershell
git add .
git status
git status --ignored --short
git diff --cached --name-only
```

Không commit các nhóm sau:

```text
.env, .env.*
.streamlit/secrets.toml
.venv/
runs/
data/demo.*
data/students/*/*.{jpg,jpeg,png,bmp}
outputs/classroom/
```

## 3. Khởi tạo và push

```powershell
git init
git add .
git status
git commit -m "Complete private classroom helmet demo"
git branch -M main
git remote add origin https://github.com/<USERNAME>/helmet-classroom-demo.git
git push -u origin main
```

Không ghi token hoặc mật khẩu vào source. Dùng đăng nhập trình duyệt/GitHub
Credential Manager.

## 4. Checklist sau khi push

Repository phải có tối thiểu:

```text
streamlit_app.py
src/classroom_demo/
models/best.onnx
models/person/object_detection_nanodet_2022nov.onnx
models/face/face_detection_yunet_2023mar.onnx
models/face/face_recognition_sface_2021dec.onnx
models/MODEL_MANIFEST.json
data/students.csv
scripts/setup_windows.ps1
scripts/preflight.py
RUN_CLASSROOM_DEMO.bat
docs/DEMO_CLASSROOM_VI.md
```

Ảnh đăng ký không xuất hiện trong GitHub là đúng thiết kế. Chuyển chúng riêng
qua kênh nội bộ được phép hoặc dùng ZIP demo riêng tư.

## 5. Điều kiện trước khi chuyển Public

Chỉ chuyển Public sau khi đã đồng thời:

- xác nhận quyền phân phối code và `best.pt`/`best.onnx`;
- thêm software `LICENSE` phù hợp;
- loại toàn bộ dữ liệu khuôn mặt, MSSV và lịch sử nhận diện;
- kiểm tra lại `NOTICE.md`, model manifest và lịch sử Git (không chỉ commit mới
  nhất) để chắc dữ liệu nhạy cảm chưa từng được đẩy lên.
