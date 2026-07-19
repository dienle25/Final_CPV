# Ảnh đăng ký khuôn mặt

Mỗi sinh viên có một thư mục mang tên MSSV. Ứng dụng đọc các tệp
`.jpg`, `.jpeg`, `.png` và `.bmp` trong thư mục đó để tạo mẫu nhận diện.

CE190579 hiện có 21 ảnh (1 ảnh thẻ và 20 khung đa góc từ video đăng ký). Bốn
sinh viên CE182206, CE190625, CE191641 và CE190256 vẫn chỉ có một ảnh khởi động,
nên chưa sẵn sàng cho demo nhận diện nhiều góc. Trước buổi demo, vào trang
**Đăng ký sinh viên** và bổ sung khoảng 6-15 ảnh rõ cho mỗi người đó:

- chính diện, xoay nhẹ trái/phải và hơi ngẩng/cúi;
- ánh sáng giống phòng demo, không ngược sáng;
- một vài ảnh có đội đúng loại mũ sẽ dùng trong demo;
- mỗi ảnh chỉ có một khuôn mặt, khuôn mặt rộng tối thiểu khoảng 100 px;
- không dùng ảnh qua bộ lọc làm đẹp mạnh.

Ứng dụng cảnh báo danh tính có ít hơn 5 ảnh và dùng ngưỡng nhận diện chặt hơn
cho trường hợp này. Khi đăng ký, ảnh quá nhỏ, không nhất quán với MSSV đang bổ
sung hoặc quá giống một MSSV khác có thể bị từ chối để tránh làm bẩn gallery.

Đây là dữ liệu sinh trắc học. Chỉ thu thập khi đã được đồng ý, lưu cục bộ và
không đưa các thư mục ảnh lên kho Git công khai.
