# 🎓 Đồ án: Nhận Diện và Đếm Người Qua Đường, Phân Biệt Giới Tính và Tuổi

## 📌 *Giới thiệu*

*Đây là đồ án của tôi cho môn học **Nhập môn Ảnh số**. Mục tiêu của tôi là xây dựng một hệ thống thị giác máy tính có khả năng:*

- *Nhận diện và đếm số lượng người xuất hiện trong video hoặc qua webcam.*
- *Nhận diện khuôn mặt của từng người.*
- *Phân loại giới tính (Nam/Nữ) và tuổi của người đó.*

---

## 💡 *Các tính năng chính*

- **Phát hiện người:** *Tôi đã sử dụng mô hình **YOLOv8** để nhận diện vị trí của từng người.*
- **Đếm người:** *Chương trình có khả năng đếm tổng số người được nhận diện trong khung hình.*
- **Phát hiện khuôn mặt:** *Tôi dùng **Haar Cascade Classifier** của OpenCV để tìm khuôn mặt trong mỗi vùng ảnh của người.*
- **Phân loại giới tính và tuổi:** *Tôi tích hợp hai mô hình học sâu (**TensorFlow/Keras**) để dự đoán giới tính và tuổi.*
- **Tối ưu tốc độ:** *Tôi đã tối ưu code để chỉ xử lý các mô hình nặng sau mỗi vài khung hình nhằm tăng hiệu suất.*

---

## ⚙️ *Cơ chế hoạt động*

*Quy trình xử lý trong đồ án của tôi gồm các bước sau:*

1. *Nhận diện người:* Sử dụng YOLOv8 để xác định vị trí người trong mỗi khung hình.
2. *Nhận diện khuôn mặt:* Dùng Haar Cascade tìm khuôn mặt trong vùng người đã cắt.
3. *Phân loại giới tính và tuổi:* Đưa khuôn mặt vào mô hình `Gender_model.h5` và `Age_model.h5`.
4. *Hiển thị:* Vẽ khung quanh người và khuôn mặt, đồng thời ghi nhãn giới tính và tuổi lên hình ảnh.

---

## 🧩 *Yêu cầu cài đặt*

*Để chạy dự án này, tôi đã sử dụng các thư viện Python sau (nên dùng Anaconda):*

```bash
# Cài đặt các thư viện cần thiết
pip install ultralytics opencv-python

# Cài đặt TensorFlow và sửa lỗi numpy
conda install tensorflow
pip uninstall numpy
pip install numpy==1.23.5
📁 Các file mô hình tôi đã sử dụng

Gender_model.h5

Age_model.h5
```
▶️ Hướng dẫn sử dụng
Cài đặt thư viện như hướng dẫn ở trên.

Tải các file mô hình và đặt vào thư mục dự án.

Chạy chương trình bằng lệnh:

bash
Sao chép
Chỉnh sửa
python ten_file_cua_ban.py
Dùng webcam: Thay dòng:

python
Sao chép
Chỉnh sửa
cap = cv2.VideoCapture('video.mp4')
bằng:

python
Sao chép
Chỉnh sửa
cap = cv2.VideoCapture(0)
🧰 Xử lý sự cố thường gặp
Lỗi ModuleNotFoundError: Cài lại thư viện bằng pip install hoặc conda install.

Lỗi khi cài TensorFlow: Dùng conda install tensorflow nếu pip không hoạt động.

Lỗi numpy has no attribute 'typeDict': Gỡ cài đặt numpy và cài lại phiên bản 1.23.5.

Tốc độ xử lý chậm: Tôi đã tối ưu để các mô hình nặng chỉ chạy mỗi 5 khung hình.

👨‍💻 Thông tin thêm
YOLOv8 được cung cấp bởi Ultralytics.
Tôi sử dụng mô hình học sâu tự huấn luyện bằng Keras/TensorFlow.
Đồ án này nhằm mục đích học tập và nghiên cứu cá nhân.

