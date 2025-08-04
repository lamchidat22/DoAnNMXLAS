Đồ án: Nhận Diện và Đếm Người Qua Đường, Phân Biệt Giới Tính và Tuổi
Giới thiệu
Đây là đồ án của tôi cho môn học Nhập môn Ảnh số. Mục tiêu của tôi là xây dựng một hệ thống thị giác máy tính có khả năng nhận diện và đếm người, đồng thời phân biệt giới tính và tuổi của họ trong các video hoặc qua webcam.

Các tính năng chính
Phát hiện người: Tôi đã sử dụng mô hình YOLOv8 để nhận diện vị trí của từng người.

Đếm người: Chương trình có khả năng đếm tổng số người được nhận diện trong khung hình.

Phát hiện khuôn mặt: Tôi đã dùng thuật toán Haar Cascade Classifier của OpenCV để tìm khuôn mặt trong mỗi vùng ảnh của người.

Phân loại giới tính và tuổi: Tôi đã tích hợp hai mô hình học sâu (TensorFlow/Keras) để dự đoán giới tính (Nam/Nữ) và tuổi của từng người.

Tối ưu tốc độ: Để cải thiện hiệu suất, tôi đã tối ưu code để chỉ xử lý các mô hình nặng sau mỗi vài khung hình.

Cơ chế hoạt động
Đồ án của tôi hoạt động theo một quy trình như sau:

Nhận diện người: Dùng mô hình YOLOv8 để xác định vị trí của tất cả mọi người trong khung hình.

Nhận diện khuôn mặt: Từ mỗi vùng ảnh của người, tôi dùng Haar Cascade Classifier để tìm vị trí khuôn mặt.

Phân loại giới tính và tuổi: Cắt vùng khuôn mặt và đưa vào các mô hình Keras (Gender_model.h5 và Age_model.h5) để phân loại.

Hiển thị: Vẽ các khung bao quanh người, khuôn mặt, và ghi nhãn giới tính, tuổi lên màn hình.

Yêu cầu cài đặt
Để chạy dự án này, tôi đã cài đặt các thư viện Python sau (trong môi trường Anaconda):

Bash

# Cài đặt các thư viện cần thiết
pip install ultralytics opencv-python

# Cài đặt TensorFlow và một phiên bản numpy cũ hơn để tránh lỗi
conda install tensorflow
pip uninstall numpy
pip install numpy==1.23.5
Các file mô hình tôi đã sử dụng
Tôi đã tải và sử dụng các file mô hình sau, bạn cần tải về và đặt chúng cùng thư mục với file Python để chạy:

Gender_model.h5

Age_model.h5

Hướng dẫn sử dụng
Cài đặt các thư viện theo hướng dẫn ở trên.

Tải các file mô hình và đặt chúng vào thư mục dự án.

Chạy chương trình bằng lệnh sau trong Terminal hoặc Anaconda Prompt:

Bash

python ten_file_cua_ban.py
Sử dụng webcam: Để dùng webcam, tôi thay đổi dòng cap = cv2.VideoCapture('video.mp4') thành cap = cv2.VideoCapture(0).

Xử lý sự cố thường gặp
Lỗi ModuleNotFoundError: Tôi sẽ dùng pip install hoặc conda install để cài đặt lại thư viện.

Lỗi khi cài TensorFlow: Nếu pip install tensorflow không được, tôi dùng conda install tensorflow.

Lỗi numpy has no attribute 'typeDict': Tôi gỡ numpy và cài đặt lại phiên bản 1.23.5.

Tốc độ xử lý chậm: Tôi đã dùng phương pháp tối ưu hóa để chỉ xử lý các mô hình nặng sau mỗi 5 khung hình.
