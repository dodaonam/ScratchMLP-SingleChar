# ScratchMLP-SingleChar

## Mục tiêu dự án

- Dự án được thực hiện với mục đích học tập và nắm vững kiến thức về Mạng neuron nhân tạo.
- Tự **xây dựng lại một mạng MLP (Multi-Layer Perceptron) từ đầu bằng NumPy dùng để dự đoán chữ cái viết tay** trên tập dữ liêu EMNIST, để hiểu rõ cách hoạt động của mô hình học sâu, thay vì dựa vào các framework có sẵn.

## Quy trình

### 1. Xây dựng nền tảng MLP

Xây dựng một mạng MLP cơ bản, hoàn toàn bằng NumPy.

- Lan truyền xuôi (forward propagation)
- Hàm mất mát (cross-entropy loss)
- Lan truyền ngược (backpropagation)
- Cập nhật trọng số bằng Gradient Descent hoặc Adam

### 2. Áp dụng chính quy hoá

- **L2 Regularization**: Thêm một thành phần phạt tỉ lệ với bình phương độ lớn của trọng số vào hàm mất mát, sau đó được tính đạo hàm trong quá trình lan truyền ngược. Mục đích giới hạn độ lớn trọng số và giảm overfitting.
- **Dropout**: Tắt ngẫu nhiên một tỉ lệ neuron trong quá trình huấn luyện. Buộc mạng học các đặc trưng tổng quát hơn và tránh phụ thuộc vào một nhóm neuron cụ thể.
- **Batch Normalization**: Chuẩn hóa đầu ra của mỗi lớp theo mini-batch, ổn định phân phối dữ liệu, giảm hiện tượng Internal Covariate Shift. Giúp gradient trở nên ổn định hơn, tốc độ hội tụ nhanh hơn, mô hình ít phụ thuộc vào khởi tạo trọng số và có khả năng tổng quát hóa tốt hơn.
- **Early Stopping**: Ngăn overfitting bằng cách dừng huấn luyện sớm khi hiệu suất trên tập validation không còn cải thiện sau một số epoch.

### 3. Tối ưu siêu tham số

Tổ hợp tối ưu của các tham số:

- Learning rate
- Lambda (L2 coefficient)
- Keep probability (Dropout)
- Số lớp ẩn
- Số node mỗi lớp

## Cấu trúc thư mục

```
ScratchMLP-SingleChar/
│
├── data/
│ ├── EMNIST/                                   # Dữ liệu gốc EMNIST
│ └── processed/                                # Dữ liệu sau khi xử lý và chia tập
│
├── models/
│ ├── mlp_weights.npz                           # Trọng số của mô hình cơ bản
│ └── mlp_weights_tuned.npz                     # Trọng số mô hình sau khi tuning
│
├── notebooks/
│ ├── eda.ipynb                                 # Phân tích và trực quan hóa dữ liệu
│ ├── evaluate_base_model.ipynb                 # Đánh giá mô hình gốc
│ └── evaluate_regularized_model.ipynb          # Đánh giá mô hình có regularization
│
├── output/
│ ├── training_history.png                      # Lịch sử huấn luyện mô hình gốc
│ ├── training_history_regularization.png       # Lịch sử mô hình có regularization
│ └── training_history_tuned.png                # Lịch sử mô hình sau tuning
│
├── src/
│ ├── init.py
│ ├── load_data.py                              # Hàm tải và xử lý dữ liệu
│ ├── model.py                                  # Định nghĩa lớp MLP
│ └── train.py                                  # Pipeline huấn luyện cơ bản
│
├── tuning/
│ ├── tuning.py                                 # Tìm kiếm siêu tham số tối ưu
│ └── train_best_model.py                       # Huấn luyện mô hình tối ưu
│
├── venv/                                       # Môi trường ảo
│
├── demo.py                                     # Chạy thử mô hình đã huấn luyện
├── requirements.txt                            # Thư viện cần thiết
└── README.md                                   # Mô tả dự án
```

## Kết quả

### 1. Mô hình cơ bản (Base Model)

- Phiên bản đầu tiên của mạng MLP được huấn luyện **chưa áp dụng bất kỳ kỹ thuật chính quy hoá (regularization)** nào.
- Kết quả trên tập kiểm thử (test set): **Accuracy: 0.8253 (82.53%)**

### 2. Mô hình có chính quy hoá (Regularized Model)

- Sau khi thêm các kỹ thuật **regularization**, mô hình được cải thiện rõ rệt.
- Kết quả trên tập kiểm thử (test set): **Accuracy: 0.8586 (85.86%)**

### 3. Mô hình sau khi tinh chỉnh siêu tham số (Tuned Model)

- Sau khi thực hiện **hyperparameter tuning** (tối ưu learning rate, lambda, keep_prob, số lớp và node), mô hình đạt hiệu suất tốt nhất.
- Kết quả trên tập kiểm thử (test set): **Accuracy: 0.8622 (86.22%)**

## Lời kết

- Dự án giúp bản thân hiểu rõ cách một mô hình AI "học" như thế nào. Cách các kỹ thuật chính quy hoá tác động đến mô hình.
- Việc sử dụng NumPy khiến mô hình chỉ chạy trên CPU. Nếu cần GPU, nên sử dụng các framework như PyTorch/TensorFlow.
- Làm phẳng ảnh sẽ phá vỡ cấu trúc không gian, khiến mô hình tuyến tính không nhìn được đặc trưng về hình dạng hoặc cạnh.
- Nếu muốn đạt độ chính xác cao hơn, nên sử dụng các mô hình CNN đã chứng minh hiệu quả vượt trội trong các tác vụ xử lý ảnh.