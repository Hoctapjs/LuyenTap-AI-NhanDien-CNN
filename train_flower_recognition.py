"""
Tập tin Python này được thiết kế để xây dựng và huấn luyện một mạng nơ-ron tích chập (CNN) dùng để phân loại hoa. 

Các chức năng chính bao gồm:

1. **Tải và xử lý dữ liệu**:
   - Tải ảnh từ thư mục dữ liệu `flowers/`, chứa các thư mục con tương ứng với các loại hoa khác nhau (ví dụ: 'daisy', 'rose').
   - Thay đổi kích thước ảnh về 128x128 pixel và chuẩn hóa giá trị pixel về khoảng [0, 1].
   - Chia dữ liệu thành tập huấn luyện (80%) và tập xác thực (20%).

2. **Xây dựng mô hình CNN**:
   - Tạo kiến trúc mạng CNN theo kiểu Sequential với 3 lớp tích chập (Convolutional), các lớp pooling (MaxPooling), và Dropout để tránh overfitting.
   - Lớp đầu ra là một lớp dày (Dense) với 5 đơn vị (tương ứng với 5 loại hoa) và hàm kích hoạt softmax để phân loại nhiều lớp.

3. **Huấn luyện mô hình**:
   - Biên dịch (compile) mô hình với bộ tối ưu Adam và hàm mất mát categorical crossentropy.
   - Huấn luyện mô hình trong 10 epoch sử dụng dữ liệu huấn luyện và đánh giá trên dữ liệu xác thực.

4. **Đánh giá và lưu mô hình**:
   - Đánh giá mô hình trên tập dữ liệu xác thực để kiểm tra độ chính xác (accuracy) và mất mát (loss).
   - Lưu mô hình đã huấn luyện vào tệp `flower_classifier.h5` để sử dụng sau này.

Tập tin này sử dụng TensorFlow/Keras để xây dựng và huấn luyện mạng nơ-ron, cùng với Matplotlib để hỗ trợ trực quan hóa (mặc dù chưa sử dụng trong tập tin này).
"""



""" tải và xử lý dữ liệu """

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn thư mục chứa dữ liệu
DATASET_PATH = "flowers/"  # Thư mục chứa các thư mục con như 'daisy', 'rose', ...

# Kích thước ảnh đầu vào
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Chuẩn bị dữ liệu
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

""" xây dựng mô hình cnn """
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Tạo mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 loài hoa
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

""" huấn luyện mô hình """
EPOCHS = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

""" đánh giá và lưu mô hình """
# Đánh giá mô hình trên dữ liệu kiểm tra
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Lưu mô hình
model.save("flower_classifier.h5")

