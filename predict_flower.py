"""
Tập tin Python này được thiết kế để thực hiện dự đoán loại hoa từ một ảnh mới sử dụng mô hình đã được huấn luyện.

Các chức năng chính bao gồm:

1. **Tải mô hình**:
   - Tải mô hình phân loại hoa đã được huấn luyện trước đó từ tệp `flower_classifier.h5`.

2. **Dự đoán loại hoa từ một ảnh**:
   - Hàm `predict_flower(image_path)` thực hiện các bước sau:
     - Tải ảnh từ đường dẫn được cung cấp và thay đổi kích thước ảnh về kích thước chuẩn 128x128 pixel.
     - Chuyển đổi ảnh sang mảng numpy và chuẩn hóa giá trị pixel về khoảng [0, 1].
     - Dự đoán xác suất của từng loại hoa dựa trên mô hình đã huấn luyện.
     - Xác định lớp hoa được dự đoán dựa trên xác suất cao nhất và ánh xạ nó với tên lớp tương ứng.

3. **Hiển thị ảnh và kết quả dự đoán**:
   - Ảnh đầu vào sẽ được hiển thị cùng với tên loại hoa được mô hình dự đoán.

4. **Ví dụ sử dụng**:
   - Gọi hàm `predict_flower("path_to_your_image.jpg")` để dự đoán loại hoa của một ảnh cụ thể, thay thế `path_to_your_image.jpg` bằng đường dẫn ảnh bạn muốn dự đoán.

Lưu ý: Kích thước ảnh và danh sách các lớp được sử dụng dựa trên dữ liệu huấn luyện ban đầu.
"""



""" dự đoán với ảnh mới """
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
""" import train_flower_recognition """

IMG_SIZE = (128, 128)

# Tải mô hình
model = load_model("flower_classifier.h5")

# Dự đoán một ảnh mới
def predict_flower(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)

    # Giả sử mô hình lưu class_names trong thuộc tính
    try:
        class_names = model.class_names
    except AttributeError:
        class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']  # Dự phòng

    """ class_names = list(train_data.class_indices.keys()) """

    predicted_class = class_names[np.argmax(predictions)]
    
    plt.imshow(img)
    plt.title(f"Dự đoán hoa là: {predicted_class}")
    plt.axis("off")
    plt.show()

# Ví dụ
predict_flower("data for test/10.jpg")

