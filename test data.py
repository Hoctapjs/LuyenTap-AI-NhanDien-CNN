"""
Tập tin Python này được thiết kế để kiểm tra tính hợp lệ của các tệp ảnh trong một thư mục và các thư mục con.

Các chức năng chính bao gồm:

1. **Thư mục chứa ảnh**:
   - Thư mục `flowers/` được chỉ định làm nơi chứa các tệp ảnh cần kiểm tra.

2. **Kiểm tra tính hợp lệ của ảnh**:
   - Hàm `validate_images(directory)` thực hiện:
     - Duyệt qua toàn bộ thư mục và các thư mục con để tìm tất cả các tệp.
     - Mở từng tệp bằng thư viện PIL (Pillow) và thực hiện lệnh `img.verify()` để kiểm tra xem tệp có phải là ảnh hợp lệ không.
     - Nếu ảnh không hợp lệ hoặc gặp lỗi khi mở, hiển thị thông báo lỗi cùng đường dẫn tệp không hợp lệ.

3. **Thông báo lỗi**:
   - Nếu phát hiện tệp không hợp lệ, chương trình sẽ in ra thông tin chi tiết về lỗi, bao gồm đường dẫn tệp và loại lỗi (IOError, SyntaxError).

4. **Ứng dụng thực tiễn**:
   - Dùng để xác minh tính chính xác và loại bỏ các tệp ảnh lỗi trước khi xử lý hoặc huấn luyện mô hình.

Lưu ý: Chỉ xác minh ảnh có thể được mở thành công; không thay đổi hoặc sửa đổi nội dung tệp.
"""


import os
from PIL import Image

image_dir = "flowers/"

def validate_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Xác minh tính hợp lệ của ảnh
            except (IOError, SyntaxError) as e:
                print(f"Invalid image file: {file_path} - {e}")

validate_images(image_dir)
