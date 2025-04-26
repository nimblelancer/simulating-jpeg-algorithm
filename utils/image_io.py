import os
import numpy as np
from PIL import Image
import io

# Đường dẫn thư mục lưu ảnh
BASE_DIR = os.path.join("assets", "images", "processing")

def ensure_dir():
    """Tạo thư mục lưu ảnh nếu chưa tồn tại"""
    os.makedirs(BASE_DIR, exist_ok=True)

def save_image(image: np.ndarray, filename: str):
    """
    Lưu ảnh dưới dạng PNG/JPG. Hỗ trợ cả ảnh xám và RGB.
    """
    ensure_dir()
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(image)
    img_pil.save(os.path.join(BASE_DIR, filename))

def save_npy(data: np.ndarray, filename: str, allow_object=False):
    """Lưu dữ liệu trung gian dưới dạng .npy"""
    ensure_dir()
    path = os.path.join(BASE_DIR, filename)
    if allow_object:
        np.save(path, np.array(data, dtype=object))
    else:
        np.save(path, np.array(data))

def load_npy(filename: str, allow_pickle: bool = False) -> np.ndarray:
    """Đọc dữ liệu .npy, có thể bật allow_pickle nếu cần load object array."""
    path = os.path.join(BASE_DIR, filename)
    return np.load(path, allow_pickle=allow_pickle)

def load_uploaded_image(uploaded_file) -> np.ndarray:
    """
    Đọc ảnh được upload từ Streamlit file_uploader.
    Trả về ảnh dưới dạng numpy array:
    - Nếu ảnh grayscale: trả về 1 channel (numpy array).
    - Nếu ảnh màu: trả về 3 channels (numpy array).
    - Nếu ảnh có 3 channels mà các giá trị trong mỗi channel giống nhau, trả về ảnh grayscale (1 channel).
    """
    if uploaded_file is None:
        return None

    # Mở ảnh từ uploaded file
    image = Image.open(uploaded_file)

    # Kiểm tra ảnh grayscale (chế độ 'L' tương ứng với ảnh grayscale)
    if image.mode == "L":
        # Ảnh là grayscale, trả về numpy array với 1 channel
        return np.array(image)

    # Kiểm tra ảnh RGB
    elif image.mode == "RGB":
        image_array = np.array(image)

        # Kiểm tra nếu tất cả 3 channels giống nhau (có nghĩa là ảnh grayscale)
        if np.all(image_array[:, :, 0] == image_array[:, :, 1]) and np.all(image_array[:, :, 0] == image_array[:, :, 2]):
            # Trả về ảnh grayscale (1 channel)
            return image_array[:, :, 0]  # Chỉ trả về một channel (ví dụ channel đầu tiên)

        # Nếu ảnh có 3 channel khác nhau, trả về numpy array với 3 channel
        return image_array

    # Nếu không phải ảnh grayscale hay ảnh màu RGB thì trả về None
    return None

def load_image(filename: str) -> np.ndarray:
    """Đọc ảnh từ file đã lưu trong thư mục xử lý"""
    path = os.path.join(BASE_DIR, filename)
    img = Image.open(path).convert("RGB")
    return np.array(img)

def list_processing_images():
    """Liệt kê toàn bộ ảnh đã lưu trong thư mục xử lý"""
    ensure_dir()
    return sorted([f for f in os.listdir(BASE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])

def clear_processing_folder():
    """Xoá toàn bộ ảnh và file .npy trong thư mục xử lý"""
    ensure_dir()
    for filename in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith(('.png', '.jpg', '.jpeg', '.npy')):
            os.remove(file_path)

def save_encoded_bytes_to_jpg(encoded_bytes: bytes, filename: str) -> int:
    """
    Lưu dữ liệu mã hóa vào file JPG.
    Trả về số byte đã ghi vào file.
    """
    file_path = os.path.join(BASE_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(encoded_bytes)
    return file_path

def build_huffman_result(dc_codes: dict, ac_codes: dict) -> dict:
    """
    Gộp bảng mã Huffman DC và AC thành một dict duy nhất, 
    định dạng key dễ đọc để hiển thị hoặc lưu trữ.
    
    Parameters:
    -----------
    dc_codes : dict
        Bảng mã Huffman cho DC coefficients (key: size, value: code)
    ac_codes : dict
        Bảng mã Huffman cho AC coefficients (key: (run, size), value: code)
    
    Returns:
    --------
    dict
        Dict chứa tất cả mã Huffman, key dạng 'DC(x)' hoặc 'AC(run,size)'
    """
    huffman_result = {}

    # Gộp DC codes
    for size, code in dc_codes.items():
        huffman_result[f"DC({size})"] = code

    # Gộp AC codes
    for (run, size), code in ac_codes.items():
        huffman_result[f"AC({run},{size})"] = code

    return huffman_result
