import numpy as np
from PIL import Image
import pickle
import json
import os
import struct
from pathlib import Path

def is_grayscale_image(img):
    """
    Kiểm tra xem ảnh có phải là ảnh xám (2D) hoặc ảnh xám giả (RGB/BGR với các kênh giống nhau).
    
    Parameters:
    -----------
    img : ndarray
        Mảng NumPy biểu diễn ảnh (2D hoặc 3D)
    
    Returns:
    --------
    bool
        True nếu ảnh là ảnh xám, False nếu là ảnh màu
    """
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return True
    if img.ndim == 3 and img.shape[2] == 3:
        # Kiểm tra mẫu nhỏ để tối ưu tốc độ
        sample_size = min(32, img.shape[0], img.shape[1])
        sample = img[:sample_size, :sample_size, :]
        return np.array_equal(sample[:, :, 0], sample[:, :, 1]) and np.array_equal(sample[:, :, 0], sample[:, :, 2])
    return False

def read_image(image_path):
    """
    Đọc ảnh từ đường dẫn, tự động nhận diện và xử lý phù hợp với ảnh màu hoặc ảnh xám.
    Nếu ảnh có 3 kênh màu nhưng các kênh giống hệt nhau, chuyển đổi thành ảnh xám (2D).
    
    Parameters:
    -----------
    image_path : str
        Đường dẫn đến file ảnh
    
    Returns:
    --------
    ndarray
        Mảng NumPy biểu diễn ảnh với shape (height, width) cho ảnh xám
        hoặc (height, width, 3) cho ảnh màu RGB, dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    IOError
        Nếu không thể đọc được ảnh
    ValueError
        Nếu giá trị pixel ngoài khoảng [0, 255]
    """
    try:
        # Đọc ảnh bằng PIL
        img_pil = Image.open(image_path)
        img = np.array(img_pil)
        
        # Loại bỏ kênh alpha nếu có
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Kiểm tra giá trị pixel
        if img.max() > 255 or img.min() < 0:
            raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
        
        # Chuyển đổi sang ảnh xám nếu cần
        if is_grayscale_image(img):
            if img.ndim == 3:
                img = img[:, :, 0]  # Lấy một kênh
        
        return np.clip(img, 0, 255).astype(np.float32)
    
    except Exception as e:
        raise IOError(f"Không thể đọc ảnh từ {image_path}: {str(e)}")

def save_image(image_path, image):
    """
    Lưu ảnh ra file.
    
    Parameters:
    -----------
    image_path : str
        Đường dẫn để lưu ảnh
    image : ndarray
        Mảng numpy biểu diễn ảnh cần lưu
    
    Returns:
    --------
    None
    """
    try:
        # Đảm bảo giá trị pixel nằm trong khoảng [0, 255] và kiểu dữ liệu là uint8
        img = np.clip(image, 0, 255).astype(np.uint8)
        
        # Kiểm tra phần mở rộng của file để xác định định dạng lưu
        ext = image_path.lower().split('.')[-1]
        
        if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            # Nếu ảnh là RGB, OpenCV cần BGR
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            # Lưu ảnh bằng OpenCV
            cv2.imwrite(image_path, img)
        else:
            # Nếu định dạng không được hỗ trợ bởi OpenCV, dùng PIL
            img_pil = Image.fromarray(img)
            img_pil.save(image_path)
            
        print(f"Đã lưu ảnh thành công tại: {image_path}")
    
    except Exception as e:
        raise IOError(f"Không thể lưu ảnh vào {image_path}: {str(e)}")

def create_jpeg_header(image_shape, quality=50, subsampling='4:2:0', is_color=True):
    """
    Tạo header cho file JPEG nén.
    
    Parameters:
    -----------
    image_shape : tuple
        Kích thước của ảnh gốc
    quality : int
        Hệ số chất lượng từ 1-100
    subsampling : str
        Kiểu lấy mẫu phụ ('4:4:4', '4:2:2', '4:2:0')
    is_color : bool
        Cờ đánh dấu ảnh màu hay ảnh xám
    
    Returns:
    --------
    dict
        Dictionary chứa các thông tin header
    """
    header = {
        'image_shape': image_shape,
        'quality': quality,
        'subsampling': subsampling,
        'is_color': is_color,
        'version': '1.0',  # Phiên bản của định dạng file
        'creation_date': None  # Sẽ được tự động thêm khi lưu
    }
    
    return header

def save_compressed_file(output_path, header, compressed_data):
    """
    Lưu dữ liệu đã nén và header vào file.
    
    Parameters:
    -----------
    output_path : str
        Đường dẫn để lưu file nén
    header : dict
        Thông tin header
    compressed_data : object
        Dữ liệu đã nén (bảng Huffman, dữ liệu đã mã hóa)
    
    Returns:
    --------
    None
    """
    try:
        # Thêm thời gian tạo file
        from datetime import datetime
        header['creation_date'] = datetime.now().isoformat()
        
        # Tạo thư mục cha nếu chưa tồn tại
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'wb') as f:
            # Ghi magic number để xác định đây là file nén của chúng ta
            f.write(b'MYJPG')
            
            # Chuyển đổi header thành JSON và lưu độ dài
            header_json = json.dumps(header).encode('utf-8')
            header_length = len(header_json)
            
            # Ghi độ dài header (4 bytes)
            f.write(struct.pack('!I', header_length))
            
            # Ghi header
            f.write(header_json)
            
            # Ghi dữ liệu nén
            pickle.dump(compressed_data, f)
            
        print(f"Đã lưu file nén thành công tại: {output_path}")
        print(f"Kích thước file: {Path(output_path).stat().st_size:,} bytes")
        
    except Exception as e:
        raise IOError(f"Không thể lưu file nén tại {output_path}: {str(e)}")

def read_compressed_file(input_path):
    """
    Đọc file nén và trả về header và dữ liệu nén.
    
    Parameters:
    -----------
    input_path : str
        Đường dẫn đến file nén
    
    Returns:
    --------
    tuple
        (header, compressed_data)
    """
    try:
        with open(input_path, 'rb') as f:
            # Đọc và kiểm tra magic number
            magic = f.read(5)
            if magic != b'MYJPG':
                raise ValueError("File không phải là định dạng JPEG tự triển khai")
            
            # Đọc độ dài header
            header_length = struct.unpack('!I', f.read(4))[0]
            
            # Đọc header
            header_json = f.read(header_length)
            header = json.loads(header_json.decode('utf-8'))
            
            # Đọc dữ liệu nén
            compressed_data = pickle.load(f)
        
        return header, compressed_data
    
    except Exception as e:
        raise IOError(f"Không thể đọc file nén từ {input_path}: {str(e)}")

def adjust_quant_tables(quality):
    """
    Điều chỉnh các bảng lượng tử hóa dựa trên hệ số chất lượng.
    
    Parameters:
    -----------
    quality : int
        Hệ số chất lượng từ 1 đến 100
    
    Returns:
    --------
    tuple
        (y_quant_table, c_quant_table) - Bảng lượng tử cho kênh sáng Y và kênh màu C
    """
    # Đảm bảo quality nằm trong khoảng [1, 100]
    quality = max(1, min(100, quality))
    
    # Bảng lượng tử hóa chuẩn cho kênh sáng (luminance - Y) theo JPEG
    y_quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Bảng lượng tử hóa chuẩn cho kênh màu (chrominance - Cb, Cr) theo JPEG
    c_quant_table = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)
    
    # Tính hệ số điều chỉnh dựa trên quality
    if quality < 50:
        scale_factor = 5000 / quality
    else:
        scale_factor = 200 - 2 * quality
    
    # Điều chỉnh bảng lượng tử hóa theo chất lượng
    if scale_factor != 100:
        y_quant_table = np.floor((y_quant_table * scale_factor + 50) / 100)
        c_quant_table = np.floor((c_quant_table * scale_factor + 50) / 100)
        
        # Giới hạn giá trị trong khoảng [1, 255]
        y_quant_table = np.clip(y_quant_table, 1, 255)
        c_quant_table = np.clip(c_quant_table, 1, 255)
    
    return y_quant_table, c_quant_table
