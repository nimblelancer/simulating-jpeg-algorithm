import numpy as np

# region DCT

# def dct_2d(block):
#     """
#     Thực hiện DCT 2D trên một khối 8x8.
    
#     Input:
#         block: Ma trận 8x8 đầu vào
    
#     Output:
#         Ma trận 8x8 sau khi áp dụng DCT
#     """
#     # Kiểm tra kích thước đầu vào
#     if block.shape != (8, 8):
#         raise ValueError("Block phải có kích thước 8x8")
    
#     # Trừ 128 để dịch giá trị về khoảng [-128, 127]
#     block = block - 128
    
#     # Khởi tạo ma trận kết quả
#     dct_result = np.zeros((8, 8))
    
#     # Triển khai công thức DCT
#     for u in range(8):
#         for v in range(8):
#             # Hệ số alpha cho u
#             cu = 1 / np.sqrt(2) if u == 0 else 1
            
#             # Hệ số alpha cho v
#             cv = 1 / np.sqrt(2) if v == 0 else 1
            
#             # Tính tổng theo công thức DCT
#             sum_val = 0
#             for x in range(8):
#                 for y in range(8):
#                     cos_x = np.cos((2 * x + 1) * u * np.pi / 16)
#                     cos_y = np.cos((2 * y + 1) * v * np.pi / 16)
#                     sum_val += block[x, y] * cos_x * cos_y
            
#             # Nhân với hệ số và lưu kết quả
#             dct_result[u, v] = 0.25 * cu * cv * sum_val
    
#     return dct_result

def dct_1d(vector):
    """
    Thực hiện DCT 1D trên một vector độ dài 8.
    
    Parameters:
    -----------
    vector : ndarray
        Vector đầu vào, shape (8,), dtype=float32
    
    Returns:
    --------
    ndarray
        Vector hệ số DCT, shape (8,), dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu vector không có độ dài 8
    """
    if vector.shape != (8,):
        raise ValueError("Vector phải có độ dài 8")
    
    vector = vector.astype(np.float32)
    n = 8
    # Ma trận DCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    dct_matrix = np.sqrt(2 / n) * ck * cos_term
    
    return np.dot(dct_matrix, vector)

def dct_2d_separable(block):
    """
    Thực hiện DCT 2D trên một khối 8x8.
    
    Parameters:
    -----------
    block : ndarray
        Khối đầu vào, shape (8, 8), dtype=float32, giá trị trong [0, 255]
    
    Returns:
    --------
    ndarray
        Khối hệ số DCT, shape (8, 8), dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu block không có shape (8, 8) hoặc giá trị ngoài [0, 255]
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    if block.max() > 255 or block.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    block = block.astype(np.float32) - 128
    n = 8
    # Ma trận DCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    dct_matrix = np.sqrt(2 / n) * ck * cos_term
    
    # DCT 2D = dct_matrix * block * dct_matrix.T
    return np.dot(np.dot(dct_matrix, block), dct_matrix.T)

def apply_dct_to_image(image_blocks):
    """
    Áp dụng DCT cho tất cả các khối 8x8 của ảnh.
    
    Parameters:
    -----------
    image_blocks : ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh đa kênh
        dtype=float32, giá trị trong [0, 255]
    
    Returns:
    --------
    ndarray
        Các khối hệ số DCT, shape giống đầu vào, dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc giá trị ngoài [0, 255]
    """
    if image_blocks.ndim not in (4, 5):
        raise ValueError("image_blocks phải là mảng 4D hoặc 5D")
    if image_blocks.max() > 255 or image_blocks.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    if image_blocks.shape[-2:] != (8, 8):
        raise ValueError("Kích thước khối phải là 8x8")
    
    image_blocks = image_blocks.astype(np.float32)
    n = 8
    # Ma trận DCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    dct_matrix = np.sqrt(2 / n) * ck * cos_term
    
    if image_blocks.ndim == 4:
        h, w = image_blocks.shape[:2]
        blocks = image_blocks.reshape(h * w, 8, 8) - 128
        dct_blocks = np.einsum('ij,kjl,lm->kim', dct_matrix, blocks, dct_matrix.T)
        return dct_blocks.reshape(h, w, 8, 8)
    
    c, h, w = image_blocks.shape[:3]
    blocks = image_blocks.reshape(c, h * w, 8, 8) - 128
    dct_blocks = np.einsum('ij,ckjl,lm->ckim', dct_matrix, blocks, dct_matrix.T)
    return dct_blocks.reshape(c, h, w, 8, 8)

def idct_1d(vector):
    """
    Thực hiện IDCT 1D trên một vector độ dài 8.
    
    Parameters:
    -----------
    vector : ndarray
        Vector hệ số DCT, shape (8,), dtype=float32
    
    Returns:
    --------
    ndarray
        Vector giá trị pixel, shape (8,), dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu vector không có độ dài 8
    """
    if vector.shape != (8,):
        raise ValueError("Vector phải có độ dài 8")
    
    vector = vector.astype(np.float32)
    n = 8
    # Ma trận IDCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    idct_matrix = np.sqrt(2 / n) * ck * cos_term.T
    
    return np.dot(idct_matrix, vector)

def idct_2d_separable(block):
    """
    Thực hiện IDCT 2D trên một khối 8x8.
    
    Parameters:
    -----------
    block : ndarray
        Khối hệ số DCT, shape (8, 8), dtype=float32
    
    Returns:
    --------
    ndarray
        Khối pixel, shape (8, 8), dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu block không có shape (8, 8)
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    
    block = block.astype(np.float32)
    n = 8
    # Ma trận IDCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    idct_matrix = np.sqrt(2 / n) * ck * cos_term.T
    
    # IDCT 2D = idct_matrix * block * idct_matrix.T
    result = np.dot(np.dot(idct_matrix, block), idct_matrix.T)
    result = np.clip(result + 128, 0, 255)
    return result.astype(np.float32)

def apply_idct_to_image(dct_blocks):
    """
    Áp dụng IDCT cho tất cả các khối 8x8 của ảnh.
    
    Parameters:
    -----------
    dct_blocks : ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh đa kênh
        dtype=float32
    
    Returns:
    --------
    ndarray
        Các khối pixel, shape giống đầu vào, dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng
    """
    if dct_blocks.ndim not in (4, 5):
        raise ValueError("dct_blocks phải là mảng 4D hoặc 5D")
    if dct_blocks.shape[-2:] != (8, 8):
        raise ValueError("Kích thước khối phải là 8x8")
    
    dct_blocks = dct_blocks.astype(np.float32)
    n = 8
    # Ma trận IDCT
    k = np.arange(n)
    i = np.arange(n).reshape(-1, 1)
    cos_term = np.cos((2 * i + 1) * k * np.pi / (2 * n))
    ck = np.ones(n)
    ck[0] = 1 / np.sqrt(2)
    idct_matrix = np.sqrt(2 / n) * ck * cos_term.T
    
    if dct_blocks.ndim == 4:
        h, w = dct_blocks.shape[:2]
        blocks = dct_blocks.reshape(h * w, 8, 8)
        pixel_blocks = np.einsum('ij,kjl,lm->kim', idct_matrix, blocks, idct_matrix.T)
        pixel_blocks = np.clip(pixel_blocks + 128, 0, 255)
        return pixel_blocks.reshape(h, w, 8, 8).astype(np.float32)
    
    c, h, w = dct_blocks.shape[:3]
    blocks = dct_blocks.reshape(c, h * w, 8, 8)
    pixel_blocks = np.einsum('ij,ckjl,lm->ckim', idct_matrix, blocks, idct_matrix.T)
    pixel_blocks = np.clip(pixel_blocks + 128, 0, 255)
    return pixel_blocks.reshape(c, h, w, 8, 8).astype(np.float32)
