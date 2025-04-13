import numpy as np

def zigzag_scan(block):
    """
    Chuyển khối 8x8 thành mảng 1D theo thứ tự zigzag.
    
    Parameters:
    -----------
    block : ndarray
        Khối 8x8, shape (8, 8), dtype=int32, hệ số lượng tử hóa
    
    Returns:
    --------
    ndarray
        Mảng 1D, shape (64,), dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc dtype không phải int32
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    if block.dtype != np.int32:
        raise ValueError("Khối phải có dtype int32")
    
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    return block.flatten()[zigzag_indices].astype(np.int32)

def run_length_encode(array):
    """
    Mã hóa RLE cho mảng 1D sau zigzag, theo chuẩn JPEG.
    
    Parameters:
    -----------
    array : ndarray
        Mảng 1D, shape (64,), dtype=int32
    
    Returns:
    --------
    tuple
        (dc, ac): dc là số nguyên, ac là list các (run, value) cho AC coefficients
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc dtype không phải int32
    """
    if array.shape != (64,) or array.dtype != np.int32:
        raise ValueError("Mảng phải có shape (64,) và dtype int32")
    
    dc = array[0]
    ac = []
    run = 0
    
    for value in array[1:]:
        if value == 0:
            run += 1
        else:
            ac.append((run, value))
            run = 0
    if run > 0:
        ac.append((run, 0))  # Kết thúc bằng chuỗi 0 (EOB)
    
    return dc, ac

def inverse_rle(rle_data):
    """
    Giải mã RLE để khôi phục mảng 1D.
    
    Parameters:
    -----------
    rle_data : tuple
        (dc, ac): dc là số nguyên, ac là list các (run, value)
    
    Returns:
    --------
    ndarray
        Mảng 1D, shape (64,), dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu dữ liệu RLE không hợp lệ
    """
    dc, ac = rle_data
    array = [dc]
    
    for run, value in ac:
        if run > 0:
            array.extend([0] * run)
        if value != 0 or run < 16:  # EOB được biểu thị bằng (run, 0)
            array.append(value)
    
    array = array[:64]  # Cắt bớt nếu dài hơn
    if len(array) < 64:
        array.extend([0] * (64 - len(array)))  # Đệm 0 nếu ngắn hơn
    
    return np.array(array, dtype=np.int32)

def inverse_zigzag(array):
    """
    Chuyển mảng 1D về khối 8x8 theo thứ tự zigzag.
    
    Parameters:
    -----------
    array : ndarray
        Mảng 1D, shape (64,), dtype=int32
    
    Returns:
    --------
    ndarray
        Khối 8x8, shape (8, 8), dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc dtype không phải int32
    """
    if array.shape != (64,) or array.dtype != np.int32:
        raise ValueError("Mảng phải có shape (64,) và dtype int32")
    
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    block = np.zeros(64, dtype=np.int32)
    block[zigzag_indices] = array
    return block.reshape(8, 8)

def apply_zigzag_and_rle(blocks):
    """
    Áp dụng zigzag scan và RLE cho tất cả các khối 8x8.
    
    Parameters:
    -----------
    blocks : ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh màu
        dtype=int32
    
    Returns:
    --------
    list
        - Ảnh xám: List các (dc, ac)
        - Ảnh màu: List các [channel][block] = (dc, ac)
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc dtype không phải int32
    """
    if blocks.ndim not in (4, 5) or blocks.shape[-2:] != (8, 8):
        raise ValueError("blocks phải là mảng 4D (h, w, 8, 8) hoặc 5D (c, h, w, 8, 8)")
    if blocks.dtype != np.int32:
        raise ValueError("blocks phải có dtype int32")
    
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    result = []
    if blocks.ndim == 4:
        h, w = blocks.shape[:2]
        flat_blocks = blocks.reshape(h * w, 64)
        zigzagged = flat_blocks[:, zigzag_indices]
        for i in range(h * w):
            dc, ac = run_length_encode(zigzagged[i])
            result.append((dc, ac))
    else:
        c, h, w = blocks.shape[:3]
        flat_blocks = blocks.reshape(c, h * w, 64)
        zigzagged = flat_blocks[:, :, zigzag_indices]
        for ch in range(c):
            channel_result = []
            for i in range(h * w):
                dc, ac = run_length_encode(zigzagged[ch, i])
                channel_result.append((dc, ac))
            result.append(channel_result)
    
    return result

def apply_inverse_zigzag_and_rle(rle_blocks, image_shape):
    """
    Giải mã RLE và zigzag để khôi phục các khối 8x8.
    
    Parameters:
    -----------
    rle_blocks : list
        - Ảnh xám: List các (dc, ac)
        - Ảnh màu: List các [channel][block] = (dc, ac)
    image_shape : tuple
        - (h, w) cho ảnh xám
        - (c, h, w) cho ảnh màu
    
    Returns:
    --------
    ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh màu
        dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu rle_blocks hoặc image_shape không hợp lệ
    """
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    if len(image_shape) == 2:
        h, w = image_shape
        if len(rle_blocks) != h * w:
            raise ValueError("Số lượng rle_blocks không khớp với image_shape")
        
        blocks = np.zeros((h * w, 64), dtype=np.int32)
        for i, rle in enumerate(rle_blocks):
            blocks[i] = inverse_rle(rle)
        
        # Inverse zigzag
        blocks[:, zigzag_indices] = blocks
        return blocks.reshape(h, w, 8, 8).astype(np.int32)
    
    elif len(image_shape) == 3:
        c, h, w = image_shape
        if len(rle_blocks) != c or any(len(channel) != h * w for channel in rle_blocks):
            raise ValueError("Số lượng rle_blocks không khớp với image_shape")
        
        blocks = np.zeros((c, h * w, 64), dtype=np.int32)
        for ch in range(c):
            for i, rle in enumerate(rle_blocks[ch]):
                blocks[ch, i] = inverse_rle(rle)
        
        # Inverse zigzag
        blocks[:, :, zigzag_indices] = blocks
        return blocks.reshape(c, h, w, 8, 8).astype(np.int32)
    
    else:
        raise ValueError("image_shape phải có dạng (h, w) hoặc (c, h, w)")