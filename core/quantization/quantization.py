import numpy as np

_quant_table_cache = {}
def adjust_quant_tables(quality):
    """
    Trả về bảng lượng tử hóa cho Y và Cb/Cr theo quality, dùng cache để tăng tốc.
    
    Returns:
        (y_quant, c_quant): Tuple gồm 2 bảng lượng tử hóa 8x8, dtype=np.uint8
    """
    if quality in _quant_table_cache:
        return _quant_table_cache[quality]

    if not 1 <= quality <= 100:
        raise ValueError("Hệ số chất lượng phải từ 1 đến 100")
    
    y_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    c_table = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # Scale theo chuẩn JPEG
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    y_quant = np.clip(np.floor((y_table * scale + 50) / 100), 1, 255).astype(np.uint8)
    c_quant = np.clip(np.floor((c_table * scale + 50) / 100), 1, 255).astype(np.uint8)

    _quant_table_cache[quality] = (y_quant, c_quant)
    return y_quant, c_quant


def optimize_quantization_for_speed(dct_blocks, quality=50):
    """
    Lượng tử hóa vector hóa cho tất cả các khối DCT.
    
    Parameters:
    -----------
    dct_blocks : ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh màu
        dtype=float32
    quality : int
        Hệ số chất lượng từ 1 đến 100, default=50
    
    Returns:
    --------
    ndarray
        Các khối lượng tử hóa, shape giống đầu vào, dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc quality ngoài [1, 100]
    """
    if dct_blocks.ndim not in (4, 5) or dct_blocks.shape[-2:] != (8, 8):
        raise ValueError("dct_blocks phải là mảng 4D (h, w, 8, 8) hoặc 5D (c, h, w, 8, 8)")
    if not 1 <= quality <= 100:
        raise ValueError("Hệ số chất lượng phải từ 1 đến 100")
    
    y_quant, c_quant = adjust_quant_tables(quality)
    quant_blocks = np.zeros_like(dct_blocks, dtype=np.int32)
    
    if dct_blocks.ndim == 4:
        quant_blocks = np.round(dct_blocks / y_quant).astype(np.int32)
    else:
        quant_blocks[0] = np.round(dct_blocks[0] / y_quant).astype(np.int32)
        quant_blocks[1:] = np.round(dct_blocks[1:] / c_quant).astype(np.int32)
    
    return quant_blocks
