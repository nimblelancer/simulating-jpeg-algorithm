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

    y_quant = np.clip(np.floor((y_table * scale + 50) / 100), 1, 255)
    c_quant = np.clip(np.floor((c_table * scale + 50) / 100), 1, 255)

    _quant_table_cache[quality] = (y_quant, c_quant)
    return y_quant, c_quant


def optimize_quantization_for_speed(dct_blocks, quality=50):
    """
    Lượng tử hóa toàn bộ khối DCT, hỗ trợ ảnh xám (4D) và ảnh màu (5D), dùng vector hóa.
    """
    if dct_blocks.ndim not in (4, 5) or dct_blocks.shape[-2:] != (8, 8):
        raise ValueError("dct_blocks phải là mảng 4D (h,w,8,8) hoặc 5D (c,h,w,8,8)")
    if not 1 <= quality <= 100:
        raise ValueError("Hệ số chất lượng phải từ 1 đến 100")
    
    y_quant, c_quant = adjust_quant_tables(quality)

    # Dùng float32 cho chia, sau đó ép về int32
    dct_blocks = dct_blocks.astype(np.float32)

    if dct_blocks.ndim == 4:
        # ảnh xám: toàn bộ khối dùng y_quant
        quant = np.round(dct_blocks / y_quant).astype(np.int32)
    else:
        # ảnh màu: mỗi channel dùng quant khác nhau
        c, h, w = dct_blocks.shape[:3]
        quant = np.empty_like(dct_blocks, dtype=np.int32)

        for ch in range(c):
            q = y_quant if ch == 0 else c_quant  # Y dùng y_quant, còn lại dùng c_quant
            # Broadcasting tự động
            quant[ch] = np.round(dct_blocks[ch] / q).astype(np.int32)

    return quant

