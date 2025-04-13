import numpy as np
from quantization.quantization_temp import adjust_quant_tables

def dequantize_block(quant_block, quant_table):
    """
    Giải lượng tử hóa một khối 8x8.
    
    Parameters:
    -----------
    quant_block : ndarray
        Khối hệ số lượng tử hóa, shape (8, 8), dtype=int32
    quant_table : ndarray
        Bảng lượng tử hóa, shape (8, 8), dtype=float32, giá trị > 0
    
    Returns:
    --------
    ndarray
        Khối hệ số DCT xấp xỉ, shape (8, 8), dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc quant_table có giá trị không hợp lệ
    """
    if quant_block.shape != (8, 8) or quant_table.shape != (8, 8):
        raise ValueError("quant_block và quant_table phải có shape (8, 8)")
    if not np.all(quant_table > 0):
        raise ValueError("Bảng lượng tử hóa phải có tất cả giá trị > 0")
    
    quant_block = quant_block.astype(np.int32)
    quant_table = quant_table.astype(np.float32)
    return (quant_block * quant_table).astype(np.float32)

def optimize_dequantization_for_speed(quant_blocks, quality=50):
    """
    Giải lượng tử hóa vector hóa cho tất cả các khối.
    
    Parameters:
    -----------
    quant_blocks : ndarray
        - 4D: shape (h, w, 8, 8) cho ảnh xám
        - 5D: shape (c, h, w, 8, 8) cho ảnh màu
        dtype=int32
    quality : int
        Hệ số chất lượng từ 1 đến 100, default=50
    
    Returns:
    --------
    ndarray
        Các khối hệ số DCT xấp xỉ, shape giống đầu vào, dtype=float32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc quality ngoài [1, 100]
    """
    if quant_blocks.ndim not in (4, 5) or quant_blocks.shape[-2:] != (8, 8):
        raise ValueError("quant_blocks phải là mảng 4D (h, w, 8, 8) hoặc 5D (c, h, w, 8, 8)")
    if not 1 <= quality <= 100:
        raise ValueError("Hệ số chất lượng phải từ 1 đến 100")
    
    y_quant, c_quant = adjust_quant_tables(quality)
    dct_blocks = np.zeros_like(quant_blocks, dtype=np.float32)
    
    if quant_blocks.ndim == 4:
        dct_blocks = (quant_blocks * y_quant).astype(np.float32)
    else:
        dct_blocks[0] = (quant_blocks[0] * y_quant).astype(np.float32)
        dct_blocks[1:] = (quant_blocks[1:] * c_quant).astype(np.float32)
    
    return dct_blocks

def apply_dequantization(quant_blocks, quality=50):
    """
    Áp dụng giải lượng tử hóa cho tất cả các khối đã lượng tử hóa.
    
    Input:
        quant_blocks: Mảng các khối hệ số đã lượng tử hóa
        quality: Hệ số chất lượng từ 1 đến 100
    
    Output:
        Mảng các khối hệ số DCT xấp xỉ ban đầu
    """
    # Tạo ma trận lượng tử tương ứng với hệ số chất lượng
    y_quant, c_quant = adjust_quant_tables(quality)
    
    # Xác định hình dạng và loại ảnh
    is_color = len(quant_blocks.shape) == 5  # Nếu có 5 chiều, đó là ảnh màu
    
    # Khởi tạo mảng kết quả
    dct_blocks = np.zeros_like(quant_blocks, dtype=np.float32)
    
    if is_color:  # Ảnh màu (YCbCr)
        c, h, w, _, _ = quant_blocks.shape
        
        # Giải lượng tử hóa cho từng kênh màu
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    # Sử dụng ma trận lượng tử khác nhau cho kênh sáng và kênh màu
                    if channel == 0:  # Kênh Y (sáng)
                        dct_blocks[channel, i, j] = dequantize_block(quant_blocks[channel, i, j], y_quant)
                    else:  # Kênh Cb hoặc Cr (màu)
                        dct_blocks[channel, i, j] = dequantize_block(quant_blocks[channel, i, j], c_quant)
    else:  # Ảnh xám
        h, w, _, _ = quant_blocks.shape
        for i in range(h):
            for j in range(w):
                # Chỉ sử dụng ma trận lượng tử sáng cho ảnh xám
                dct_blocks[i, j] = dequantize_block(quant_blocks[i, j], y_quant)
    
    return dct_blocks