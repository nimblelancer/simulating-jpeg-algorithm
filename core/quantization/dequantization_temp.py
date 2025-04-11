import numpy as np
from utils.image_io import adjust_quant_tables

def dequantize_block(quant_block, quant_table):
    """
    Giải lượng tử hóa một khối 8x8.
    
    Input:
        quant_block: Khối hệ số đã lượng tử hóa 8x8
        quant_table: Ma trận lượng tử hóa 8x8
    
    Output:
        Ma trận hệ số DCT xấp xỉ ban đầu
    """
    # Nhân hệ số đã lượng tử hóa với ma trận lượng tử
    return quant_block * quant_table

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

def optimize_dequantization_for_speed(quant_blocks, quality=50):
    """
    Phiên bản tối ưu hóa hiệu suất của quá trình giải lượng tử hóa.
    Sử dụng phép tính vec-tơ hóa của numpy thay vì vòng lặp.
    
    Input:
        quant_blocks: Mảng các khối hệ số đã lượng tử hóa
        quality: Hệ số chất lượng từ 1 đến 100
    
    Output:
        Mảng các khối hệ số DCT xấp xỉ ban đầu
    """
    # Tạo ma trận lượng tử tương ứng với hệ số chất lượng
    y_quant, c_quant = adjust_quant_tables(quality)
    
    # Xác định hình dạng và loại ảnh
    is_color = len(quant_blocks.shape) == 5
    
    # Tạo mảng kết quả với kiểu dữ liệu float32
    dct_blocks = np.zeros_like(quant_blocks, dtype=np.float32)
    
    if is_color:
        c, h, w, block_h, block_w = quant_blocks.shape
        
        # Giải lượng tử hóa từng kênh màu một cách vec-tơ hóa
        # Kênh Y (sáng)
        dct_blocks[0] = quant_blocks[0] * y_quant
        
        # Kênh Cb và Cr (màu)
        for channel in range(1, c):
            dct_blocks[channel] = quant_blocks[channel] * c_quant
    else:
        # Ảnh xám - giải lượng tử hóa tất cả các khối cùng lúc
        dct_blocks = quant_blocks * y_quant
    
    return dct_blocks