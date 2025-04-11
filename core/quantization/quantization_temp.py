import numpy as np
from utils.image_io import adjust_quant_tables

def quantize_block(dct_block, quant_table):
    """
    Lượng tử hóa một khối DCT 8x8.
    
    Input:
        dct_block: Khối hệ số DCT 8x8
        quant_table: Ma trận lượng tử hóa 8x8
    
    Output:
        Ma trận hệ số đã lượng tử hóa
    """
    # Chia hệ số DCT cho ma trận lượng tử và làm tròn
    return np.round(dct_block / quant_table)

def apply_quantization(dct_blocks, quality=50):
    """
    Áp dụng lượng tử hóa cho tất cả các khối DCT của ảnh.
    
    Input:
        dct_blocks: Mảng các khối hệ số DCT
                   - Shape có thể là (h, w, 8, 8) cho ảnh xám
                   - Hoặc (3, h, w, 8, 8) cho ảnh màu YCbCr
        quality: Hệ số chất lượng từ 1 đến 100
    
    Output:
        Mảng các khối đã lượng tử hóa với cùng kích thước
    """
    # Tạo ma trận lượng tử tương ứng với hệ số chất lượng
    y_quant, c_quant = adjust_quant_tables(quality)
    
    # Xác định hình dạng và loại ảnh
    is_color = len(dct_blocks.shape) == 5  # Nếu có 5 chiều, đó là ảnh màu
    
    # Khởi tạo mảng kết quả
    quant_blocks = np.zeros_like(dct_blocks)
    
    if is_color:  # Ảnh màu (YCbCr)
        c, h, w, _, _ = dct_blocks.shape
        
        # Lượng tử hóa cho từng kênh màu
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    # Sử dụng ma trận lượng tử khác nhau cho kênh sáng và kênh màu
                    if channel == 0:  # Kênh Y (sáng)
                        quant_blocks[channel, i, j] = quantize_block(dct_blocks[channel, i, j], y_quant)
                    else:  # Kênh Cb hoặc Cr (màu)
                        quant_blocks[channel, i, j] = quantize_block(dct_blocks[channel, i, j], c_quant)
    else:  # Ảnh xám
        h, w, _, _ = dct_blocks.shape
        for i in range(h):
            for j in range(w):
                # Chỉ sử dụng ma trận lượng tử sáng cho ảnh xám
                quant_blocks[i, j] = quantize_block(dct_blocks[i, j], y_quant)
    
    return quant_blocks

def optimize_quantization_for_speed(dct_blocks, quality=50):
    """
    Phiên bản tối ưu hóa hiệu suất của quá trình lượng tử hóa.
    Sử dụng phép tính vec-tơ hóa của numpy thay vì vòng lặp.
    
    Input:
        dct_blocks: Mảng các khối hệ số DCT
        quality: Hệ số chất lượng từ 1 đến 100
    
    Output:
        Mảng các khối đã lượng tử hóa
    """
    # Tạo ma trận lượng tử tương ứng với hệ số chất lượng
    y_quant, c_quant = adjust_quant_tables(quality)
    
    # Xác định hình dạng và loại ảnh
    is_color = len(dct_blocks.shape) == 5
    
    if is_color:
        c, h, w, block_h, block_w = dct_blocks.shape
        quant_blocks = np.zeros_like(dct_blocks)
        
        # Lượng tử hóa từng kênh màu một cách vec-tơ hóa
        # Kênh Y (sáng)
        quant_blocks[0] = np.round(dct_blocks[0] / y_quant)
        
        # Kênh Cb và Cr (màu)
        for channel in range(1, c):
            quant_blocks[channel] = np.round(dct_blocks[channel] / c_quant)
    else:
        # Ảnh xám - lượng tử hóa tất cả các khối cùng lúc
        quant_blocks = np.round(dct_blocks / y_quant)
    
    return quant_blocks

