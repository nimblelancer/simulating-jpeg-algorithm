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
    
    Input:
        vector: Vector đầu vào có độ dài 8
    
    Output:
        Vector kết quả sau khi áp dụng DCT 1D
    """
    n = len(vector)
    result = np.zeros(n)
    
    for k in range(n):
        ck = 1 / np.sqrt(2) if k == 0 else 1
        sum_val = 0
        for i in range(n):
            sum_val += vector[i] * np.cos((2 * i + 1) * k * np.pi / (2 * n))
        result[k] = ck * sum_val * np.sqrt(2 / n)
    
    return result

def dct_2d_separable(block):
    """
    Thực hiện DCT 2D trên một khối 8x8 bằng cách sử dụng phương pháp tách biệt.
    
    Input:
        block: Ma trận 8x8 đầu vào
    
    Output:
        Ma trận 8x8 sau khi áp dụng DCT
    """
    # Kiểm tra kích thước đầu vào
    if block.shape != (8, 8):
        raise ValueError("Block phải có kích thước 8x8")
    
    # Trừ 128 để dịch giá trị về khoảng [-128, 127]
    block = block - 128
    
    # Áp dụng DCT 1D cho mỗi hàng
    temp = np.zeros((8, 8))
    for i in range(8):
        temp[i, :] = dct_1d(block[i, :])
    
    # Áp dụng DCT 1D cho mỗi cột của kết quả trung gian
    result = np.zeros((8, 8))
    for j in range(8):
        result[:, j] = dct_1d(temp[:, j])
    
    return result

def apply_dct_to_image(image_blocks):
    """
    Áp dụng DCT cho tất cả các khối 8x8 của ảnh.
    
    Input:
        image_blocks: Mảng 4D (h, w, 8, 8) chứa các khối 8x8 của ảnh
                     hoặc mảng 5D (c, h, w, 8, 8) nếu có nhiều kênh màu
    
    Output:
        Mảng các khối sau khi áp dụng DCT với cùng kích thước đầu vào
    """
    # Xác định hình dạng của mảng đầu vào
    is_multichannel = len(image_blocks.shape) == 5
    
    if is_multichannel:
        c, h, w, block_h, block_w = image_blocks.shape
        dct_blocks = np.zeros_like(image_blocks)
        
        # Áp dụng DCT cho từng kênh màu và từng khối
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    dct_blocks[channel, i, j] = dct_2d_separable(image_blocks[channel, i, j])
    else:
        h, w, block_h, block_w = image_blocks.shape
        dct_blocks = np.zeros_like(image_blocks)
        
        # Áp dụng DCT cho từng khối
        for i in range(h):
            for j in range(w):
                dct_blocks[i, j] = dct_2d_separable(image_blocks[i, j])
    
    return dct_blocks

# endregion DCT

# region IDCT
def idct_1d(vector):
    """
    Thực hiện IDCT 1D trên một vector độ dài 8.
    
    Input:
        vector: Vector đầu vào có độ dài 8 (các hệ số DCT)
    
    Output:
        Vector kết quả sau khi áp dụng IDCT 1D
    """
    n = len(vector)
    result = np.zeros(n)
    
    for i in range(n):
        sum_val = 0
        for k in range(n):
            ck = 1 / np.sqrt(2) if k == 0 else 1
            sum_val += ck * vector[k] * np.cos((2 * i + 1) * k * np.pi / (2 * n))
        result[i] = sum_val * np.sqrt(2 / n)
    
    return result

def idct_2d_separable(block):
    """
    Thực hiện IDCT 2D trên một khối 8x8 bằng cách sử dụng phương pháp tách biệt.
    
    Input:
        block: Ma trận 8x8 đầu vào chứa các hệ số DCT
    
    Output:
        Ma trận 8x8 sau khi áp dụng IDCT, đại diện cho khối pixel ban đầu
    """
    # Kiểm tra kích thước đầu vào
    if block.shape != (8, 8):
        raise ValueError("Block phải có kích thước 8x8")
    
    # Áp dụng IDCT 1D cho mỗi cột
    temp = np.zeros((8, 8))
    for j in range(8):
        temp[:, j] = idct_1d(block[:, j])
    
    # Áp dụng IDCT 1D cho mỗi hàng của kết quả trung gian
    result = np.zeros((8, 8))
    for i in range(8):
        result[i, :] = idct_1d(temp[i, :])
    
    # Cộng lại 128 để chuyển về khoảng giá trị pixel [0, 255]
    result = result + 128
    
    # Giới hạn giá trị pixel trong khoảng [0, 255]
    result = np.clip(result, 0, 255)
    
    return result

def apply_idct_to_image(dct_blocks):
    """
    Áp dụng IDCT cho tất cả các khối 8x8 của ảnh đã được nén.
    
    Input:
        dct_blocks: Mảng 4D (h, w, 8, 8) chứa các khối 8x8 của hệ số DCT
                   hoặc mảng 5D (c, h, w, 8, 8) nếu có nhiều kênh màu
    
    Output:
        Mảng các khối pixel sau khi áp dụng IDCT với cùng kích thước đầu vào
    """
    # Xác định hình dạng của mảng đầu vào
    is_multichannel = len(dct_blocks.shape) == 5
    
    if is_multichannel:
        c, h, w, block_h, block_w = dct_blocks.shape
        pixel_blocks = np.zeros_like(dct_blocks)
        
        # Áp dụng IDCT cho từng kênh màu và từng khối
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    pixel_blocks[channel, i, j] = idct_2d_separable(dct_blocks[channel, i, j])
    else:
        h, w, block_h, block_w = dct_blocks.shape
        pixel_blocks = np.zeros_like(dct_blocks)
        
        # Áp dụng IDCT cho từng khối
        for i in range(h):
            for j in range(w):
                pixel_blocks[i, j] = idct_2d_separable(dct_blocks[i, j])
    
    return pixel_blocks
# endregion IDCT

