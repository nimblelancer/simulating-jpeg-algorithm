import numpy as np

def dct_1d(vector):
    """
    Thực hiện DCT 1D trên một vector độ dài 8, chuẩn hóa 'ortho'.
    """
    if vector.shape != (8,):
        raise ValueError("Vector phải có độ dài 8")
    vector = vector.astype(np.float32)
    n = 8
    result = np.zeros(n, dtype=np.float32)

    for k in range(n):
        alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
        sum_val = 0
        for i in range(n):
            sum_val += vector[i] * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))
        result[k] = alpha * sum_val

    return result

def dct_2d_separable(block):
    """
    Thực hiện DCT 2D trên một khối 8x8, sử dụng dct_1d.
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    # Level-shift chuẩn JPEG
    shifted = block.astype(np.float32) - 128.0
    # DCT theo chiều hàng
    temp = np.apply_along_axis(dct_1d, 1, shifted)
    # DCT theo chiều cột
    result = np.apply_along_axis(dct_1d, 0, temp)
    return result

def apply_dct_to_image(image_blocks):
    """
    Áp dụng DCT cho tất cả các khối 8x8 của ảnh bằng dct_2d_separable.
    image_blocks: 4D (h,w,8,8) hoặc 5D (c,h,w,8,8), giá trị [0,255]
    """
    if image_blocks.ndim not in (4, 5):
        raise ValueError("image_blocks phải là mảng 4D hoặc 5D")
    if image_blocks.shape[-2:] != (8, 8):
        raise ValueError("Kích thước khối phải là 8x8")
    # Kiểm tra giá trị pixel trước level-shift
    if image_blocks.max() > 255 or image_blocks.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255] trước level-shift")

    blocks = image_blocks.astype(np.float32)
    shape = blocks.shape
    flat_blocks = blocks.reshape(-1, 8, 8)

    dct_flat = np.stack([dct_2d_separable(b) for b in flat_blocks])
    return dct_flat.reshape(shape)

def idct_1d(vector):
    """
    Thực hiện IDCT 1D với chuẩn hóa 'ortho', tương đương scipy.idct(type=2, norm='ortho').
    """
    if vector.shape != (8,):
        raise ValueError("Vector phải có độ dài 8")
    n = 8
    result = np.zeros(n, dtype=np.float32)

    for i in range(n):
        sum_val = 0
        for k in range(n):
            alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
            sum_val += alpha * vector[k] * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))
        result[i] = sum_val

    return result

def idct_2d_separable(block):
    """
    Thực hiện IDCT 2D với chuẩn hóa 'ortho', theo cách separable: 2 lần IDCT 1D.
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    block = block.astype(np.float32)

    # IDCT theo cột trước, rồi hàng
    temp = np.apply_along_axis(idct_1d, 0, block)
    result = np.apply_along_axis(idct_1d, 1, temp)

    # Cộng lại 128 (level-shift ngược) → clip giá trị
    return np.clip(np.round(result + 128.0), 0, 255).astype(np.float32)


def apply_idct_to_image(dct_blocks):
    """
    Áp dụng IDCT 2D thủ công (separable), hỗ trợ cả ảnh xám (4D) và ảnh màu (5D).
    """
    if dct_blocks.ndim not in (4, 5) or dct_blocks.shape[-2:] != (8, 8):
        raise ValueError("dct_blocks phải là mảng 4D hoặc 5D với block size 8x8")

    blocks = dct_blocks.astype(np.float32)
    shape = blocks.shape
    flat_blocks = blocks.reshape(-1, 8, 8)

    idct_flat = np.stack([idct_2d_separable(b) for b in flat_blocks])
    return idct_flat.reshape(shape)
