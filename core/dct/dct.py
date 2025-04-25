import numpy as np

def dct_2d_separable(block):
    """
    Thực hiện DCT 2D trên một khối 8x8, sử dụng ma trận hóa thay vì loop để tăng tốc.
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")

    block = block.astype(np.float32)
    C = _dct_matrix(8)
    return C @ (block - 128.0) @ C.T

def _dct_matrix(n):
    C = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
            C[k, i] = alpha * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))
    return C.astype(np.float32)

def apply_dct_to_image(image_blocks):
    """
    Áp dụng DCT cho tất cả các khối 8x8 của ảnh bằng dct_2d_separable.
    image_blocks: 4D (h,w,8,8) hoặc 5D (c,h,w,8,8), giá trị [0,255]
    """
    if image_blocks.ndim not in (4, 5):
        raise ValueError("image_blocks phải là mảng 4D hoặc 5D")
    if image_blocks.shape[-2:] != (8, 8):
        raise ValueError("Kích thước khối phải là 8x8")
    if image_blocks.max() > 255 or image_blocks.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255] trước level-shift")

    blocks = image_blocks.astype(np.float32)
    shape = blocks.shape
    flat_blocks = blocks.reshape(-1, 8, 8)

    # DCT tăng tốc với einsum
    C = _dct_matrix(8)
    C_T = C.T
    dct_flat = np.einsum('ij,njk,kl->nil', C, flat_blocks - 128.0, C_T)

    return dct_flat.reshape(shape)

def _idct_matrix(n):
    C = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            alpha = np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)
            C[i, k] = alpha * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))
    return C.astype(np.float32)

def apply_idct_to_image(dct_blocks):
    """
    Áp dụng IDCT 2D hiệu suất cao, hỗ trợ cả ảnh xám (4D) và ảnh màu (5D).
    """
    if dct_blocks.ndim not in (4, 5) or dct_blocks.shape[-2:] != (8, 8):
        raise ValueError("dct_blocks phải là mảng 4D hoặc 5D với block size 8x8")

    blocks = dct_blocks.astype(np.float32)
    shape = blocks.shape
    flat_blocks = blocks.reshape(-1, 8, 8)

    C = _idct_matrix(8)
    C_T = C.T
    idct_flat = np.einsum('ij,njk,kl->nil', C, flat_blocks, C_T)
    idct_flat = np.clip(np.round(idct_flat + 128.0), 0, 255).astype(np.float32)

    return idct_flat.reshape(shape)
