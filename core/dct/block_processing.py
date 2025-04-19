import numpy as np

def pad_image_to_multiple_of_8(image):
    """
    Thêm padding để chiều cao và chiều rộng của ảnh chia hết cho 8.
    
    Parameters:
    -----------
    image : ndarray
        Ảnh 2D (H, W) hoặc 3D (H, W, C), dtype=float32, giá trị trong [0, 255]
    
    Returns:
    --------
    ndarray
        Ảnh sau khi pad, shape (H', W') hoặc (H', W', C) với H', W' chia hết cho 8,
        dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng hoặc giá trị pixel ngoài [0, 255]
    """
    if image.ndim not in (2, 3):
        raise ValueError("Ảnh đầu vào phải là mảng 2D hoặc 3D")
    if image.max() > 255 or image.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    if image.ndim == 2:
        H, W = image.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        return np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge').astype(np.float32)
    
    H, W, C = image.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge').astype(np.float32)

def split_into_blocks(image):
    """
    Chia ảnh thành các khối 8x8.
    
    Parameters:
    -----------
    image : ndarray
        Ảnh 2D (H, W) hoặc 3D (H, W, C), dtype=float32, giá trị trong [0, 255],
        H và W phải chia hết cho 8
    
    Returns:
    --------
    ndarray
        - 2D: shape (H//8, W//8, 8, 8)
        - 3D: shape (C, H//8, W//8, 8, 8)
        dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng, H/W không chia hết cho 8, hoặc giá trị pixel ngoài [0, 255]
    """
    if image.ndim not in (2, 3):
        raise ValueError("Ảnh đầu vào phải là mảng 2D hoặc 3D")
    if image.max() > 255 or image.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    if image.ndim == 2:
        H, W = image.shape
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError("Chiều cao và chiều rộng phải chia hết cho 8")
        h_blocks, w_blocks = H // 8, W // 8
        blocks = image.reshape(h_blocks, 8, w_blocks, 8)
        return blocks.transpose(0, 2, 1, 3).astype(np.float32)  # (h_blocks, w_blocks, 8, 8)
    
    H, W, C = image.shape
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError("Chiều cao và chiều rộng phải chia hết cho 8")
    h_blocks, w_blocks = H // 8, W // 8
    blocks = image.transpose(2, 0, 1).reshape(C, h_blocks, 8, w_blocks, 8)
    return blocks.transpose(0, 1, 3, 2, 4).astype(np.float32)  # (C, h_blocks, w_blocks, 8, 8)

def merge_blocks(blocks):
    """
    Ghép các khối 8x8 thành ảnh đầy đủ.
    
    Parameters:
    -----------
    blocks : ndarray
        - 4D: shape (H//8, W//8, 8, 8) cho ảnh 2D
        - 5D: shape (C, H//8, W//8, 8, 8) cho ảnh 3D
        dtype=float32, giá trị trong [0, 255]
    
    Returns:
    --------
    ndarray
        Ảnh 2D (H, W) hoặc 3D (H, W, C), dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng hoặc giá trị pixel ngoài [0, 255]
    """
    if blocks.ndim not in (4, 5):
        raise ValueError("Khối đầu vào phải là mảng 4D hoặc 5D")
    if blocks.max() > 255 or blocks.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    if blocks.ndim == 4:
        h_blocks, w_blocks, block_h, block_w = blocks.shape
        if block_h != 8 or block_w != 8:
            raise ValueError("Kích thước khối phải là 8x8")
        image = blocks.transpose(0, 2, 1, 3).reshape(h_blocks * 8, w_blocks * 8)
        return image.astype(np.float32)
    
    C, h_blocks, w_blocks, block_h, block_w = blocks.shape
    if block_h != 8 or block_w != 8:
        raise ValueError("Kích thước khối phải là 8x8")
    image = blocks.transpose(0, 1, 3, 2, 4).reshape(C, h_blocks * 8, w_blocks * 8)
    return image.transpose(1, 2, 0).astype(np.float32)  # (H, W, C)