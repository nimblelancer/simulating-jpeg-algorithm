import numpy as np

def rgb_to_ycbcr(image):
    """
    Chuyển đổi ảnh RGB sang không gian màu YCbCr.
    
    Parameters:
    -----------
    image : ndarray
        Mảng NumPy 3D (H, W, 3) với các giá trị pixel RGB trong [0, 255]
    
    Returns:
    --------
    ndarray
        Mảng NumPy 3D (H, W, 3) gồm 3 kênh Y, Cb, Cr, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng hoặc giá trị pixel ngoài [0, 255]
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng (H, W, 3)")
    if image.max() > 255 or image.min() < 0:
        raise ValueError("Giá trị pixel RGB phải nằm trong [0, 255]")
    
    image = image.astype(np.float32)
    # Ma trận chuyển đổi
    coeffs = np.array([
        [0.299, 0.587, 0.114],      # Y
        [-0.1687, -0.3313, 0.5],    # Cb
        [0.5, -0.4187, -0.0813]     # Cr
    ])
    ycbcr = np.dot(image, coeffs.T)
    ycbcr[:, :, 1:] += 128  # Thêm offset cho Cb, Cr
    
    return np.clip(ycbcr, 0, 255)

def ycbcr_to_rgb(ycbcr):
    """
    Chuyển đổi ảnh YCbCr về không gian màu RGB.
    
    Parameters:
    -----------
    ycbcr : ndarray
        Mảng NumPy 3D (H, W, 3) gồm 3 kênh Y, Cb, Cr, giá trị trong [0, 255]
    
    Returns:
    --------
    ndarray
        Mảng NumPy 3D (H, W, 3) với giá trị RGB, dtype=uint8, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng hoặc giá trị pixel ngoài [0, 255]
    """
    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng (H, W, 3)")
    if ycbcr.max() > 255 or ycbcr.min() < 0:
        raise ValueError("Giá trị YCbCr phải nằm trong [0, 255]")
    
    ycbcr = ycbcr.astype(np.float32)
    # Ma trận chuyển đổi
    coeffs = np.array([
        [1.0, 0.0, 1.402],           # R
        [1.0, -0.344136, -0.714136], # G
        [1.0, 1.772, 0.0]            # B
    ])
    rgb = np.dot(ycbcr, coeffs.T)
    rgb[:, :, 0] += 1.402 * -128
    rgb[:, :, 1] += (-0.344136 * -128) + (-0.714136 * -128)
    rgb[:, :, 2] += 1.772 * -128
    
    return np.clip(rgb, 0, 255).astype(np.uint8)
