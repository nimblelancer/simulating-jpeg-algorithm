import numpy as np

def rgb_to_ycbcr(image):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng (H, W, 3)")
    if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Ảnh phải có dtype số (int hoặc float)")
    if np.isnan(image).any() or np.isinf(image).any():
        raise ValueError("Ảnh không được chứa NaN hoặc Inf")
    if image.max() > 255 or image.min() < 0:
        raise ValueError("Giá trị pixel RGB phải nằm trong [0, 255]")
    
    image = image.astype(np.float32)
    coeffs = np.array([
        [0.299, 0.587, 0.114],
        [-0.1687, -0.3313, 0.5],
        [0.5, -0.4187, -0.0813]
    ])
    ycbcr = image @ coeffs.T
    ycbcr[:, :, 1:] += 128
    return np.clip(ycbcr, 0, 255).astype(np.uint8)

def ycbcr_to_rgb(ycbcr):
    if ycbcr.ndim != 3 or ycbcr.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng (H, W, 3)")
    if not np.issubdtype(ycbcr.dtype, np.integer) and not np.issubdtype(ycbcr.dtype, np.floating):
        raise ValueError("Ảnh phải có dtype số (int hoặc float)")
    if np.isnan(ycbcr).any() or np.isinf(ycbcr).any():
        raise ValueError("Ảnh không được chứa NaN hoặc Inf")
    if ycbcr.max() > 255 or ycbcr.min() < 0:
        raise ValueError("Giá trị YCbCr phải nằm trong [0, 255]")

    ycbcr = ycbcr.astype(np.float32)
    ycbcr[:, :, 1:] -= 128  # Trừ offset trước
    coeffs = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    rgb = ycbcr @ coeffs.T
    return np.clip(rgb, 0, 255).astype(np.uint8)
