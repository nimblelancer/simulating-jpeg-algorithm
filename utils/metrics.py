import numpy as np
import os

def calculate_psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1: numpy array of the first image (H, W, C) or (H, W)
        img2: numpy array of the second image (H, W, C) or (H, W)
        
    Returns:
        float: PSNR value in decibels (dB). Returns inf if MSE is zero.
    """
    # Ensure images are numpy arrays
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2, dtype=np.float64)
    
    # Check if images have the same shape
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)
    
    # If MSE is zero, return infinity
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 255.0  # Assuming 8-bit images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def mse(img1, img2):
    """
    Tính toán Mean Squared Error (MSE) giữa hai ảnh.

    Tham số:
        img1 (np.ndarray): Ảnh gốc (H, W) hoặc (H, W, 3)
        img2 (np.ndarray): Ảnh so sánh (cùng kích thước với img1)

    Trả về:
        float: Giá trị MSE

    Ngoại lệ:
        ValueError: Nếu hai ảnh khác kích thước
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Ảnh có kích thước khác nhau: {img1.shape} vs {img2.shape}")
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def ssim(img1, img2, auto_convert=True):
    """
    Tính Structural Similarity Index (SSIM) giữa hai ảnh grayscale.

    Tham số:
        img1 (np.ndarray): Ảnh đầu vào 1 (grayscale hoặc RGB)
        img2 (np.ndarray): Ảnh đầu vào 2 (grayscale hoặc RGB)
        auto_convert (bool): Tự động chuyển RGB về grayscale nếu cần

    Trả về:
        dict: {
            "value": float hoặc None,
            "text": chuỗi mô tả (ví dụ: "0.9123" hoặc lỗi)
        }

    Ngoại lệ:
       
    """
    if img1.shape != img2.shape:
        return {"value": None, "text": f"Ảnh khác kích thước: {img1.shape} vs {img2.shape}"}
    
    if img1.ndim == 3 and auto_convert:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)
    elif img1.ndim != 2:
        return {"value": None, "text": "Ảnh không phải grayscale hoặc không thể chuyển đổi"}

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = 6.5025
    C2 = 58.5225

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
    ssim_val = numerator / denominator

    return {"value": ssim_val, "text": f"{ssim_val:.4f}"}

def compression_ratio(original_path, compressed_path):
    """
    Tính tỷ lệ nén giữa hai file ảnh.

    Tham số:
        original_path (str): Đường dẫn tới file ảnh gốc
        compressed_path (str): Đường dẫn tới file ảnh nén

    Trả về:
        dict: {
            "value": float hoặc None,
            "text": chuỗi mô tả (ví dụ: "2.15x" hoặc lỗi)
        }

    Ngoại lệ:
    """
    try:
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"File gốc không tồn tại: {original_path}")
        if not os.path.exists(compressed_path):
            raise FileNotFoundError(f"File nén không tồn tại: {compressed_path}")

        original_size = os.path.getsize(original_path)
        compressed_size = os.path.getsize(compressed_path)

        if compressed_size == 0:
            return {"value": None, "text": "File nén bằng 0 byte"}

        ratio = original_size / compressed_size
        return {"value": ratio, "text": f"{ratio:.2f}:1"}
    except Exception as e:
        return {"value": None, "text": str(e)}