import numpy as np

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