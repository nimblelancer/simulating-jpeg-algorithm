import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
from PIL import Image

BASE_DIR = os.path.join("assets", "images", "processing")

def get_image_size_info(original_path: str, compressed_path: str) -> dict:
    """
    Trả về kích thước file ảnh và bitrate cho ảnh gốc và ảnh nén.
    - size_bytes: kích thước file (byte)
    - size_kb: kích thước file (KB)
    - bitrate: bits per pixel (bpp)
    """
    result = {}

    # Kiểm tra kích thước và bitrate ảnh gốc
    if os.path.isfile(original_path):
        size_bytes = os.path.getsize(original_path)
        size_kb = size_bytes / 1024

        try:
            img = Image.open(original_path)
            width, height = img.size
            num_pixels = width * height
            bitrate = (size_bytes * 8) / num_pixels  # bits per pixel
            result["original"] = {
                "size_bytes": size_bytes,
                "size_kb": round(size_kb, 2),
                "bitrate_bpp": round(bitrate, 3),
            }
        except Exception as e:
            result["original"] = {"error": f"Could not open original image: {str(e)}"}
    else:
        result["original"] = {"error": "Original file not found"}

    # Kiểm tra kích thước và bitrate ảnh nén
    if os.path.isfile(compressed_path):
        size_bytes = os.path.getsize(compressed_path)
        size_kb = size_bytes / 1024

        try:
            img = Image.open(compressed_path)
            width, height = img.size
            num_pixels = width * height
            bitrate = (size_bytes * 8) / num_pixels  # bits per pixel
            result["compressed"] = {
                "size_bytes": size_bytes,
                "size_kb": round(size_kb, 2),
                "bitrate_bpp": round(bitrate, 3),
            }
        except Exception as e:
            result["compressed"] = {"error": f"Could not open compressed image: {str(e)}"}
    else:
        result["compressed"] = {"error": "Compressed file not found"}

    return result

def get_compression_ratio(original_path: str, compressed_path: str) -> dict:
    """
    Tính tỉ lệ nén giữa ảnh gốc và ảnh sau nén.
    - original_filename: tên ảnh gốc (ví dụ 'original.png')
    - compressed_filename: tên ảnh đã nén (ví dụ 'reconstructed.jpg')
    
    Trả về dict chứa kích thước và compression ratio.
    """

    if not os.path.isfile(original_path) or not os.path.isfile(compressed_path):
        return {"error": "One or both files not found."}

    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)

    if compressed_size == 0:
        return {"error": "Compressed file size is 0."}

    ratio = original_size / compressed_size

    return {
        "original_kb": round(original_size / 1024, 2),
        "compressed_kb": round(compressed_size / 1024, 2),
        "compression_ratio": round(ratio, 2),
    }

def calculate_image_metrics(original_path: str, compressed_path: str) -> dict:
    """
    Tính SSIM và PSNR giữa ảnh gốc và ảnh nén.
    Cần đảm bảo ảnh cùng kích thước và cùng mode (RGB hoặc Grayscale).
    """

    if not os.path.isfile(original_path) or not os.path.isfile(compressed_path):
        return {"error": "One or both image files not found."}

    try:
        original = Image.open(original_path).convert("RGB")
        compressed = Image.open(compressed_path).convert("RGB")
        original_np = np.array(original)
        compressed_np = np.array(compressed)

        if original_np.shape != compressed_np.shape:
            return {"error": "Images must have the same shape for SSIM/PSNR."}

        ssim_val = ssim(original_np, compressed_np, channel_axis=-1, data_range=255)
        psnr_val = psnr(original_np, compressed_np, data_range=255)

        return {
            "ssim": round(ssim_val, 4),
            "psnr": round(psnr_val, 2),
        }

    except Exception as e:
        return {"error": str(e)}

def analyze_compression(original_path: str, compressed_path: str) -> dict:
    size_info = get_image_size_info(original_path, compressed_path)
    ratio = get_compression_ratio(original_path, compressed_path)
    metrics = calculate_image_metrics(original_path, compressed_path)

    return {
        **size_info,
        "compression_ratio": ratio,
        **metrics
    }
