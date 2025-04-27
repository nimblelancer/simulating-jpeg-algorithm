import csv
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from utils.image_io import load_uploaded_image
from jpeg_processor import JPEGProcessor

# Đường dẫn file ảnh gốc
uploaded_file = "assets/images/test/input_gray.jpg"  # <-- Đổi thành đúng đường dẫn của bạn

# Đọc ảnh gốc
original_image = load_uploaded_image(uploaded_file)
original_array = np.array(original_image)

# File CSV để lưu kết quả
output_csv = "jpeg_psnr_ssim_results.csv"

# Ghi header cho file CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Quality Factor", "PSNR", "SSIM"])

# Vòng lặp qua các giá trị quality factor từ 1 đến 100
for quality_factor in range(1, 101):
    jpeg = JPEGProcessor(quality_factor)
    
    # Encode
    result = jpeg.encode_pipeline(original_image)
    
    # Decode
    decompressed_image = jpeg.decode_pipeline(
        result['encoded_data'],
        result['dc_codes'],
        result['ac_codes'],
        result['padded_shape'],
        result['total_bits'],
        original_array.shape
    )

    # Đảm bảo ảnh giải nén đúng kiểu ndarray
    if isinstance(decompressed_image, Image.Image):
        decompressed_array = np.array(decompressed_image)
    else:
        decompressed_array = decompressed_image

    # Tính PSNR và SSIM
    current_psnr = psnr(original_array, decompressed_array, data_range=255)
    current_ssim = ssim(original_array, decompressed_array, channel_axis=-1, data_range=255)

    # Ghi kết quả vào file CSV
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([quality_factor, current_psnr, current_ssim])

print(f"Hoàn thành! Kết quả đã lưu vào {output_csv}")
