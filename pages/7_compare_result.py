import streamlit as st
import numpy as np
from PIL import Image
import os

st.title("7️⃣ Compare Results")

# Dummy giải mã ảnh (ảnh cuối sau nén)
def get_decoded_image():
    # Trả về ảnh đen trắng giả lập với kích thước nhỏ
    return np.clip(np.random.randn(256, 256) * 30 + 128, 0, 255).astype(np.uint8)

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(mse, max_pixel=255.0):
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Load ảnh gốc
try:
    original = Image.open("assets/images/test/original.png")
except FileNotFoundError:
    st.warning("Vui lòng upload ảnh ở bước 1.")
    st.stop()

original_array = np.array(original)

# Kiểm tra ảnh gốc là ảnh màu (RGB)
if len(original_array.shape) == 3 and original_array.shape[2] == 3:
    # Nếu ảnh gốc là ảnh màu, chuyển ảnh giải mã thành ảnh màu có kích thước đúng
    decoded_array = get_decoded_image()
    decoded_image_resized = np.array(Image.fromarray(decoded_array).resize((original.width, original.height)))
    decoded_array_resized = np.repeat(decoded_image_resized[:, :, np.newaxis], 3, axis=2)  # Chuyển thành ảnh màu 3 kênh
else:
    # Nếu ảnh gốc là ảnh đen trắng, chỉ cần resize ảnh giải mã
    decoded_array = get_decoded_image()
    decoded_array_resized = np.array(Image.fromarray(decoded_array).resize(original.size))

decoded_image = Image.fromarray(decoded_array_resized)

# Tính toán MSE & PSNR
mse = calculate_mse(original_array, decoded_array_resized)
psnr = calculate_psnr(mse)

# Dummy size file (giả lập)
original_size = os.path.getsize("assets/images/test/original.png") / 1024  # KB
compressed_size = original_size * 0.35  # giả lập 65% nén

# Hiển thị ảnh
col1, col2 = st.columns(2)
with col1:
    st.image(original, caption="Ảnh gốc", use_container_width=True)
with col2:
    st.image(decoded_image, caption="Ảnh sau giải nén", use_container_width=True)

st.markdown("---")

# Thông số đánh giá
st.subheader("📈 Kết quả so sánh")
st.write(f"**PSNR:** {psnr:.2f} dB")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**Kích thước ảnh gốc:** {original_size:.2f} KB")
st.write(f"**Kích thước ảnh nén:** {compressed_size:.2f} KB (giả lập)")
