import streamlit as st
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

st.title("6️⃣ Decode Image")

# Dummy function để pass logic
def decode_image_dummy():
    # Trả về ảnh đen trắng giả lập (giải mã từ DCT giả lập)
    return np.clip(np.random.randn(256, 256) * 30 + 128, 0, 255).astype(np.uint8)

# Hàm tính DCT 2D
def compute_dct(image_array):
    return dct(dct(image_array.T, norm='ortho').T, norm='ortho')

# Hàm tính IDCT (giải mã DCT)
def compute_idct(dct_array):
    return idct(idct(dct_array.T, norm='ortho').T, norm='ortho')

try:
    original_image = Image.open("assets/images/test/original.png")
except FileNotFoundError:
    st.warning("Vui lòng upload ảnh ở bước 1.")
    st.stop()

# Convert ảnh gốc thành mảng numpy
original_array = np.array(original_image.convert("L"))

# Kiểm tra kích thước của ảnh gốc và thay đổi kích thước ảnh giải mã sao cho khớp
decoded_array = decode_image_dummy()

# Chuyển ảnh giải mã thành kích thước giống ảnh gốc
decoded_array_resized = np.resize(decoded_array, original_array.shape)

# Tính DCT của ảnh gốc
dct_original = compute_dct(original_array)

# Tính DCT của ảnh đã giải mã đã thay đổi kích thước
dct_decoded = compute_dct(decoded_array_resized)

# Giải mã lại ảnh đã giải mã từ DCT (IDCT)
decoded_image = Image.fromarray(np.clip(compute_idct(dct_decoded), 0, 255).astype(np.uint8))

col1, col2 = st.columns(2)

with col1:
    st.image(original_image, caption="Ảnh gốc", use_container_width=True)

with col2:
    st.image(decoded_image, caption="Ảnh tái tạo sau giải mã", use_container_width=True)

# So sánh hệ số DCT gốc và sau giải nén
st.write("**So sánh hệ số DCT gốc và sau giải nén:**")
st.write("Dưới đây là sự khác biệt giữa các hệ số DCT của ảnh gốc và ảnh đã giải mã:")
st.write("Hệ số DCT gốc (một phần):")
st.write(dct_original[:5, :5])  # Hiển thị một phần của DCT gốc

st.write("Hệ số DCT của ảnh đã giải mã (một phần):")
st.write(dct_decoded[:5, :5])  # Hiển thị một phần của DCT ảnh giải mã

# Hiển thị sự khác biệt giữa DCT gốc và giải mã
dct_difference = np.abs(dct_original - dct_decoded)
st.write("Sự khác biệt giữa các hệ số DCT gốc và sau giải mã (một phần):")
st.write(dct_difference[:5, :5])  # Hiển thị sự khác biệt của một phần
