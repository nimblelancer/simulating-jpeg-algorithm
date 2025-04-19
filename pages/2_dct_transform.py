import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

st.title("2️⃣ DCT Transform")

# Dummy DCT (chỉ để hiển thị giao diện)
def dummy_dct(image_array):
    return np.random.randn(*image_array.shape) * 50

try:
    image = Image.open("assets/images/test/original.png")  
except FileNotFoundError:
    st.warning("Vui lòng upload ảnh ở bước 1 trước.")
    st.stop()

image_array = np.array(image)
dct_array = dummy_dct(image_array)

# Hiển thị ảnh gốc
st.write("### 🖼️ Ảnh gốc")
st.image(image, caption="Ảnh gốc", width=500)

# 🔥 Hiển thị heatmap với kích thước phù hợp
st.write("### 🔥 Ma trận DCT (log-scaled heatmap)")

dct_log = np.log1p(np.abs(dct_array))

# Tạo figure nhỏ hơn
fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

# Vẽ heatmap
cax = ax.imshow(dct_log, cmap='hot')
ax.tick_params(labelsize=6)
ax.set_title("DCT Heatmap", fontsize=8)
ax.set_xlabel("Tọa độ X", fontsize=7)
ax.set_ylabel("Tọa độ Y", fontsize=7)

# Colorbar gọn lại
cbar = fig.colorbar(cax, ax=ax, shrink=0.6)
cbar.ax.tick_params(labelsize=6)

# Tối ưu khoảng cách, padding
fig.tight_layout(pad=0.5)

# Dùng bbox_inches để cắt phần dư
st.pyplot(fig, bbox_inches='tight')



channel = st.selectbox("Chọn kênh màu để xem Block DCT", ["Red", "Green", "Blue", "Trung bình"])
if channel == "Trung bình":
    block = np.mean(dct_array[:8, :8, :], axis=2)
else:
    idx = {"Red": 0, "Green": 1, "Blue": 2}[channel]
    block = dct_array[:8, :8, idx]

st.write(f"### 🔍 Block DCT 8x8 đầu tiên - {channel}")
st.dataframe(np.round(block, 2))


