# File: pages/3_quantization.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("3️⃣ Quantization")

# Dummy dữ liệu
np.random.seed(42)
dct_before = np.random.randn(8, 8) * 50
quant_matrix = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])
dct_after = np.round(dct_before / quant_matrix)

# Hiển thị ảnh DCT trước và sau lượng tử hóa
col1, col2 = st.columns(2)
with col1:
    st.write("### ⬅️ Trước lượng tử hóa (DCT 8x8)")
    st.dataframe(np.round(dct_before, 2))
with col2:
    st.write("### ➡️ Sau lượng tử hóa")
    st.dataframe(dct_after.astype(int))

st.markdown("---")

# Hiển thị ma trận lượng tử hóa chuẩn
st.write("### 📏 Ma trận lượng tử hóa JPEG chuẩn")
st.dataframe(quant_matrix.astype(int))