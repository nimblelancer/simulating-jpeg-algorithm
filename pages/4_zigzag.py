# File: pages/4_zigzag.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("4️⃣ ZigZag Scan")

# Dummy dữ liệu: ma trận lượng tử hóa 8x8
matrix = np.arange(1, 65).reshape(8, 8)

# Dummy zigzag (thứ tự đọc đơn giản)
def zigzag_order(matrix):
    # Chỉ mô phỏng: flatten theo thứ tự dummy
    return matrix.flatten()

zigzag_vector = zigzag_order(matrix)

# Hiển thị ma trận gốc
st.write("### 🧮 Ma trận sau lượng tử hóa (8x8)")
st.dataframe(matrix)

# Hiển thị vector zigzag
st.write("### 🧵 Vector ZigZag hóa")
st.write(zigzag_vector)

# Hiển thị minh họa đường đi zigzag (giả lập)
st.write("### 🎯 Minh họa đường đi ZigZag")
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.ones_like(matrix), cmap="gray", vmin=0, vmax=1)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{matrix[i, j]}", va='center', ha='center', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
st.pyplot(fig)
