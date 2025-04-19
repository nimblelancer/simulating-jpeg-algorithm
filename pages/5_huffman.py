import streamlit as st
import numpy as np
from PIL import Image

st.title("5️⃣ Huffman Coding")

# Dummy hàm giả lập
def build_huffman_tree(data):
    return {"0": "A", "10": "B", "11": "C"}

def huffman_encode(data):
    return "011011010"

# Mở ảnh gốc
image = Image.open("assets/images/test/original.png")
img_array = np.array(image)

# Chỉ hiển thị ảnh gốc với kích thước điều chỉnh theo cột
st.image(image, caption="Ảnh gốc", width=500)

# Hiển thị cây Huffman
st.write("**Cây Huffman giả lập:**")
tree = build_huffman_tree(img_array)
st.json(tree)

# Hiển thị chuỗi bit Huffman
st.write("**Chuỗi bit sau khi mã hóa:**")
bitstring = huffman_encode(img_array)
st.code(bitstring, language="text")
