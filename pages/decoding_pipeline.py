import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from utils.image_io import load_npy

def app():
    st.title("🔄 JPEG Decoding Pipeline")
    st.write("Explore each step of the JPEG decoding process.")

    # Load dữ liệu decode
    inverse_zigzag_vectors = load_npy("decode_step_huffman_decode.npy", allow_pickle=True)
    dequantized_blocks = load_npy("decode_step_inverse_zigzag.npy")
    idct_blocks = load_npy("decode_step_dequantized.npy")
    blocks = load_npy("decode_step_idct.npy")

    # Số block
    num_blocks = blocks.shape[0]
    block_height_idx = st.slider("Select Block Height Index", 0, blocks.shape[0] - 1, 0)
    block_width_idx = st.slider("Select Block Width Index", 0, blocks.shape[1]- 1, 0)
    block_idx = block_height_idx * blocks.shape[1] + block_width_idx

    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Huffman Decoding",
        "🔀 Inverse ZigZag",
        "🔁 Dequantization",
        "📥 Inverse DCT"
    ])

    with tab1:
        st.subheader("📤 Huffman Decoding")
        st.markdown("Giải mã bitstream đã nén sử dụng bảng mã Huffman.")

        st.markdown("Vector sau khi giải mã Huffman:")
        vector = inverse_zigzag_vectors[block_idx]
        dc = int(vector[0])
        ac = [(int(run), int(val)) for run, val in vector[1]]
        zigzag_vector = [dc, ac]
        st.code(f"{zigzag_vector}")

        st.markdown(f"- DC coefficient: Giá trị tần số thấp (sáng/tối của block).")
        st.markdown(f"- (run, value): Số lượng giá trị 0 trước một giá trị khác 0, và giá trị đó.")
        st.markdown(f"- (0, 0): Dấu hiệu kết thúc block (EOB).")

    with tab2:
        st.subheader("🔀 Inverse ZigZag")
        st.markdown("Sắp xếp lại vector 1 chiều thành ma trận 8x8.")
        
        matrix = dequantized_blocks[block_height_idx, block_width_idx, :, :]
        st.write("Matrix sau Inverse ZigZag:")

        fig = px.imshow(matrix, color_continuous_scale="gray")
        fig.update_layout(title=f"Inverse ZigZag Matrix - Block #{block_idx}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🔁 Dequantization")
        st.markdown("Khôi phục các hệ số gốc bằng cách nhân với ma trận lượng tử hóa.")

        matrix = idct_blocks[block_height_idx, block_width_idx, :, :]
        st.write("Matrix sau Dequantization:")

        fig = px.imshow(matrix, color_continuous_scale="gray")
        fig.update_layout(title=f"Dequantized Coefficients - Block #{block_idx}")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("📥 Inverse DCT")
        st.markdown("Biến đổi về miền không gian để khôi phục ảnh gốc.")

        matrix = blocks[block_height_idx, block_width_idx, :, :]
        st.write("Block tái tạo sau Inverse DCT:")

        fig = px.imshow(matrix, color_continuous_scale="gray")
        fig.update_layout(title=f"Reconstructed Block - Block #{block_idx}")
        st.plotly_chart(fig, use_container_width=True)
