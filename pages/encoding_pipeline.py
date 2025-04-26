import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from utils.image_io import load_npy, build_huffman_result
import matplotlib.pyplot as plt

def app():
    st.title("🛠 JPEG Encoding Pipeline")
    st.write("Explore each step of the JPEG encoding process.")

    blocks = load_npy("encode_step_blocks.npy")
    dct_blocks = load_npy("encode_step_dct.npy")       # shape (N, 8, 8)
    quantized_blocks = load_npy("encode_step_quantized.npy")
    zigzag_vectors = load_npy("encode_step_rle.npy", allow_pickle=True)
    encoded_dc_original = st.session_state['encoded_dc_original']
    dc_codes = st.session_state['dc_codes']
    ac_codes = st.session_state['ac_codes']
    huffman_result = build_huffman_result(dc_codes, ac_codes)

    # Block slider
    num_blocks = blocks.shape[0]
    block_height_idx = st.slider("Select Block Height Index", 0, blocks.shape[0]-1, 0)
    block_width_idx = st.slider("Select Block Width Index", 0, blocks.shape[1]-1, 0)
    block_idx = block_height_idx * blocks.shape[1] + block_width_idx

    # Tabs layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧱 Block Splitting", 
        "📐 DCT Transform", 
        "🎚️ Quantization", 
        "🧩 ZigZag Scan", 
        "🧮 Huffman Encoding"
    ])

    with tab1:
        st.subheader("🧱 Block Splitting (8x8)")
        st.markdown("Ảnh được chia thành các block 8x8 để xử lý từng phần nhỏ.")
        fig = px.imshow(blocks[block_height_idx, block_width_idx, :, :], color_continuous_scale="gray")
        fig.update_layout(title=f"Block #{block_idx} (Pixel Values)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📐 DCT Transform")
        st.markdown("Biến đổi mỗi block 8x8 sang miền tần số bằng Discrete Cosine Transform (DCT).")
        fig = px.imshow(dct_blocks[block_height_idx, block_width_idx, :, :], color_continuous_scale="gray")
        fig.update_layout(title=f"DCT Coefficients of Block #{block_idx}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🎚️ Quantization")
        st.markdown("Lượng tử hóa DCT coefficients để loại bỏ tần số cao (giảm chi tiết ít quan trọng).")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Before Quantization (DCT)")
            fig1 = px.imshow(dct_blocks[block_height_idx, block_width_idx, :, :], color_continuous_scale="gray")
            fig1.update_layout(title=f"DCT Block #{block_idx}")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.write("After Quantization")
            fig2 = px.imshow(quantized_blocks[block_height_idx, block_width_idx, :, :], color_continuous_scale="gray")
            fig2.update_layout(title=f"Quantized Block #{block_idx}")
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("🧩 ZigZag Scan")
        st.markdown("Chuyển ma trận 8x8 thành vector theo thứ tự ZigZag để gom các số 0 lại gần nhau.")
        st.write("Quantized block (input for ZigZag):")
        st.code(quantized_blocks[block_height_idx, block_width_idx, :, :])
        # Vẽ ma trận với đường đi ZigZag
        zigzag_indices = np.array([
            0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
        ])
        
        # Chuyển chỉ số tuyến tính thành tọa độ (i, j)
        zigzag_coords = [(idx // 8, idx % 8) for idx in zigzag_indices]
        
        # Lấy ma trận quantized
        matrix = quantized_blocks[block_height_idx, block_width_idx, :, :]
        
        # Vẽ ma trận và đường đi ZigZag
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(matrix, cmap='coolwarm')  # Heatmap của ma trận
        
        # Thêm giá trị vào từng ô
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f'{matrix[i, j]}', ha='center', va='center', color='black')
        
        # Vẽ đường đi ZigZag
        x_coords = [coord[1] + 0.5 for coord in zigzag_coords]  # Tọa độ x (cột)
        y_coords = [coord[0] + 0.5 for coord in zigzag_coords]  # Tọa độ y (hàng)
        ax.plot(x_coords, y_coords, 'k--', linewidth=1, alpha=0.7)  # Đường nét đứt màu đen
        
        # Ẩn các trục để hình ảnh gọn gàng
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.set_xticklabels(np.arange(8))
        ax.set_yticklabels(np.arange(8))
        ax.set_title("ZigZag Scan Path")
        
        # Hiển thị hình ảnh trong Streamlit
        st.pyplot(fig)
        st.markdown("Vector chứa: [DC coefficient, danh sách các cặp (run, value) cho AC coefficients].")
        st.markdown(f"- DC coefficient: Giá trị tần số thấp (sáng/tối của block).")
        st.markdown(f"- (run, value): Số lượng giá trị 0 trước một giá trị khác 0, và giá trị đó.")
        st.markdown(f"- (0, 0): Dấu hiệu kết thúc block (EOB).")
        st.write(f"ZigZag vector for block #{block_idx}:")
        zigzag_vector = zigzag_vectors[block_idx].tolist()  # Chuyển DC và danh sách AC sang list
        dc_original  = encoded_dc_original[block_idx]  # DC coefficient
        zigzag_vector = [int(dc_original), [(int(run), int(value)) for run, value in zigzag_vector[1]]]
        st.code(f"{zigzag_vector}")

    with tab5:
        st.subheader("🧮 Huffman Encoding")
        st.markdown("Nén vector bằng mã hóa Huffman dựa trên tần suất xuất hiện.")

        # Huffman codes là dict {'symbol': 'code'}
        st.write("**Sample Huffman Codes:**")
        sample_codes = dict(list(huffman_result.items())[:10])  # Show 10 đầu tiên
        st.json(sample_codes)

        st.markdown("Bạn có thể tải toàn bộ mã Huffman để phân tích thêm.")
        st.download_button("📥 Tải toàn bộ Huffman codes", data=str(huffman_result), file_name="huffman_codes.txt")