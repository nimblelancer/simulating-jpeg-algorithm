import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image

def app():
    st.title("🔄 JPEG Decoding Pipeline")
    st.write("Explore each step of the JPEG decoding process.")

    # Tabs cho từng bước
    tab1, tab2, tab3, tab4 = st.tabs([
        "Huffman Decoding",
        "Inverse ZigZag",
        "Dequantization",
        "Inverse DCT"
    ])

    # Dữ liệu giả lập
    mock_image = np.random.rand(256, 256) * 255  # Ảnh giả lập
    mock_dequantized = np.random.rand(8, 8) * 100  # Ma trận sau dequantization
    mock_idct = np.random.rand(256, 256) * 255  # Ảnh sau IDCT

    with tab1:
        st.subheader("Huffman Decoding")
        st.write("Decode the compressed bitstream using Huffman codes.")
        
        st.image(mock_image, caption="Partially Decoded Image", clamp=True)
        # GHI CHÚ: Tích hợp với phần core
        # Hiển thị ảnh sau Huffman decoding từ JPEGProcessor.huffman_decode()

    with tab2:
        st.subheader("Inverse ZigZag")
        st.write("Rearrange 1D vector back into 2D matrix.")
        
        st.image(mock_image, caption="Image after Inverse ZigZag", clamp=True)
        # GHI CHÚ: Tích hợp với phần core
        # Hiển thị ma trận hoặc ảnh từ JPEGProcessor.inverse_zigzag()

    with tab3:
        st.subheader("Dequantization")
        st.write("Scale coefficients using quantization table.")
        
        fig = px.imshow(mock_dequantized, color_continuous_scale="gray", title="Dequantized Coefficients")
        st.plotly_chart(fig)
        # GHI CHÚ: Tích hợp với phần core
        # Hiển thị ma trận sau dequantization từ JPEGProcessor.dequantize()

    with tab4:
        st.subheader("Inverse DCT")
        st.write("Transform back to spatial domain using Inverse DCT.")
        
        st.image(mock_idct, caption="Reconstructed Image", clamp=True)
        # GHI CHÚ: Tích hợp với phần core
        # Hiển thị ảnh sau IDCT từ JPEGProcessor.inverse_dct()
        
        st.checkbox("Overlay Original Image", key="overlay")
        # GHI CHÚ: Tích hợp overlay để so sánh ảnh gốc và ảnh tái tạo
