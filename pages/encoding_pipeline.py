import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

def app():
    st.title("🛠 JPEG Encoding Pipeline")
    st.write("Explore each step of the JPEG encoding process.")

    # Tabs cho từng bước
    tab1, tab2, tab3, tab4 = st.tabs(["DCT Transform", "Quantization", "ZigZag Scan", "Huffman Encoding"])

    # Dữ liệu giả lập (nếu chưa có dữ liệu thật tích hợp từ JPEGProcessor)
    mock_image = np.random.rand(256, 256) * 255  # Ảnh giả lập
    mock_dct = np.random.rand(8, 8) * 100  # Ma trận DCT giả lập
    mock_quantized = np.round(mock_dct / 10)  # Ma trận sau quantization
    mock_zigzag = np.random.randint(0, 100, size=64)  # Vector zigzag giả lập
    mock_huffman_codes = {"symbol1": "00", "symbol2": "01", "symbol3": "10"}  # Mã Huffman giả lập

    with tab1:
        st.subheader("DCT Transform")
        st.write("Transform the image into frequency domain using Discrete Cosine Transform.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(mock_image, caption="Image Blocks (8x8)", clamp=True)
            # GHI CHÚ: Tích hợp với phần core
            # Hiển thị ảnh sau khi chia block 8x8 từ JPEGProcessor.dct_transform()
        with col2:
            fig = px.imshow(mock_dct, color_continuous_scale="gray", title="DCT Coefficients")
            st.plotly_chart(fig)
            # GHI CHÚ: Tích hợp với phần core
            # Lấy ma trận DCT từ JPEGProcessor.dct_transform() và hiển thị heatmap

        with st.expander("View DCT Matrix for a Block"):
            st.write("DCT Matrix Values:")
            st.write(mock_dct)
            # GHI CHÚ: Cho phép người dùng chọn block và hiển thị ma trận DCT tương ứng

    with tab2:
        st.subheader("Quantization")
        st.write("Reduce precision of DCT coefficients using a quantization table.")

        quality_factor = st.slider("Quality Factor", 1, 100, 75, key="quant")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(mock_dct, color_continuous_scale="gray", title="Before Quantization")
            st.plotly_chart(fig)
        with col2:
            fig = px.imshow(mock_quantized, color_continuous_scale="gray", title="After Quantization")
            st.plotly_chart(fig)
            # GHI CHÚ: Tích hợp với phần core
            # Lấy ma trận trước/sau quantization từ JPEGProcessor.quantize()

        with st.expander("View Quantization Matrix"):
            st.write("Quantization Matrix Values:")
            st.write(mock_quantized)
            # GHI CHÚ: Hiển thị ma trận Q từ JPEGProcessor.get_quantization_table()

    with tab3:
        st.subheader("ZigZag Scan")
        st.write("Rearrange coefficients into a 1D vector using ZigZag pattern.")

        fig = go.Figure(data=go.Scatter(y=mock_zigzag, mode="lines+markers"))
        fig.update_layout(title="ZigZag Scan Result", xaxis_title="Index", yaxis_title="Value")
        st.plotly_chart(fig)
        # GHI CHÚ: Tích hợp với phần core
        # Lấy vector zigzag từ JPEGProcessor.zigzag_scan() và hiển thị

    with tab4:
        st.subheader("Huffman Encoding")
        st.write("Compress the data using Huffman coding.")

        st.write("Huffman Codes:")
        st.write(mock_huffman_codes)
        # GHI CHÚ: Tích hợp với phần core
        # Lấy Huffman tree và mã từ JPEGProcessor.huffman_encode()
        # Sử dụng graphviz hoặc pyvis để vẽ cây Huffman

        st.write("Bit Length Comparison:")
        fig = px.bar(x=["Before", "After"], y=[1000, 500], title="Bit Length Before vs After")
        st.plotly_chart(fig)
        # GHI CHÚ: Tính toán độ dài bit từ JPEGProcessor.huffman_encode()
