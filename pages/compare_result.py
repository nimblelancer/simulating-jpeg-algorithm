import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image

def app():
    st.title("📊 Compare Results")
    st.write("Compare the original and compressed images.")

    # Dữ liệu giả lập
    mock_original = (np.random.rand(256, 256) * 255).astype(np.uint8)
    mock_compressed = (mock_original * 0.95).astype(np.uint8)

    # Hiển thị ảnh song song
    col1, col2 = st.columns(2)
    with col1:
        st.image(mock_original, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mock_compressed, caption="Compressed Image", use_container_width=True)
    # GHI CHÚ: Tích hợp với phần core
    # Lấy ảnh gốc từ st.session_state['image'] và ảnh nén từ st.session_state['compressed_image']

    # Slider so sánh
    st.subheader("Before/After Slider")
    # GHI CHÚ: Tích hợp slider so sánh bằng cách sử dụng thư viện như streamlit-image-comparison
    # Hoặc tạo custom slider với st.slider và overlay hai ảnh

    # Thông số so sánh
    st.subheader("Compression Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Size", "500 KB")
    with col2:
        st.metric("Compressed Size", "150 KB")
    with col3:
        st.metric("Compression Ratio", "3.33:1")
    with col4:
        st.metric("PSNR", "35.2 dB")
    # GHI CHÚ: Tích hợp với phần core
    # Tính toán các chỉ số (size, ratio, PSNR, SSIM) từ JPEGProcessor.get_metrics()

    # Biểu đồ
    st.subheader("Visual Analysis")
    fig = px.bar(x=["Original", "Compressed"], y=[500, 150], title="File Size Comparison")
    st.plotly_chart(fig)

    fig = px.histogram(
        x=mock_original.flatten() - mock_compressed.flatten(),
        nbins=50,
        title="Error Histogram"
    )
    st.plotly_chart(fig)
    # GHI CHÚ: Tạo histogram từ sự khác biệt pixel giữa ảnh gốc và ảnh nén
