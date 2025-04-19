import streamlit as st
import numpy as np
from PIL import Image
import io
import plotly.express as px
from jpeg_processor import JPEGProcessor

def app():
    st.title("📸 Upload Your Image")
    st.write("Upload an image to start the JPEG compression process.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Đọc và hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        # Hiển thị thông tin ảnh
        st.subheader("Image Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Resolution", f"{image.size[0]} x {image.size[1]}")
        with col2:
            st.metric("Size", f"{round(len(uploaded_file.getvalue()) / 1024, 2)} KB")
        with col3:
            st.metric("Format", image.format)

        # Thiết lập thông số nén
        st.subheader("Compression Settings")
        quality_factor = st.slider("Quality Factor", 1, 100, 75)
        color_mode = st.selectbox("Color Mode", ["RGB", "Grayscale"])

        # Nút Compress và Decompress
        col_compress, col_decompress = st.columns(2)
        with col_compress:
            if st.button("Compress"):
                st.session_state['image'] = image
                st.session_state['quality_factor'] = quality_factor
                st.session_state['color_mode'] = color_mode

                jpeg = JPEGProcessor
                # GHI CHÚ: Tích hợp với phần core
                # Gọi JPEGProcessor.encode(image, quality_factor, color_mode)
                # Lưu kết quả vào st.session_state['compressed_image']
                st.success("Compression completed! Navigate to other pages to explore.")
        with col_decompress:
            if st.button("Decompress"):
                # GHI CHÚ: Tích hợp với phần core
                # Gọi JPEGProcessor.decode(compressed_image)
                # Lưu kết quả vào st.session_state['decompressed_image']
                st.success("Decompression completed! Navigate to other pages to explore.")
