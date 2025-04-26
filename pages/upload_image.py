import streamlit as st
import numpy as np
from PIL import Image
import io
import plotly.express as px
from jpeg_processor import JPEGProcessor
from utils.image_io import load_uploaded_image, save_image, load_image, ensure_dir

def app():
    st.title("📸 Upload Your Image")
    st.write("Upload an image to start the JPEG compression process.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Đọc và hiển thị ảnh gốc
        image = load_uploaded_image(uploaded_file)
        is_color_image = image.ndim == 3 and image.shape[2] == 3
        st.image(image, caption="Original Image", use_container_width=True)
        st.session_state['original_image_path'] = f"assets/images/processing/original.png"
        
        # Hiển thị thông tin ảnh
        st.subheader("Image Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            height, width = image.shape[:2]
            st.metric("Resolution", f"{width} x {height}")
        with col2:
            # Kích thước tệp (giữ nguyên)
            st.metric("Size", f"{round(len(uploaded_file.getvalue()) / 1024, 2)} KB")
        with col3:
            # Lấy định dạng từ tên tệp
            file_extension = uploaded_file.name.split('.')[-1].upper()
            st.metric("Format", file_extension)

        # Thiết lập thông số nén
        st.subheader("Compression Settings")
        quality_factor = st.slider("Quality Factor", 1, 100, 80)

        jpeg = JPEGProcessor(quality_factor)
        # Nút Compress và Decompress
        col_compress, col_decompress = st.columns(2)
        with col_compress:
            if st.button("Compress"):
                st.session_state['original_shape'] = image.shape
                st.session_state['quality_factor'] = quality_factor
                if is_color_image:
                    image = Image.open(uploaded_file)
                    ensure_dir()
                    image.save("assets/images/processing/decompressed_image.jpg", format="JPEG", quality=quality_factor, optimize=True)
                    st.image(load_image("decompressed_image.jpg"), caption="Decompressed Image", use_container_width=True)
                    st.success("JPEG Pipeline completed!")
                else:
                    result = jpeg.encode_pipeline(image)
                    st.session_state['encoded_dc_original'] = result['encoded_dc_original']
                    st.session_state['compressed_image_path'] = f"assets/images/processing/compressed_image.jpg"
                    st.session_state['encoded_data'] = result['encoded_data']
                    st.session_state['dc_codes'] = result['dc_codes']
                    st.session_state['ac_codes'] = result['ac_codes']
                    st.session_state['padded_shape'] = result['padded_shape']
                    st.session_state['total_bits'] = result['total_bits']
                    st.success("Compression completed! Navigate to other pages to explore.")
        with col_decompress:
            if st.button("Decompress"):
                encoded_data = st.session_state.get('encoded_data')
                if encoded_data is None:
                    st.warning("Please compress an image first!")
                else:
                    decompressed_image = jpeg.decode_pipeline(encoded_data, st.session_state['dc_codes'], st.session_state['ac_codes'], st.session_state['padded_shape'], st.session_state['total_bits'], st.session_state.get('original_shape'))
                    st.image(decompressed_image, caption="Decompressed Image", use_container_width=True)
                    st.session_state['decompressed_image_path'] = f"assets/images/processing/decompressed_image.jpg"
                    st.success("Decompression completed! Navigate to other pages to explore.")
