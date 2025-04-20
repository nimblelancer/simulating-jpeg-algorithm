import streamlit as st
import numpy as np
from PIL import Image
import io
import plotly.express as px
from jpeg_processor import JPEGProcessor
from utils.image_io import read_image, save_image

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
        # color_mode = st.selectbox("Color Mode", ["RGB", "Grayscale"])

        jpeg = JPEGProcessor(quality_factor)
        # Nút Compress và Decompress
        col_compress, col_decompress = st.columns(2)
        with col_compress:
            if st.button("Compress"):
                image_array = read_image(uploaded_file)
                st.session_state['original_shape'] = image_array.shape
                save_image(image_array, "original.jpg")

                st.session_state['image'] = image
                st.session_state['quality_factor'] = quality_factor
                # st.session_state['color_mode'] = color_mode
                
                result = jpeg.encode_pipeline(image_array)
                st.session_state['compressed_image'] = result['encoded_data']
                st.session_state['dc_codes'] = result['dc_codes']
                st.session_state['ac_codes'] = result['ac_codes']
                st.session_state['padded_shape'] = result['padded_shape']
                st.session_state['total_bits'] = result['total_bits']
                st.success("Compression completed! Navigate to other pages to explore.")
        with col_decompress:
            if st.button("Decompress"):
                encoded_data = st.session_state.get('compressed_image')
                if encoded_data is None:
                    st.warning("Please compress an image first!")
                else:
                    print("Original shape:", st.session_state.get('original_shape'))
                    decompressed_image = jpeg.decode_pipeline(encoded_data, st.session_state['dc_codes'], st.session_state['ac_codes'], st.session_state['padded_shape'], st.session_state['total_bits'], st.session_state.get('original_shape'))
                    st.session_state['decompressed_image'] = decompressed_image
                    st.image(decompressed_image, caption="Decompressed Image", use_container_width=True)
                    save_image(decompressed_image, "decompressed_image.jpg")
                st.success("Decompression completed! Navigate to other pages to explore.")
