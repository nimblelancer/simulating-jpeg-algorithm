import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
from streamlit_image_comparison import image_comparison
from utils.metrics import analyze_compression


def app():
    st.title("📊 Compare Results")
    st.write("So sánh ảnh gốc và ảnh sau nén từ thuật toán JPEG.")

    if 'original_image_path' not in st.session_state or 'compressed_image_path' not in st.session_state:
        st.warning("⚠️ Vui lòng tải ảnh và thực hiện nén trước khi so sánh.")
        return

    original_path = st.session_state['original_image_path']
    compressed_path = st.session_state['compressed_image_path']
    decompressed_path = st.session_state['decompressed_image_path']

    # Phân tích nén
    result = analyze_compression(original_path, compressed_path, decompressed_path)

    # Hiển thị ảnh song song
    st.subheader("🖼️ Image Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_path, caption="Original Image", use_container_width=True)
    with col2:
        st.image(decompressed_path, caption="Decompressed Image", use_container_width=True)

    # Slider so sánh
    st.subheader("🧮 Before/After Slider")
    image_comparison(
        img1=original_path,
        img2=decompressed_path,
        label1="Original",
        label2="Decompressed",
        width=700
    )

    # Thông số so sánh
    st.subheader("📊 Compression Metrics")
    cols = st.columns(6)

    cols[0].metric("Original Size", f"{result['original']['size_kb']:.2f} KB")
    cols[1].metric("Compressed Size", f"{result['compressed']['size_kb']:.2f} KB")
    cols[2].metric("Decompressed Size", f"{result['decompressed']['size_kb']:.2f} KB")
    cols[3].metric("Compression Ratio", f"{result['compression_ratio']['compression_ratio']:.2f}:1")
    cols[4].metric("PSNR", f"{result['psnr']:.2f} dB")
    cols[5].metric("SSIM", f"{result['ssim']:.4f}")

    # Biểu đồ trực quan
    st.subheader("📈 Visual Analysis")
    fig1 = px.bar(
        x=["Original", "Compressed", "Decompressed"],
        y=[
            result["original"]["size_kb"], 
            result["compressed"]["size_kb"],
            result["decompressed"]["size_kb"]
        ],  # Truy cập đúng giá trị size_kb
        labels={"x": "Image Type", "y": "File Size (KB)"},
        title="📦 File Size Comparison"
    )
    st.plotly_chart(fig1)

    # Histogram pixel error
    original = np.array(Image.open(original_path).convert("L"))
    decompressed = np.array(Image.open(decompressed_path).convert("L"))
    error = original.astype(np.int16) - decompressed.astype(np.int16)

    fig2 = px.histogram(
        x=error.flatten(),
        nbins=50,
        labels={"x": "Pixel Error"},
        title="📉 Histogram of Pixel Differences (Original vs Decompressed)"
    )
    st.plotly_chart(fig2)
