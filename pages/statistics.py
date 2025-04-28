import streamlit as st
import pandas as pd
import plotly.express as px

def app():
    st.title("📈 JPEG Compression Analysis")
    st.write("Summary of PSNR and SSIM over different Quality Factors.")

    # Đọc file CSV thực tế
    try:
        data = pd.read_csv("jpeg_psnr_ssim_results.csv")
    except FileNotFoundError:
        st.error("Không tìm thấy file jpeg_psnr_ssim_results.csv. Vui lòng kiểm tra lại!")
        return


    # Hiển thị bảng dữ liệu
    st.subheader("Compression Metrics Table")
    st.dataframe(data)

    # Vẽ biểu đồ PSNR theo Quality Factor
    st.subheader("PSNR vs Quality Factor")
    fig_psnr = px.line(
        data,
        x="Quality Factor",
        y="PSNR",
        markers=True,
        title="PSNR (dB) vs Quality Factor",
        labels={"PSNR": "PSNR (dB)", "Quality Factor": "Quality Factor"},
    )
    st.plotly_chart(fig_psnr)

    # Vẽ biểu đồ SSIM theo Quality Factor
    st.subheader("SSIM vs Quality Factor")
    fig_ssim = px.line(
        data,
        x="Quality Factor",
        y="SSIM",
        markers=True,
        title="SSIM vs Quality Factor",
        labels={"SSIM": "SSIM", "Quality Factor": "Quality Factor"},
    )
    st.plotly_chart(fig_ssim)

    # Gợi ý thêm: Nếu muốn chọn range Quality Factor để zoom
    st.subheader("Custom Range Filter")
    qf_min, qf_max = st.slider(
        "Select Quality Factor Range:",
        min_value=int(data["Quality Factor"].min()),
        max_value=int(data["Quality Factor"].max()),
        value=(1, 100)
    )

    filtered_data = data[(data["Quality Factor"] >= qf_min) & (data["Quality Factor"] <= qf_max)]

    st.write(f"Showing results for Quality Factor from {qf_min} to {qf_max}")
    st.dataframe(filtered_data)

    fig_filtered = px.line(
        filtered_data,
        x="Quality Factor",
        y=["PSNR", "SSIM"],
        markers=True,
        title="PSNR and SSIM in Selected Range",
    )
    st.plotly_chart(fig_filtered)
