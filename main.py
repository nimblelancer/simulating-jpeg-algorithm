import streamlit as st
from navigation import render_selected_page
from utils.image_io import clear_processing_folder
import atexit

st.set_page_config(page_title="JPEG Visualizer", layout="wide")

# Ẩn sidebar mặc định của Streamlit
st.markdown("""
    <style>
        div[data-testid="stSidebarNav"]{
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Đăng ký hàm cleanup
def on_exit():
    clear_processing_folder()

atexit.register(on_exit)

# Gọi menu và render trang phù hợp
render_selected_page()