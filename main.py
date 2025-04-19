import streamlit as st
from navigation import render_selected_page

st.set_page_config(page_title="JPEG Visualizer", layout="wide")

# Ẩn sidebar mặc định của Streamlit
st.markdown("""
    <style>
        div[data-testid="stSidebarNav"]{
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Gọi menu và render trang phù hợp
render_selected_page()