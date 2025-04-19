from streamlit_option_menu import option_menu
import pages.upload_image as upload_image
import pages.encoding_pipeline as encoding_pipeline
import pages.decoding_pipeline as decoding_pipeline
import pages.compare_result as compare_result
import pages.statistics as statistics
import pages.about as about
import streamlit as st

def render_selected_page():
    with st.sidebar:
        selected = option_menu(
            "JPEG Visualization",
            ["Upload Image", "Encoding Pipeline", "Decoding Pipeline", "Compare Result", "Statistics", "About"],
            icons=["upload", "gear", "gear", "image", "bar-chart", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Upload Image":
        upload_image.app()

    elif selected == "Encoding Pipeline":
        encoding_pipeline.app()

    elif selected == "Decoding Pipeline":
        decoding_pipeline.app()

    elif selected == "Compare Result":
        compare_result.app()

    elif selected == "Statistics":
        statistics.app()

    elif selected == "About":
        about.app()
