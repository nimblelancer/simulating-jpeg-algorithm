import streamlit as st
import os
import importlib

# Import all the modules for each step
import_modules = [
    "1_upload_image",
    "2_dct_transform",
    "3_quantization",
    "4_zigzag",
    "5_huffman",
    "6_decode_image",
    "7_compare_result"
]

# Create temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

def main():
    st.set_page_config(
        page_title="JPEG Compression Simulator",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Step names for display
    step_names = [
        "1. Upload Image",
        "2. DCT Transform",
        "3. Quantization",
        "4. Zigzag Scanning",
        "5. Huffman Coding",
        "6. Image Decoding",
        "7. Compare Results"
    ]

    # Initialize current step if not in session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

    # Sidebar for step selection
    st.sidebar.title("🧭 JPEG Compression Steps")
    st.sidebar.write("Follow each step to visualize the JPEG compression process")

    # Chỉ cho phép chọn đến current_step
    available_steps = step_names[:st.session_state.current_step]
    selected_step_label = st.sidebar.radio("Step", available_steps)
    st.session_state.current_step = step_names.index(selected_step_label) + 1

    # Hiển thị tiến trình phía trên (progress bar dạng ngang)
    st.markdown("### 🔄 Step Progress")
    progress_cols = st.columns(len(step_names))
    for i, col in enumerate(progress_cols):
        step_num = i + 1
        if step_num < st.session_state.current_step:
            col.markdown(
                f"<div style='text-align: center; background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;'>{step_names[i]}</div>",
                unsafe_allow_html=True
            )
        elif step_num == st.session_state.current_step:
            col.markdown(
                f"<div style='text-align: center; background-color: #008CBA; color: white; padding: 5px; border-radius: 5px;'>{step_names[i]}</div>",
                unsafe_allow_html=True
            )
        else:
            col.markdown(
                f"<div style='text-align: center; background-color: #f1f1f1; color: #555; padding: 5px; border-radius: 5px;'>{step_names[i]}</div>",
                unsafe_allow_html=True
            )

    # Nút điều hướng
    st.write("")  # Add space
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.session_state.current_step > 1:
            if st.button("← Previous Step"):
                st.session_state.current_step -= 1
                st.rerun()

    with col3:
        if st.session_state.current_step < len(step_names):
            if st.button("Next Step →"):
                if check_step_requirements(st.session_state.current_step):
                    st.session_state.current_step += 1
                    st.rerun()
                else:
                    st.error("Please complete the current step before proceeding.")

    # Import và chạy module tương ứng
    module_name = import_modules[st.session_state.current_step - 1]
    try:
        module = importlib.import_module(module_name)
        module.app()
    except Exception as e:
        st.error(f"Error running step {st.session_state.current_step}: {str(e)}")
        st.exception(e)


def check_step_requirements(step):
    """Check if requirements for the current step are met before allowing to proceed"""
    if step == 1:  # Upload Image
        return 'image' in st.session_state and st.session_state.image is not None
    elif step == 2:  # DCT Transform
        return os.path.exists(os.path.join("temp", "original.jpg"))
    elif step == 3:  # Quantization
        return os.path.exists(os.path.join("temp", "dct_coefficients.npy"))
    elif step == 4:  # Zigzag
        return os.path.exists(os.path.join("temp", "quantized_coefficients.npy"))
    elif step == 5:  # Huffman
        return os.path.exists(os.path.join("temp", "zigzag_data.npy"))
    elif step == 6:  # Decode
        return os.path.exists(os.path.join("temp", "zigzag_data.npy"))
    elif step == 7:  # Compare
        return (
            os.path.exists(os.path.join("temp", "original.jpg")) and
            os.path.exists(os.path.join("temp", "decoded.jpg"))
        )
    return False


if __name__ == "__main__":
    main()
