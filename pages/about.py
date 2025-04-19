import streamlit as st

def app():
    st.title("ℹ️ About JPEG Compression Visualization")
    st.write("Learn about the project and the JPEG compression pipeline.")

    st.markdown("""
    ## Project Overview
    This application visualizes the JPEG compression process, breaking down each step of encoding and decoding to help users understand how images are compressed and reconstructed.

    ### JPEG Pipeline
    1. **Encoding**:
       - DCT Transform: Convert image to frequency domain.
       - Quantization: Reduce coefficient precision.
       - ZigZag Scan: Rearrange coefficients.
       - Huffman Encoding: Compress data losslessly.
    2. **Decoding**:
       - Huffman Decoding: Reconstruct coefficients.
       - Inverse ZigZag: Rebuild 2D matrix.
       - Dequantization: Scale coefficients.
       - Inverse DCT: Reconstruct image.

    ## Team
    - **Your Name**: Developer and algorithm designer.
    - **Contact**: [Your Email or GitHub]

    ## Resources
    - [JPEG Algorithm Documentation](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
    - [GitHub Repository](https://github.com/nimblelancer/simulating-jpeg-algorithm)
    """)
