import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

def get_quantization_matrix(quality=50):
    # Standard JPEG quantization matrix for luminance
    # Higher values = more compression and loss
    base_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Adjust matrix based on quality factor
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    q_matrix = np.floor((base_matrix * scale + 50) / 100)
    q_matrix = np.clip(q_matrix, 1, 255).astype(np.int32)
    
    return q_matrix

def apply_quantization(dct_block, q_matrix):
    # Element-wise division and rounding
    return np.round(dct_block / q_matrix)

def visualize_quantization(dct_block, quantized_block, q_matrix):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display DCT coefficients
    dct_log = np.log(abs(dct_block) + 1)
    im1 = ax1.imshow(dct_log, cmap='viridis')
    ax1.set_title('DCT Coefficients')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Display quantization matrix
    im2 = ax2.imshow(q_matrix, cmap='hot')
    ax2.set_title('Quantization Matrix')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Display quantized DCT coefficients
    im3 = ax3.imshow(quantized_block, cmap='viridis')
    ax3.set_title('Quantized DCT')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def app():
    st.title("Step 3: Quantization")
    
    # Check if DCT coefficients exist
    dct_file = os.path.join("temp", "dct_coefficients.npy")
    if not os.path.exists(dct_file):
        st.warning("Please complete the DCT transformation step first.")
        return
    
    # Load the DCT coefficients
    dct_image = np.load(dct_file)
    
    st.write("""
    Quantization is a lossy step in JPEG compression where DCT coefficients are divided by values 
    from a quantization matrix. Higher frequency components (bottom-right of each 8x8 block) are 
    quantized more heavily, as they are less perceptible to the human eye.
    """)
    
    # Allow user to select quality factor
    quality = st.slider("Compression Quality (1-100)", 1, 100, 50, 
                       help="Lower values mean higher compression and lower quality")
    
    # Get quantization matrix based on quality
    q_matrix = get_quantization_matrix(quality)
    
    # Display the quantization matrix
    st.subheader("Quantization Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(q_matrix, cmap='hot')
    plt.colorbar(im)
    ax.set_title(f"Quantization Matrix (Quality: {quality})")
    st.pyplot(fig)
    
    # Apply quantization to the entire image
    height, width = dct_image.shape
    quantized_image = np.zeros_like(dct_image)
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Get the current 8x8 block
            i_end = min(i+8, height)
            j_end = min(j+8, width)
            
            dct_block = dct_image[i:i_end, j:j_end]
            
            # If the block is smaller than 8x8 (at edges), pad it
            if dct_block.shape[0] < 8 or dct_block.shape[1] < 8:
                padded_block = np.zeros((8, 8), dtype=np.float32)
                padded_block[:dct_block.shape[0], :dct_block.shape[1]] = dct_block
                dct_block = padded_block
                
                # Apply quantization
                quantized_block = apply_quantization(dct_block, q_matrix)
                
                # Store back (only the valid part)
                quantized_image[i:i_end, j:j_end] = quantized_block[:i_end-i, :j_end-j]
            else:
                # Apply quantization
                quantized_block = apply_quantization(dct_block, q_matrix)
                
                # Store the quantized coefficients
                quantized_image[i:i_end, j:j_end] = quantized_block
    
    # Save the quantized coefficients
    np.save(os.path.join("temp", "quantized_coefficients.npy"), quantized_image)
    
    # Visualize the effect on a sample block
    st.subheader("Visualization of Quantization on a Sample Block")
    
    # Create sliders for block selection
    max_i = height - 8
    max_j = width - 8
    i_block = st.slider("Vertical position", 0, max_i, max_i // 2, 8)
    j_block = st.slider("Horizontal position", 0, max_j, max_j // 2, 8)
    
    # Get the selected block
    dct_block = dct_image[i_block:i_block+8, j_block:j_block+8]
    quantized_block = quantized_image[i_block:i_block+8, j_block:j_block+8]
    
    # Visualize
    fig = visualize_quantization(dct_block, quantized_block, q_matrix)
    st.pyplot(fig)
    
    # Calculate and display zeros
    zero_count = np.sum(quantized_image == 0)
    total_coefficients = quantized_image.size
    zero_percentage = (zero_count / total_coefficients) * 100
    
    st.subheader("Compression Effect")
    st.write(f"Percentage of zero coefficients: {zero_percentage:.2f}%")
    st.write(f"Total zero coefficients: {zero_count} out of {total_coefficients}")
    
    # Visualize before and after
    st.subheader("DCT Coefficients Before and After Quantization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dct_log = np.log(abs(dct_image) + 1)
        dct_normalized = (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))
        st.image(dct_normalized, caption="Before Quantization", use_column_width=True)
    
    with col2:
        quant_log = np.log(abs(quantized_image) + 1)
        quant_normalized = (quant_log - np.min(quant_log)) / (np.max(quant_log) - np.min(quant_log))
        st.image(quant_normalized, caption="After Quantization", use_column_width=True)
    
    st.success(f"Quantization completed with quality factor {quality}! Proceed to the next step.")

if __name__ == "__main__":
    app()