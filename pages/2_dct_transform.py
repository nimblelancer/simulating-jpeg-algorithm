import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

def apply_dct(image_block):
    # Convert to float
    f = np.float32(image_block)
    # Apply DCT
    return cv2.dct(f)

def visualize_dct(original_block, dct_block):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original block
    ax1.imshow(original_block, cmap='gray')
    ax1.set_title('Original Block')
    ax1.axis('off')
    
    # Display DCT coefficients (log scale for better visualization)
    dct_log = np.log(abs(dct_block) + 1)
    im = ax2.imshow(dct_log, cmap='viridis')
    ax2.set_title('DCT Coefficients (Log Scale)')
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    return fig

def app():
    st.title("Step 2: Discrete Cosine Transform (DCT)")
    
    # Check if image exists in session state
    if 'image' not in st.session_state or st.session_state.image is None:
        st.warning("Please upload an image in the previous step first.")
        return
    
    # Get the image
    image = st.session_state.image
    
    # Convert to grayscale for simplicity
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    st.write("The JPEG compression divides the image into 8x8 blocks and applies DCT to each block.")
    
    # Apply DCT transformation
    height, width = gray_image.shape
    dct_image = np.zeros_like(gray_image, dtype=np.float32)
    
    # Process 8x8 blocks
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Get the current 8x8 block
            block = gray_image[i:i+8, j:j+8]
            
            # If the block is smaller than 8x8 (at the edges), pad it
            if block.shape[0] < 8 or block.shape[1] < 8:
                padded_block = np.zeros((8, 8), dtype=np.uint8)
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block
            
            # Apply DCT
            dct_block = apply_dct(block)
            
            # Store the DCT coefficients
            dct_image[i:i+8, j:j+8] = dct_block
    
    # Save the DCT image
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    np.save(os.path.join(temp_dir, "dct_coefficients.npy"), dct_image)
    
    # Visualize original vs DCT for a sample block
    st.subheader("Select a block to visualize DCT transformation")
    
    # Create sliders for block selection
    max_i = height - 8
    max_j = width - 8
    i_block = st.slider("Vertical position", 0, max_i, max_i // 2, 8)
    j_block = st.slider("Horizontal position", 0, max_j, max_j // 2, 8)
    
    # Get the selected block
    selected_block = gray_image[i_block:i_block+8, j_block:j_block+8]
    dct_block = apply_dct(selected_block)
    
    # Visualize
    fig = visualize_dct(selected_block, dct_block)
    st.pyplot(fig)
    
    # Display the overall effect
    st.subheader("DCT Coefficients of the Entire Image")
    # For visualization purposes, we'll use log scale
    dct_log = np.log(abs(dct_image) + 1)
    
    # Normalize for display
    dct_normalized = (dct_log - np.min(dct_log)) / (np.max(dct_log) - np.min(dct_log))
    
    # Display
    st.image(dct_normalized, caption="DCT Coefficients (Log Scale)", use_column_width=True)
    
    st.success("DCT transformation completed! You can proceed to the next step.")

if __name__ == "__main__":
    app()