import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

def inverse_quantization(quantized_block, q_matrix):
    # Element-wise multiplication
    return quantized_block * q_matrix

def inverse_zigzag_scan(zigzag, rows=8, cols=8):
    """Convert a 1D zigzag array back to a 2D block"""
    block = np.zeros((rows, cols))
    
    # Start from top-left corner (0, 0)
    i, j = 0, 0
    
    # Direction initially set to "up-right"
    going_up = True
    
    for idx in range(len(zigzag)):
        block[i, j] = zigzag[idx]
        
        if going_up:
            # Moving up-right
            if j == cols - 1:  # Reached right boundary
                i += 1
                going_up = False
            elif i == 0:  # Reached top boundary
                j += 1
                going_up = False
            else:  # Continue moving up-right
                i -= 1
                j += 1
        else:
            # Moving down-left
            if i == rows - 1:  # Reached bottom boundary
                j += 1
                going_up = True
            elif j == 0:  # Reached left boundary
                i += 1
                going_up = True
            else:  # Continue moving down-left
                i += 1
                j -= 1
    
    return block

def get_quantization_matrix(quality=50):
    # Standard JPEG quantization matrix for luminance
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

def app():
    st.title("Step 6: Image Decoding")
    
    # Check if previous steps were completed
    if not os.path.exists(os.path.join("temp", "zigzag_data.npy")):
        st.warning("Please complete the previous steps first.")
        return
    
    # Check if original image exists
    if not os.path.exists(os.path.join("temp", "original.jpg")):
        st.warning("Original image not found. Please go back to step 1.")
        return
    
    # Load original image for dimensions
    original_img = cv2.imread(os.path.join("temp", "original.jpg"))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Load zigzag data
    zigzag_data = np.load(os.path.join("temp", "zigzag_data.npy"))
    
    st.write("""
    Now we'll decode the compressed image by reversing each step of the compression process:
    1. Convert zigzag 1D array back to 2D blocks
    2. Apply inverse quantization
    3. Apply inverse DCT
    4. Combine all blocks to reconstruct the image
    """)
    
    # Get quality factor for quantization
    quality = st.slider("Compression Quality (Used for Quantization)", 1, 100, 50)
    q_matrix = get_quantization_matrix(quality)
    
    # Get dimensions of the original image
    height, width = original_img.shape[0], original_img.shape[1]
    
    # For grayscale processing
    if len(original_img.shape) == 3:
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_original = original_img
    
    # Create empty image for reconstruction
    reconstructed = np.zeros(gray_original.shape, dtype=np.uint8)
    
    # Process each block
    block_idx = 0
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if block_idx >= len(zigzag_data):
                break
                
            # Get current zigzag data and convert back to 2D block
            zigzag = zigzag_data[block_idx]
            quantized_block = inverse_zigzag_scan(zigzag)
            
            # Apply inverse quantization
            dequantized_block = inverse_quantization(quantized_block, q_matrix)
            
            # Apply inverse DCT
            idct_block = cv2.idct(dequantized_block.astype(np.float32))
            
            # Clip values to valid pixel range
            idct_block = np.clip(idct_block, 0, 255).astype(np.uint8)
            
            # Calculate valid dimensions (for edge blocks)
            i_end = min(i + 8, height)
            j_end = min(j + 8, width)
            block_height = i_end - i
            block_width = j_end - j
            
            # Place the reconstructed block into the image
            reconstructed[i:i_end, j:j_end] = idct_block[:block_height, :block_width]
            
            block_idx += 1
    
    # Save the reconstructed image
    cv2.imwrite(os.path.join("temp", "reconstructed.jpg"), reconstructed)
    
    # Display original and reconstructed images
    st.subheader("Original vs. Reconstructed Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(gray_original, caption="Original Image (Grayscale)", use_column_width=True)
    
    with col2:
        st.image(reconstructed, caption=f"Reconstructed Image (Quality: {quality})", use_column_width=True)
    
    # Visualize the decoding process for a sample block
    st.subheader("Decoding Process Visualization for a Sample Block")
    
    # Create sliders for block selection
    max_i = (height - 1) // 8
    max_j = (width - 1) // 8
    i_block = st.slider("Block row", 0, max_i, max_i // 2)
    j_block = st.slider("Block column", 0, max_j, max_j // 2)
    
    # Calculate block index
    block_idx = i_block * ((width + 7) // 8) + j_block
    
    if block_idx < len(zigzag_data):
        # Get the zigzag data for this block
        zigzag = zigzag_data[block_idx]
        
        # Perform each step of the decoding process
        quantized_block = inverse_zigzag_scan(zigzag)
        dequantized_block = inverse_quantization(quantized_block, q_matrix)
        idct_block = cv2.idct(dequantized_block.astype(np.float32))
        idct_clipped = np.clip(idct_block, 0, 255).astype(np.uint8)
        
        # Visualize each step
        fig, axs = plt.subplots(1, 4, figsize=(15, 4))
        
        # Zigzag array visualization
        axs[0].bar(range(len(zigzag)), zigzag)
        axs[0].set_title("Zigzag Coefficients")
        axs[0].set_xlabel("Position")
        axs[0].set_ylabel("Value")
        
        # Quantized block
        im1 = axs[1].imshow(quantized_block, cmap='viridis')
        axs[1].set_title("Quantized Block")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        
        # Dequantized block
        im2 = axs[2].imshow(dequantized_block, cmap='viridis')
        axs[2].set_title("Dequantized Block")
        plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        
        # IDCT result
        im3 = axs[3].imshow(idct_clipped, cmap='gray')
        axs[3].set_title("IDCT Result (Pixel Values)")
        plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show the original block
        i_start = i_block * 8
        j_start = j_block * 8
        i_end = min(i_start + 8, height)
        j_end = min(j_start + 8, width)
        
        original_block = gray_original[i_start:i_end, j_start:j_end]
        reconstructed_block = reconstructed[i_start:i_end, j_start:j_end]
        
        # Compare original vs reconstructed
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = ax1.imshow(original_block, cmap='gray')
        ax1.set_title("Original Block")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        im2 = ax2.imshow(reconstructed_block, cmap='gray')
        ax2.set_title("Reconstructed Block")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Calculate and display metrics
    st.subheader("Image Quality Metrics")
    
    # Mean Squared Error (MSE)
    mse = np.mean((gray_original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    
    # Create a metrics display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    with col2:
        st.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{psnr:.2f} dB")
    
    st.write("""
    - Lower MSE indicates better reconstruction quality.
    - Higher PSNR indicates better reconstruction quality. Typically, PSNR values above 30dB represent good quality.
    """)
    
    # Provide size comparison
    original_size = os.path.getsize(os.path.join("temp", "original.jpg"))
    
    # We'll save the reconstructed image with the specified quality to estimate the size
    quality_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cv2.imwrite(os.path.join("temp", f"reconstructed_q{quality}.jpg"), reconstructed, quality_param)
    compressed_size = os.path.getsize(os.path.join("temp", f"reconstructed_q{quality}.jpg"))
    
    # Calculate the compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    st.subheader("Compression Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Size", f"{original_size / 1024:.2f} KB")
    with col2:
        st.metric("Compressed Size", f"{compressed_size / 1024:.2f} KB")
    with col3:
        st.metric("Compression Ratio", f"{compression_ratio:.2f}:1")
    
    st.success("Image decoding completed! Proceed to the final comparison step.")

if __name__ == "__main__":
    app()