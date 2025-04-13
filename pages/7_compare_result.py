import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original, compressed):
    """Calculate various image quality metrics"""
    # Mean Squared Error (MSE)
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    
    # Structural Similarity Index (SSIM)
    ssim_value = ssim(original, compressed, data_range=255)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim_value
    }

def visualize_difference(original, compressed):
    """Visualize the difference between original and compressed images"""
    # Compute absolute difference
    diff = np.abs(original.astype(np.float32) - compressed.astype(np.float32)).astype(np.uint8)
    
    # Apply a colormap to better visualize differences
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    
    return diff, diff_colored

def app():
    st.title("Step 7: Compare Results")
    
    # Check if previous steps were completed
    if not os.path.exists(os.path.join("temp", "original.jpg")) or \
       not os.path.exists(os.path.join("temp", "reconstructed.jpg")):
        st.warning("Please complete the previous steps first.")
        return
    
    # Load original and reconstructed images
    original_img = cv2.imread(os.path.join("temp", "original.jpg"))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    reconstructed_img = cv2.imread(os.path.join("temp", "reconstructed.jpg"))
    reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for comparison
    if len(original_img.shape) == 3:
        gray_original = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_original = original_img
    
    if len(reconstructed_img.shape) == 3:
        gray_reconstructed = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_reconstructed = reconstructed_img
    
    st.write("""
    In this final step, we'll compare the original image with the compressed and reconstructed image 
    to evaluate the quality and efficiency of the JPEG compression process.
    """)
    
    # Side-by-side comparison of original and reconstructed images
    st.subheader("Original vs. Reconstructed Image")
    
    # Allow toggling between grayscale and color (if original was color)
    show_color = st.checkbox("Show in color (if available)", value=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if show_color and len(original_img.shape) == 3:
            st.image(original_img, caption="Original Image", use_column_width=True)
        else:
            st.image(gray_original, caption="Original Image (Grayscale)", use_column_width=True)
    
    with col2:
        if show_color and len(reconstructed_img.shape) == 3:
            st.image(reconstructed_img, caption="Reconstructed Image", use_column_width=True)
        else:
            st.image(gray_reconstructed, caption="Reconstructed Image (Grayscale)", use_column_width=True)
    
    # Calculate metrics
    metrics = calculate_metrics(gray_original, gray_reconstructed)
    
    # Display metrics
    st.subheader("Image Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.2f}")
        st.write("Lower is better")
    
    with col2:
        st.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{metrics['psnr']:.2f} dB")
        st.write("Higher is better (>30dB is good)")
    
    with col3:
        st.metric("Structural Similarity (SSIM)", f"{metrics['ssim']:.4f}")
        st.write("Higher is better (1.0 is perfect)")
    
    # Explanation of metrics
    with st.expander("Explanation of Quality Metrics"):
        st.write("""
        - **Mean Squared Error (MSE)**: The average of the squares of the errors between the original and compressed image. 
          Lower values indicate better quality.
        
        - **Peak Signal-to-Noise Ratio (PSNR)**: A logarithmic ratio of peak signal power to noise power. 
          Higher values (typically above 30dB) indicate better quality.
        
        - **Structural Similarity Index (SSIM)**: A perceptual metric that considers luminance, contrast, and structure. 
          Values range from -1 to 1, with 1 representing perfect similarity.
        """)
    
    # Show file sizes and compression ratio
    st.subheader("Compression Statistics")
    
    # Get qualities from previous steps if available
    quality = 50  # Default
    if os.path.exists(os.path.join("temp", "reconstructed_q50.jpg")):
        for q in range(1, 101):
            if os.path.exists(os.path.join("temp", f"reconstructed_q{q}.jpg")):
                quality = q
                break
    
    # Calculate file sizes
    original_size = os.path.getsize(os.path.join("temp", "original.jpg"))
    compressed_path = os.path.join("temp", f"reconstructed_q{quality}.jpg")
    if os.path.exists(compressed_path):
        compressed_size = os.path.getsize(compressed_path)
    else:
        # Save with the current quality if not available
        quality_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(compressed_path, cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR), quality_param)
        compressed_size = os.path.getsize(compressed_path)
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    savings_percentage = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Size", f"{original_size / 1024:.2f} KB")
    
    with col2:
        st.metric("Compressed Size", f"{compressed_size / 1024:.2f} KB")
    
    with col3:
        st.metric("Compression Ratio", f"{compression_ratio:.2f}:1 ({savings_percentage:.1f}% smaller)")
    
    # Visualize the differences between original and compressed
    st.subheader("Difference Visualization")
    
    diff, diff_colored = visualize_difference(gray_original, gray_reconstructed)
    
    # Display the difference images
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(diff, caption="Absolute Difference (Grayscale)", use_column_width=True,
                clamp=True)
        st.write("Brighter pixels indicate larger differences")
    
    with col2:
        st.image(diff_colored, caption="Color-mapped Difference", use_column_width=True)
        st.write("Blue: Small differences, Red: Large differences")
    
    # Histogram comparison
    st.subheader("Histogram Comparison")
    
    # Calculate histograms
    hist_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    hist_reconstructed = cv2.calcHist([gray_reconstructed], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_original, hist_original, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_reconstructed, hist_reconstructed, 0, 1, cv2.NORM_MINMAX)
    
    # Plot histograms
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(hist_original, color='blue', alpha=0.7, label='Original')
    ax.plot(hist_reconstructed, color='red', alpha=0.7, label='Reconstructed')
    
    ax.set_title("Pixel Intensity Histogram Comparison")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Normalized Frequency")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Allow zooming in to inspect details
    st.subheader("Detail Inspection")
    
    # Create a zoomed view of a region
    zoom_level = st.slider("Zoom Level", 1, 10, 4)
    
    # Let user select region to zoom
    height, width = gray_original.shape
    center_y = st.slider("Vertical Position", 0, height-1, height//2)
    center_x = st.slider("Horizontal Position", 0, width-1, width//2)
    
    # Calculate zoom region
    region_size = min(100, height, width) // zoom_level
    y_start = max(0, center_y - region_size//2)
    y_end = min(height, center_y + region_size//2)
    x_start = max(0, center_x - region_size//2)
    x_end = min(width, center_x + region_size//2)
    
    # Extract regions
    region_original = gray_original[y_start:y_end, x_start:x_end]
    region_reconstructed = gray_reconstructed[y_start:y_end, x_start:x_end]
    region_diff = diff[y_start:y_end, x_start:x_end]
    
    # Display zoomed regions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(region_original, caption="Original (Zoomed)", use_column_width=True)
    
    with col2:
        st.image(region_reconstructed, caption="Reconstructed (Zoomed)", use_column_width=True)
    
    with col3:
        st.image(region_diff, caption="Difference (Zoomed)", use_column_width=True, clamp=True)
    
    # Summary and conclusions
    st.subheader("Summary and Conclusions")
    
    st.write(f"""
    The JPEG compression process has reduced the image size from {original_size / 1024:.2f} KB to 
    {compressed_size / 1024:.2f} KB, achieving a compression ratio of {compression_ratio:.2f}:1 
    ({savings_percentage:.1f}% space saving).
    
    **Image Quality Assessment:**
    - MSE: {metrics['mse']:.2f} (Lower is better)
    - PSNR: {metrics['psnr']:.2f} dB (Higher is better)
    - SSIM: {metrics['ssim']:.4f} (Higher is better)
    
    **Observations:**
    - {'Good quality compression with minimal visible artifacts.' if metrics['psnr'] > 30 else 'Some compression artifacts are visible due to the lossy compression.'}
    - {'The structure of the image is very well preserved.' if metrics['ssim'] > 0.9 else 'Some structural details are lost in the compression process.'}
    - {'The histogram comparison shows that the overall pixel distribution is well preserved.' if np.mean(np.abs(hist_original - hist_reconstructed)) < 0.1 else 'The histogram comparison indicates some changes in the pixel intensity distribution.'}
    """)
    
    st.success("JPEG Compression Visualization Complete! You have now seen the entire JPEG compression process from start to finish.")

if __name__ == "__main__":
    app()