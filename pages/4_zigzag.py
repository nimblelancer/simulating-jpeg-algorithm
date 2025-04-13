import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

def zigzag_scan(block):
    """Convert an 8x8 block to a 1D array using zigzag pattern"""
    rows, cols = block.shape
    solution = []
    
    # Start from top-left corner (0, 0)
    i, j = 0, 0
    
    # Direction initially set to "up-right"
    going_up = True
    
    for _ in range(rows * cols):
        solution.append(block[i, j])
        
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
    
    return np.array(solution)

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

def visualize_zigzag(block, zigzag_array):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display original block
    im1 = ax1.imshow(block, cmap='viridis')
    ax1.set_title('Original 8x8 Block')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Display zigzag array as a 1D plot
    ax2.stem(zigzag_array, use_line_collection=True)
    ax2.set_title('Zigzag Scanned Values')
    ax2.set_xlabel('Position in Zigzag Scan')
    ax2.set_ylabel('Coefficient Value')
    
    plt.tight_layout()
    return fig

def visualize_zigzag_pattern():
    """Visualize the zigzag scanning pattern on an 8x8 grid"""
    pattern = np.zeros((8, 8))
    
    # Start from top-left corner (0, 0)
    i, j = 0, 0
    
    # Direction initially set to "up-right"
    going_up = True
    
    for idx in range(64):
        pattern[i, j] = idx
        
        if going_up:
            # Moving up-right
            if j == 7:  # Reached right boundary
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
            if i == 7:  # Reached bottom boundary
                j += 1
                going_up = True
            elif j == 0:  # Reached left boundary
                i += 1
                going_up = True
            else:  # Continue moving down-left
                i += 1
                j -= 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pattern, cmap='viridis')
    
    # Add text annotations for each cell
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, f"{int(pattern[i, j])}",
                          ha="center", va="center", color="white", fontsize=9)
    
    ax.set_title("Zigzag Scanning Pattern")
    plt.colorbar(im)
    
    return fig

def run_length_encode(zigzag_array):
    """Perform run-length encoding on zigzag array"""
    # Count zeros at the end
    i = len(zigzag_array) - 1
    while i >= 0 and zigzag_array[i] == 0:
        i -= 1
    
    # Truncate trailing zeros
    truncated = zigzag_array[:i+1]
    
    # Perform run-length encoding
    rle = []
    zero_count = 0
    
    for val in truncated:
        if val == 0:
            zero_count += 1
        else:
            rle.append((zero_count, val))
            zero_count = 0
    
    # If there are zeros at the end, add a special EOB marker
    if i < len(zigzag_array) - 1:
        rle.append("EOB")  # End of block marker
    
    return rle

def app():
    st.title("Step 4: Zigzag Scanning")
    
    # Check if quantized coefficients exist
    quant_file = os.path.join("temp", "quantized_coefficients.npy")
    if not os.path.exists(quant_file):
        st.warning("Please complete the quantization step first.")
        return
    
    # Load the quantized coefficients
    quantized_image = np.load(quant_file)
    
    st.write("""
    Zigzag scanning converts 8x8 blocks of quantized DCT coefficients into a 1D array. 
    This step orders coefficients from low to high frequency, which helps group non-zero values together 
    for more efficient run-length encoding.
    """)
    
    # Display the zigzag pattern
    st.subheader("Zigzag Scanning Pattern")
    pattern_fig = visualize_zigzag_pattern()
    st.pyplot(pattern_fig)
    
    # Apply zigzag scanning to all blocks
    height, width = quantized_image.shape
    zigzag_data = []
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Get the current 8x8 block (handle edge cases)
            i_end = min(i+8, height)
            j_end = min(j+8, width)
            block = quantized_image[i:i_end, j:j_end]
            
            # If the block is smaller than 8x8, pad it
            if block.shape[0] < 8 or block.shape[1] < 8:
                padded_block = np.zeros((8, 8), dtype=np.float32)
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block
            
            # Apply zigzag scanning
            zigzag = zigzag_scan(block)
            
            # Store the zigzag scanned data
            zigzag_data.append(zigzag)
    
    # Save zigzag data
    np.save(os.path.join("temp", "zigzag_data.npy"), np.array(zigzag_data))
    
    # Apply run-length encoding
    rle_data = [run_length_encode(zigzag) for zigzag in zigzag_data]
    
    # Save run-length encoded data
    with open(os.path.join("temp", "rle_data.txt"), 'w') as f:
        for block_idx, rle in enumerate(rle_data):
            f.write(f"Block {block_idx}: {rle}\n")
    
    # Allow user to visualize zigzag on a specific block
    st.subheader("Visualize Zigzag Scanning on a Sample Block")
    
    # Create sliders for block selection
    max_i = (height - 1) // 8
    max_j = (width - 1) // 8
    i_block = st.slider("Block row", 0, max_i, max_i // 2)
    j_block = st.slider("Block column", 0, max_j, max_j // 2)
    
    # Get the selected block
    block_idx = i_block * ((width + 7) // 8) + j_block
    if block_idx < len(zigzag_data):
        i_start = i_block * 8
        j_start = j_block * 8
        i_end = min(i_start + 8, height)
        j_end = min(j_start + 8, width)
        
        selected_block = quantized_image[i_start:i_end, j_start:j_end]
        
        # If the block is smaller than 8x8, pad it for visualization
        if selected_block.shape[0] < 8 or selected_block.shape[1] < 8:
            padded_block = np.zeros((8, 8), dtype=np.float32)
            padded_block[:selected_block.shape[0], :selected_block.shape[1]] = selected_block
            selected_block = padded_block
        
        selected_zigzag = zigzag_data[block_idx]
        
        # Visualize
        fig = visualize_zigzag(selected_block, selected_zigzag)
        st.pyplot(fig)
        
        # Show run-length encoding
        st.subheader("Run-Length Encoding of Selected Block")
        rle = rle_data[block_idx]
        
        # Count non-zero elements
        non_zero = np.count_nonzero(selected_zigzag)
        
        st.write(f"Original block has {non_zero} non-zero elements out of 64")
        st.write(f"Run-Length Encoded data: {rle}")
        
        # Calculate compression ratio
        if non_zero > 0:
            compression_ratio = 64 / (len(rle) * 2 if "EOB" not in rle else len(rle) * 2 - 1)
            st.write(f"Compression ratio (assuming 2 bytes per RLE entry): {compression_ratio:.2f}:1")
    
    # Show statistics for all blocks
    st.subheader("Overall Zigzag Compression Statistics")
    
    # Calculate zeros after quantization
    total_zeros_quant = np.sum(quantized_image == 0)
    total_elements = quantized_image.size
    zero_percentage = (total_zeros_quant / total_elements) * 100
    
    st.write(f"Percentage of zero coefficients after quantization: {zero_percentage:.2f}%")
    
    # Count total entries in RLE
    total_rle_entries = sum(len(rle) for rle in rle_data)
    
    # Estimate bits (rough approximation)
    estimated_bits_original = total_elements * 8  # Assuming 8 bits per coefficient
    estimated_bits_rle = total_rle_entries * 16   # Assuming 16 bits per RLE entry (8 for zero count, 8 for value)
    
    est_compression_ratio = estimated_bits_original / estimated_bits_rle if estimated_bits_rle > 0 else float('inf')
    
    st.write(f"Estimated overall compression ratio from zigzag + RLE: {est_compression_ratio:.2f}:1")
    
    st.success("Zigzag scanning completed! Proceed to the Huffman coding step.")

if __name__ == "__main__":
    app()