import argparse
import os
import numpy as np
from utils.image_io import save_image, read_compressed_file
from core.color_processing.color_transform import ycbcr_to_rgb
from core.color_processing.subsampling import apply_chroma_upsampling
from core.dct.block_processing import merge_blocks
from core.dct.dct import apply_idct_to_image
from core.quantization.dequantization_temp import optimize_dequantization_for_speed
from core.entropy_coding.zigzag_rle import apply_inverse_zigzag_and_rle
from core.entropy_coding.huffman.huffman_decoder import decode_huffman_from_file

def decompress_image(input_path, output_path):
    """
    Decompress a JPEG-like compressed file back to an image
    
    Args:
        input_path (str): Path to the compressed file
        output_path (str): Path where the decompressed image will be saved
    
    Returns:
        tuple: Compressed size and decompressed size in bytes
    """
    # 1. Read the compressed file
    compressed_data = read_compressed_file(input_path)
    compressed_size = os.path.getsize(input_path)
    
    # 2. Extract metadata and compressed data
    metadata = compressed_data['metadata']
    y_huffman = compressed_data['y_data']
    cb_huffman = compressed_data['cb_data']
    cr_huffman = compressed_data['cr_data']
    
    width = metadata['width']
    height = metadata['height']
    quality = metadata['quality']
    subsampling = metadata['subsampling']
    y_shape = metadata['y_shape']
    cb_shape = metadata['cb_shape']
    cr_shape = metadata['cr_shape']
    
    # 3. Huffman decoding
    y_rle = decode_huffman_from_file(y_huffman, metadata['y_huffman_table'])
    cb_rle = decode_huffman_from_file(cb_huffman, metadata['cb_huffman_table'])
    cr_rle = decode_huffman_from_file(cr_huffman, metadata['cr_huffman_table'])
    
    # 4. Run-length decoding
    y_zigzag = run_length_decode(y_rle)
    cb_zigzag = run_length_decode(cb_rle)
    cr_zigzag = run_length_decode(cr_rle)
    
    # 5. Inverse zigzag scan to convert 1D sequences back to 2D blocks
    y_quantized = inverse_zigzag_scan(y_zigzag)
    cb_quantized = inverse_zigzag_scan(cb_zigzag)
    cr_quantized = inverse_zigzag_scan(cr_zigzag)
    
    # 6. Dequantize DCT coefficients
    y_dct_blocks = apply_dequantization(y_quantized, quality=quality, is_luma=True)
    cb_dct_blocks = apply_dequantization(cb_quantized, quality=quality, is_luma=False)
    cr_dct_blocks = apply_dequantization(cr_quantized, quality=quality, is_luma=False)
    
    # 7. Apply inverse DCT to each block
    y_blocks = apply_idct_to_image(y_dct_blocks)
    cb_blocks = apply_idct_to_image(cb_dct_blocks)
    cr_blocks = apply_idct_to_image(cr_dct_blocks)
    
    # 8. Combine blocks back into channels
    y_channel = merge_blocks(y_blocks, y_shape)
    cb_channel = merge_blocks(cb_blocks, cb_shape)
    cr_channel = merge_blocks(cr_blocks, cr_shape)
    
    # 9. Create YCbCr image with subsampled chroma channels
    subsampled_ycbcr = np.zeros((height, width, 3), dtype=np.float32)
    subsampled_ycbcr[..., 0] = y_channel
    subsampled_ycbcr[..., 1] = cb_channel
    subsampled_ycbcr[..., 2] = cr_channel
    
    # 10. Apply chroma upsampling to restore full resolution
    ycbcr_image = apply_chroma_upsampling(subsampled_ycbcr, mode=subsampling)
    
    # 11. Convert YCbCr back to RGB
    rgb_image = ycbcr_to_rgb(ycbcr_image)
    
    # 12. Clip values to valid RGB range [0, 255] and convert to uint8
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    
    # 13. Save the decompressed image
    save_image(rgb_image, output_path)
    decompressed_size = os.path.getsize(output_path)
    
    return compressed_size, decompressed_size


def main():
    """Main function to parse arguments and start decompression"""
    parser = argparse.ArgumentParser(description='JPEG Image Decompression')
    parser.add_argument('input', help='Input compressed file')
    parser.add_argument('output', help='Output image file')
    
    args = parser.parse_args()
    
    try:
        compressed_size, decompressed_size = decompress_image(args.input, args.output)
        
        # Print decompression statistics
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Decompressed size: {decompressed_size:,} bytes")
        print(f"Expansion ratio: {decompressed_size / compressed_size:.2f}:1")
        print(f"Image decompressed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()