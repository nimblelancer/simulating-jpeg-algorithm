import argparse
import os
import numpy as np
from utils.image_io import read_image, save_compressed_file
from core.color_processing.color_transform import rgb_to_ycbcr
from core.color_processing.subsampling import apply_chroma_subsampling
from core.dct.block_processing import split_into_blocks, pad_image_to_multiple_of_8
from core.dct.dct import apply_dct_to_image
from core.quantization.quantization_temp import optimize_quantization_for_speed
from core.entropy_coding.zigzag_rle import apply_zigzag_and_rle
from core.entropy_coding.huffman.huffman_encoder import apply_huffman_to_encoded_data


def compress_image(input_path="assets/images/test/input.jpg", output_path="assets/images/test/output.compressed", quality=75, subsampling='4:2:0'):
    """
    Compress an image using JPEG-like algorithm
    
    Args:
        input_path (str): Path to the input image file
        output_path (str): Path where the compressed file will be saved
        quality (int): Compression quality (1-100), lower means more compression
        subsampling (str): Chroma subsampling mode ('4:4:4', '4:2:2', or '4:2:0')
    
    Returns:
        tuple: Original size and compressed size in bytes
    """
    # 1. Read the image
    image = read_image(input_path)
    height, width = image.shape[:2]
    original_size = os.path.getsize(input_path)
    # 2. Convert RGB to YCbCr color space
    ycbcr_image = rgb_to_ycbcr(image)
    # 3. Apply chroma subsampling
    subsampled_ycbcr = apply_chroma_subsampling(ycbcr_image, subsampling)
    # 4. Padding (if need) and Split each channel into 8x8 blocks
    padded_ycbcr = pad_image_to_multiple_of_8(subsampled_ycbcr)
    y_blocks = split_into_blocks(padded_ycbcr[..., 0])
    cb_blocks = split_into_blocks(padded_ycbcr[..., 1])
    cr_blocks = split_into_blocks(padded_ycbcr[..., 2])
    # 5. Apply DCT to each block
    y_dct_blocks = apply_dct_to_image(y_blocks)
    cb_dct_blocks = apply_dct_to_image(cb_blocks)
    cr_dct_blocks = apply_dct_to_image(cr_blocks)
    print("Done step 5")
    # 6. Quantize DCT coefficients
    # JPEG has different quantization tables for luminance and chrominance
    y_quantized = apply_quantization(y_dct_blocks, quality=quality)
    cb_quantized = apply_quantization(cb_dct_blocks, quality=quality)
    cr_quantized = apply_quantization(cr_dct_blocks, quality=quality)
    print("Done step 6")
    # 8. Zigzag scan and Run-length encoding (RLE)
    y_rle = apply_zigzag_and_rle(y_quantized)
    cb_rle = apply_zigzag_and_rle(cb_quantized)
    cr_rle = apply_zigzag_and_rle(cr_quantized)
    print("Done step 8")
    # 9. Huffman encoding
    y_huffman, y_huffman_table = apply_huffman_to_encoded_data(y_rle)
    cb_huffman, cb_huffman_table = apply_huffman_to_encoded_data(cb_rle)
    cr_huffman, cr_huffman_table = apply_huffman_to_encoded_data(cr_rle)
    print("Done step 9")
    # 10. Create metadata for the compressed file
    metadata = {
        'width': width,
        'height': height,
        'quality': quality,
        'subsampling': subsampling,
        'y_shape': subsampled_ycbcr[..., 0].shape,
        'cb_shape': subsampled_ycbcr[..., 1].shape,
        'cr_shape': subsampled_ycbcr[..., 2].shape,
        'y_huffman_table': y_huffman_table,
        'cb_huffman_table': cb_huffman_table,
        'cr_huffman_table': cr_huffman_table
    }
    print("Done step 10")
    # 11. Save the compressed data
    compressed_data = {
        'metadata': metadata,
        'y_data': y_huffman,
        'cb_data': cb_huffman,
        'cr_data': cr_huffman
    }
    print("Done step 11")
    save_compressed_file(compressed_data, output_path)
    compressed_size = os.path.getsize(output_path)
    
    return original_size, compressed_size


def main():
    """Main function to parse arguments and start compression"""
    parser = argparse.ArgumentParser(description='JPEG Image Compression')
    # parser.add_argument('input', help='Input image file')
    # parser.add_argument('output', help='Output compressed file')
    # parser.add_argument('-q', '--quality', type=int, default=75,
    #                     help='Compression quality (1-100, default: 75)')
    # parser.add_argument('-s', '--subsampling', choices=['4:4:4', '4:2:2', '4:2:0'],
    #                     default='4:2:0', help='Chroma subsampling mode (default: 4:2:0)')
    
    # args = parser.parse_args()
    
    # # Validate quality parameter
    # if args.quality < 1 or args.quality > 100:
    #     print("Quality must be between 1 and 100")
    #     return
    
    try:
        # original_size, compressed_size = compress_image(
        #     args.input, args.output, args.quality, args.subsampling
        # )

        original_size, compressed_size = compress_image()
        
        # Print compression statistics
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"Space saved: {(1 - compressed_size / original_size) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()