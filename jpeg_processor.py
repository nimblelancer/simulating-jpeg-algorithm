import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils.image_io import save_image, save_npy, save_encoded_bytes_to_jpg
from core.color_processing.color_transform import rgb_to_ycbcr, ycbcr_to_rgb
from core.color_processing.subsampling import apply_chroma_subsampling
from core.dct.block_processing import pad_image_to_multiple_of_8, split_into_blocks, merge_blocks
from core.dct.dct import apply_dct_to_image, apply_idct_to_image
from core.quantization.quantization import optimize_quantization_for_speed
from core.quantization.dequantization import optimize_dequantization_for_speed
from core.entropy_coding.zigzag_rle import apply_zigzag_and_rle, apply_inverse_zigzag_and_rle
from core.entropy_coding.huffman.huffman_encoder import build_frequency_table, build_huffman_tree, build_huffman_codes, huffman_encode
from core.entropy_coding.huffman.huffman_decoder import huffman_decode_bitstring
from PIL import Image

class JPEGProcessor:
    """
    Lớp xử lý pipeline nén và giải nén JPEG, hỗ trợ visualization cho Streamlit.
    
    Attributes:
    -----------
    quality : int
        Hệ số chất lượng (1-100)
    intermediates : dict
        Lưu kết quả trung gian của các bước
    """
    def __init__(self, quality=50):
        if not 1 <= quality <= 100:
            raise ValueError("Hệ số chất lượng phải từ 1 đến 100")
        self.quality = quality

    def encode_pipeline(self, image):
        """
        Pipeline nén JPEG, lưu kết quả trung gian.
        
        Parameters:
        -----------
        image : ndarray
            Ảnh đầu vào: (h, w) hoặc (h, w, 3), dtype=float32, giá trị [0, 255]
        
        Returns:
        --------
        dict
            {
                'encoded_data': bytes,
                'dc_codes': dict,
                'ac_codes': dict,
                'shape': tuple,
                'total_bits': int,
                'intermediates': dict
            }
        """
        
        # Kiểm tra đầu vào
        if image.ndim not in (2, 3):
            raise ValueError("Ảnh phải là mảng 2D (xám) hoặc 3D (màu)")
        if image.max() > 255 or image.min() < 0:
            raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
        save_image(image, "original.png")
        
        # Bước 1: Chuyển RGB sang YCbCr nếu là ảnh màu
        if image.ndim == 3:
            print(" Start RGB to YCbCr")
            image = rgb_to_ycbcr(image)
            save_image(image, "encode_step_ycbcr.png")
        print(" Done RGB to YCbCr")
        
        # Bước 2: Padding ảnh và chia thành các khối 8x8
        print(" Start padding image")
        image = pad_image_to_multiple_of_8(image)
        print(" Done padding image")
        print(" Start split into blocks")
        blocks = split_into_blocks(image)
        print(" Done split into blocks")
        save_npy(blocks, "encode_step_blocks.npy") 
        if blocks.ndim == 4:
            num_blocks = blocks.shape[0] * blocks.shape[1]  # ảnh xám
        else:
            num_blocks = blocks.shape[1] * blocks.shape[2]  # ảnh màu

        # Bước 3: DCT
        print(" Start DCT")
        dct_blocks = apply_dct_to_image(blocks)
        print(" Done DCT")
        save_npy(dct_blocks, "encode_step_dct.npy")
        
        # Bước 4: Lượng tử hóa
        print(" Start Quantization")
        quant_blocks = optimize_quantization_for_speed(dct_blocks, self.quality)
        print(" Done Quantization")
        save_npy(quant_blocks, "encode_step_quantized.npy")

        # Bước 5: Zigzag và RLE
        print(" Start Zigzag và RLE")
        rle_data, dc_original = apply_zigzag_and_rle(quant_blocks)
        print(" Done Zigzag và RLE")
        save_npy(rle_data, "encode_step_rle.npy", allow_object=True)

        # Bước 6: Huffman
        print(" Start Huffman encode")
        # Nếu ảnh màu:
        if isinstance(rle_data[0], list):  # Ảnh màu
            flat_rle = [item for channel in rle_data for item in channel]
        else:
            flat_rle = rle_data  # Ảnh xám

        dc_freq, ac_freq = build_frequency_table(flat_rle)
        dc_tree = build_huffman_tree(dc_freq)
        ac_tree = build_huffman_tree(ac_freq)
        dc_codes = build_huffman_codes(dc_tree)
        ac_codes = build_huffman_codes(ac_tree)
        encoded_data, total_bits = huffman_encode(rle_data, dc_codes, ac_codes)
        save_encoded_bytes_to_jpg(encoded_data, "compressed_image.jpg")
        print(" Done Huffman encode")

        # Lưu shape
        if blocks.ndim == 4:
            # Ảnh xám
            padded_shape = (blocks.shape[0] * 8, blocks.shape[1] * 8)
        elif blocks.ndim == 5:
            # Ảnh màu
            padded_shape = (blocks.shape[0], blocks.shape[1] * 8, blocks.shape[2] * 8)
        
        return {
            'encoded_data': encoded_data,
            'dc_codes': dc_codes,
            'ac_codes': ac_codes,
            'padded_shape': padded_shape,
            'total_bits': total_bits,
            'encoded_dc_original': dc_original
        }

    def decode_pipeline(self, encoded_data, dc_codes, ac_codes, padded_shape, total_bits, original_shape):
        """
        Pipeline giải nén JPEG, lưu kết quả trung gian.
        
        Parameters:
        -----------
        encoded_data : bytes
            Dữ liệu mã hóa
        dc_codes : dict
            Bảng mã Huffman cho DC
        ac_codes : dict
            Bảng mã Huffman cho AC
        shape : tuple
            Shape gốc: (h, w) hoặc (c, h, w)
        
        Returns:
        --------
        dict
            {
                'image': ndarray,
                'intermediates': dict
            }
        """
        
        # Kiểm tra đầu vào
        if not encoded_data or not dc_codes or not ac_codes:
            raise ValueError("Dữ liệu và bảng mã phải không rỗng")
        if len(padded_shape) not in (2, 3):
            raise ValueError("shape phải là (h, w) hoặc (c, h, w)")
        
        # Bước 1: Giải mã Huffman
        if len(padded_shape) == 2:
            image_height, image_width = padded_shape
            num_channels = 1
        elif len(padded_shape) == 3:
            _, image_height, image_width = padded_shape
            num_channels = 3
        else:
            raise ValueError("padded_shape không hợp lệ")
        rle_data = huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits, image_width, image_height, num_channels)
        save_npy(rle_data, "decode_step_huffman_decode.npy", allow_object=True)

        # Bước 2: Giải RLE và zigzag
        quant_blocks = apply_inverse_zigzag_and_rle(rle_data, padded_shape)
        save_npy(quant_blocks, "decode_step_inverse_zigzag.npy")

        # Bước 3: Giải lượng tử hóa
        print("Quality at dequantization:", self.quality)
        dct_blocks = optimize_dequantization_for_speed(quant_blocks, self.quality)
        save_npy(dct_blocks, "decode_step_dequantized.npy")

        # Bước 4: IDCT
        pixel_blocks = apply_idct_to_image(dct_blocks)
        save_npy(pixel_blocks, "decode_step_idct.npy")

        # Bước 5: Gộp khối
        image = merge_blocks(pixel_blocks, original_shape)

        # Bước 6: Chuyển YCbCr sang RGB nếu là ảnh màu
        if image.ndim == 3:
            image = ycbcr_to_rgb(image)
            print("Pixel values range:", np.min(image), np.max(image))
            save_image(image, "decompressed_image.jpg")
            return image
        print("Image shape:", image.shape)
        save_image(image.astype(np.uint8), "decompressed_image.jpg")
        return image.astype(np.uint8)
