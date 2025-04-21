import numpy as np
import matplotlib.pyplot as plt
import os
from utils.image_io import read_image, save_compressed_file
from utils.metrics import calculate_psnr
from core.color_processing.color_transform import rgb_to_ycbcr, ycbcr_to_rgb
from core.color_processing.subsampling import apply_chroma_subsampling
from core.dct.block_processing import pad_image_to_multiple_of_8, split_into_blocks, merge_blocks
from core.dct.dct import apply_dct_to_image, apply_idct_to_image
from core.quantization.quantization import optimize_quantization_for_speed
from core.quantization.dequantization import optimize_dequantization_for_speed
from core.entropy_coding.zigzag_rle import apply_zigzag_and_rle, apply_inverse_zigzag_and_rle
from core.entropy_coding.huffman.huffman_encoder import build_frequency_table, build_huffman_tree, build_huffman_codes, huffman_encode
from core.entropy_coding.huffman.huffman_decoder import huffman_decode_bitstring

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
        self.intermediates = {}

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
        self.intermediates = {}
        
        # Kiểm tra đầu vào
        if image.ndim not in (2, 3):
            raise ValueError("Ảnh phải là mảng 2D (xám) hoặc 3D (màu)")
        if image.max() > 255 or image.min() < 0:
            raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
        self.intermediates['input'] = image.copy()
        
        # Bước 1: Chuyển RGB sang YCbCr nếu là ảnh màu
        if image.ndim == 3:
            image = rgb_to_ycbcr(image)
            self.intermediates['ycbcr'] = image.copy()
        
        # Bước 2: Padding ảnh và chia thành các khối 8x8
        image = pad_image_to_multiple_of_8(image)
        blocks = split_into_blocks(image)
        if blocks.ndim == 4:
            num_blocks = blocks.shape[0] * blocks.shape[1]  # ảnh xám
        else:
            num_blocks = blocks.shape[1] * blocks.shape[2]  # ảnh màu

        self.intermediates['blocks'] = blocks.copy()
        
        # Bước 3: DCT
        dct_blocks = apply_dct_to_image(blocks)
        self.intermediates['dct'] = dct_blocks.copy()
        
        # Bước 4: Lượng tử hóa
        quant_blocks = optimize_quantization_for_speed(dct_blocks, self.quality)
        self.intermediates['quantized'] = quant_blocks.copy()
        
        # Bước 5: Zigzag và RLE
        rle_data = apply_zigzag_and_rle(quant_blocks)
        self.intermediates['rle'] = rle_data.copy()
        
        # Bước 6: Huffman
        dc_freq, ac_freq = build_frequency_table(rle_data)
        dc_tree = build_huffman_tree(dc_freq)
        ac_tree = build_huffman_tree(ac_freq)
        dc_codes = build_huffman_codes(dc_tree)
        ac_codes = build_huffman_codes(ac_tree)
        encoded_data, total_bits = huffman_encode(rle_data, dc_codes, ac_codes)
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
            'intermediates': self.intermediates
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
        self.intermediates = {}
        
        # Kiểm tra đầu vào
        if not encoded_data or not dc_codes or not ac_codes:
            raise ValueError("Dữ liệu và bảng mã phải không rỗng")
        if len(padded_shape) not in (2, 3):
            raise ValueError("shape phải là (h, w) hoặc (c, h, w)")
        
        # Bước 1: Giải mã Huffman
        rle_data = huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits)
        print("rle_data type:", type(rle_data))
        print("rle_data length:", len(rle_data))
        print("First element type:", type(rle_data[0]))
        self.intermediates['rle'] = rle_data
        
        # Bước 2: Giải RLE và zigzag
        quant_blocks = apply_inverse_zigzag_and_rle(rle_data, padded_shape)
        self.intermediates['quantized'] = quant_blocks.copy()
        
        # Bước 3: Giải lượng tử hóa
        print("Quality at dequantization:", self.quality)
        dct_blocks = optimize_dequantization_for_speed(quant_blocks, self.quality)
        self.intermediates['dct'] = dct_blocks.copy()
        
        # Bước 4: IDCT
        pixel_blocks = apply_idct_to_image(dct_blocks)
        self.intermediates['blocks'] = pixel_blocks.copy()
        
        # Bước 5: Gộp khối
        image = merge_blocks(pixel_blocks, original_shape)
        self.intermediates['merged'] = image.copy()
        
        # Bước 6: Chuyển YCbCr sang RGB nếu là ảnh màu
        if image.ndim == 3:
            image = ycbcr_to_rgb(image)
            self.intermediates['rgb'] = image.copy()
        print("Image shape:", image.shape)
        return image.astype(np.uint8)
        # return {
        #     'image': image,
        #     'intermediates': self.intermediates
        # }

    def save_intermediate_images(self, output_dir):
        """
        Lưu kết quả trung gian dưới dạng ảnh để Streamlit visualize.
        
        Parameters:
        -----------
        output_dir : str
            Thư mục lưu ảnh
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for step, data in self.intermediates.items():
            if step in ('input', 'ycbcr', 'merged', 'rgb'):
                plt.imsave(f"{output_dir}/{step}.png", data.astype(np.uint8), cmap='gray' if data.ndim == 2 else None)
            elif step == 'blocks':
                if data.ndim == 4:
                    img = merge_blocks(data)
                    plt.imsave(f"{output_dir}/{step}.png", img.astype(np.uint8), cmap='gray')
                else:
                    for ch in range(data.shape[0]):
                        img = merge_blocks(data[ch])
                        plt.imsave(f"{output_dir}/{step}_ch{ch}.png", img.astype(np.uint8), cmap='gray')
            elif step in ('dct', 'quantized'):
                if data.ndim == 4:
                    sample = data[0, 0]
                    plt.figure()
                    plt.imshow(sample, cmap='hot')
                    plt.colorbar()
                    plt.savefig(f"{output_dir}/{step}_sample.png")
                    plt.close()
                else:
                    for ch in range(data.shape[0]):
                        sample = data[ch, 0, 0]
                        plt.figure()
                        plt.imshow(sample, cmap='hot')
                        plt.colorbar()
                        plt.savefig(f"{output_dir}/{step}_ch{ch}_sample.png")
                        plt.close()

    def compare_images(self, img1, img2):
        """
        So sánh hai ảnh bằng PSNR.
        
        Parameters:
        -----------
        img1, img2 : ndarray
            Hai ảnh cùng shape, dtype=float32, giá trị [0, 255]
        
        Returns:
        --------
        float
            Giá trị PSNR (dB)
        """
        return calculate_psnr(img1, img2)