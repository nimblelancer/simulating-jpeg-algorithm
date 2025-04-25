import numpy as np
import matplotlib.pyplot as plt
import json
import os
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

def save_matrix_sample(blocks, filename, num_blocks=100):
        with open(filename, "w") as f:
            flat_blocks = blocks.reshape(-1, 8, 8)
            for i in range(min(num_blocks, flat_blocks.shape[0])):
                f.write(f"Block {i}:\n")
                np.savetxt(f, flat_blocks[i], fmt="%.2f")
                f.write("\n")

def save_rle_sample(rle_data, filename, num_blocks=5):
    with open(filename, "w") as f:
        for i, (dc, ac) in enumerate(rle_data[:num_blocks]):
            f.write(f"Block {i}:\n")
            f.write(f"  DC: {dc}\n")
            f.write(f"  AC: {ac}\n\n")
def stringify_keys(d):
    return {str(k): v for k, v in d.items()}

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
            print(" Start RGB to YCbCr")
            image = rgb_to_ycbcr(image)
            self.intermediates['ycbcr'] = image.copy()
            # np.save("encode_step1_ycbcr.npy", image)
        print(" Done RGB to YCbCr")
        
        # Bước 2: Padding ảnh và chia thành các khối 8x8
        print(" Start padding image")
        image = pad_image_to_multiple_of_8(image)
        print(" Done padding image")
        print(" Start split into blocks")
        blocks = split_into_blocks(image)
        print(" Done split into blocks")
        if blocks.ndim == 4:
            num_blocks = blocks.shape[0] * blocks.shape[1]  # ảnh xám
        else:
            num_blocks = blocks.shape[1] * blocks.shape[2]  # ảnh màu

        self.intermediates['blocks'] = blocks.copy()
        # np.save("encode_step2_blocks.npy", blocks)
        
        # Bước 3: DCT
        print(" Start DCT")
        dct_blocks = apply_dct_to_image(blocks)
        print(" Done DCT")
        self.intermediates['dct'] = dct_blocks.copy()
        # np.save("encode_step3_dct.npy", dct_blocks)
        # save_matrix_sample(dct_blocks, "encode_step3_dct_sample.txt")
        
        # Bước 4: Lượng tử hóa
        print(" Start Quantization")
        quant_blocks = optimize_quantization_for_speed(dct_blocks, self.quality)
        print(" Done Quantization")
        self.intermediates['quantized'] = quant_blocks.copy()
        # np.save("encode_step4_quantized.npy", quant_blocks)
        save_matrix_sample(quant_blocks, "encode_step4_quantized_sample.txt")
        # Mới kiểm tra đến đây
        # Bước 5: Zigzag và RLE
        print(" Start Zigzag và RLE")
        rle_data = apply_zigzag_and_rle(quant_blocks)
        print(" Done Zigzag và RLE")
        self.intermediates['rle'] = rle_data.copy()
        # save_rle_sample(rle_data, "encode_step5_rle.txt")

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
        print(" Done Huffman encode")
        # with open("encode_step6_dc_codes.json", "w") as f:
        #     json.dump(stringify_keys(dc_codes), f, indent=2)
        # with open("encode_step6_ac_codes.json", "w") as f:
        #     json.dump(stringify_keys(ac_codes), f, indent=2)
        # with open("encode_step6_encoded_bits.txt", "w") as f:
        #     f.write(f"Total bits: {total_bits}\n")
        #     f.write(f"First 512 bits: {''.join(format(byte, '08b') for byte in encoded_data[:64])}\n")

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
        if len(padded_shape) == 2:
            image_height, image_width = padded_shape
            num_channels = 1
        elif len(padded_shape) == 3:
            _, image_height, image_width = padded_shape
            num_channels = 3
        else:
            raise ValueError("padded_shape không hợp lệ")
        rle_data = huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits, image_width, image_height, num_channels)
        self.intermediates['rle'] = rle_data
        # with open("decompress_step1_rle.txt", "w") as f:
        #     for idx, (dc, ac) in enumerate(rle_data):
        #         f.write(f"Block {idx}  DC: {dc}\n")
        #         f.write(f"         AC: {ac}\n\n")

        # Bước 2: Giải RLE và zigzag
        quant_blocks = apply_inverse_zigzag_and_rle(rle_data, padded_shape)
        self.intermediates['quantized'] = quant_blocks.copy()
        # np.save("decompress_step2_quantized.npy", quant_blocks)
        flat_q = quant_blocks.reshape(-1, 8, 8)
        with open("decompress_step2_sample_blocks.txt", "w") as f:
            for i in range(min(100, flat_q.shape[0])):
                f.write(f"Block {i}:\n")
                np.savetxt(f, flat_q[i], fmt="%d")
                f.write("\n")

        # Bước 3: Giải lượng tử hóa
        print("Quality at dequantization:", self.quality)
        dct_blocks = optimize_dequantization_for_speed(quant_blocks, self.quality)
        self.intermediates['dct'] = dct_blocks.copy()

        # np.save("decompress_step3_dct.npy", dct_blocks)
        # Sample 5 DCT block
        # flat_d = dct_blocks.reshape(-1, 8, 8)
        # with open("decompress_step3_sample_blocks.txt", "w") as f:
        #     for i in range(min(100, flat_d.shape[0])):
        #         f.write(f"DCT Block {i}:\n")
        #         np.savetxt(f, flat_d[i], fmt="%.2f")
        #         f.write("\n")

        # Bước 4: IDCT
        pixel_blocks = apply_idct_to_image(dct_blocks)
        self.intermediates['blocks'] = pixel_blocks.copy()

        # np.save("decompress_step4_pixel_blocks.npy", pixel_blocks)

        # Bước 5: Gộp khối
        image = merge_blocks(pixel_blocks, original_shape)
        self.intermediates['merged'] = image.copy()

        # merged_img = Image.fromarray(
        #     image.astype(np.uint8) if image.ndim == 2 else image[:, :, 0].astype(np.uint8)
        # )
        # merged_img.save("decompress_step5_merged.png")

        # Bước 6: Chuyển YCbCr sang RGB nếu là ảnh màu
        if image.ndim == 3:
            image = ycbcr_to_rgb(image)
            self.intermediates['rgb'] = image.copy()
            print("Pixel values range:", np.min(image), np.max(image))
            return image
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