import numpy as np
from collections import defaultdict

def huffman_decode_bitstring(encoded_data, huffman_codes):
    """
    Giải mã chuỗi bit dựa trên bảng mã Huffman.
    
    Input:
        encoded_data: Chuỗi bit đã mã hóa ('0' và '1')
        huffman_codes: Dict với key là giá trị và value là mã Huffman
    
    Output:
        Mảng giá trị đã giải mã
    """
    # Đảo ngược dict mã để tìm kiếm nhanh
    reversed_codes = {code: value for value, code in huffman_codes.items()}
    
    # Giải mã
    decoded_data = []
    current_code = ''
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_codes:
            decoded_data.append(reversed_codes[current_code])
            current_code = ''
    
    return decoded_data

def huffman_decode_jpeg_data(encoded_data, huffman_codes, block_shape, chunk_size=64):
    """
    Giải mã dữ liệu JPEG đã được mã hóa Huffman.
    
    Input:
        encoded_data: Chuỗi bit đã mã hóa ('0' và '1')
        huffman_codes: Dict với key là giá trị và value là mã Huffman
        block_shape: Kích thước ban đầu của mảng các khối (c, h, w, 8, 8) hoặc (h, w, 8, 8)
        chunk_size: Kích thước của mỗi khối sau khi làm phẳng (mặc định là 64 cho khối 8x8)
    
    Output:
        Mảng các khối đã được khôi phục với kích thước ban đầu
    """
    # Giải mã Huffman
    decoded_flat = huffman_decode_bitstring(encoded_data, huffman_codes)
    
    # Xác định xem có nhiều kênh màu hay không
    is_multichannel = len(block_shape) == 5
    
    if is_multichannel:
        c, h, w, block_h, block_w = block_shape
        result = np.zeros(block_shape, dtype=np.float32)
        
        # Tính toán số lượng phần tử sau khi giải mã RLE cho mỗi khối
        elements_per_block = 2 * chunk_size  # 2 vì mỗi cặp RLE có (giá trị, số lần lặp)
        total_blocks = c * h * w
        
        # Tạo lại các khối
        block_idx = 0
        pos = 0
        
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    # Lấy phần dữ liệu RLE cho khối hiện tại
                    if pos + elements_per_block <= len(decoded_flat):
                        block_rle = decoded_flat[pos:pos + elements_per_block]
                        pos += elements_per_block
                    else:
                        # Xử lý trường hợp khối cuối cùng có thể không đủ dữ liệu
                        block_rle = decoded_flat[pos:]
                    
                    # Giải mã RLE
                    block_zigzag = run_length_decode(block_rle)
                    
                    # Điền đủ 64 phần tử nếu thiếu (thường là thêm số 0)
                    if len(block_zigzag) < chunk_size:
                        block_zigzag = np.pad(block_zigzag, (0, chunk_size - len(block_zigzag)))
                    
                    # Chuyển lại từ zigzag thành khối 8x8
                    block = inverse_zigzag_scan(block_zigzag[:chunk_size])
                    
                    # Gán vào kết quả
                    result[channel, i, j] = block
                    
                    block_idx += 1
    else:
        h, w, block_h, block_w = block_shape
        result = np.zeros(block_shape, dtype=np.float32)
        
        # Tính toán số lượng phần tử sau khi giải mã RLE cho mỗi khối
        elements_per_block = 2 * chunk_size  # 2 vì mỗi cặp RLE có (giá trị, số lần lặp)
        total_blocks = h * w
        
        # Tạo lại các khối
        block_idx = 0
        pos = 0
        
        for i in range(h):
            for j in range(w):
                # Lấy phần dữ liệu RLE cho khối hiện tại
                if pos + elements_per_block <= len(decoded_flat):
                    block_rle = decoded_flat[pos:pos + elements_per_block]
                    pos += elements_per_block
                else:
                    # Xử lý trường hợp khối cuối cùng có thể không đủ dữ liệu
                    block_rle = decoded_flat[pos:]
                
                # Giải mã RLE
                block_zigzag = run_length_decode(block_rle)
                
                # Điền đủ 64 phần tử nếu thiếu (thường là thêm số 0)
                if len(block_zigzag) < chunk_size:
                    block_zigzag = np.pad(block_zigzag, (0, chunk_size - len(block_zigzag)))
                
                # Chuyển lại từ zigzag thành khối 8x8
                block = inverse_zigzag_scan(block_zigzag[:chunk_size])
                
                # Gán vào kết quả
                result[i, j] = block
                
                block_idx += 1
    
    return result

def decode_huffman_from_file(file_path):
    """
    Đọc và giải mã dữ liệu đã được mã hóa Huffman từ file.
    
    Input:
        file_path: Đường dẫn đến file đã được mã hóa
    
    Output:
        Dữ liệu đã được giải mã
    """
    # Đọc dữ liệu từ file
    with open(file_path, 'rb') as f:
        # Đọc kích thước của bảng mã Huffman
        codes_size = int.from_bytes(f.read(4), byteorder='big')
        
        # Đọc bảng mã Huffman
        huffman_codes_bytes = f.read(codes_size)
        huffman_codes = eval(huffman_codes_bytes.decode('utf-8'))
        
        # Đọc kích thước của block_shape
        shape_size = int.from_bytes(f.read(4), byteorder='big')
        
        # Đọc block_shape
        block_shape_bytes = f.read(shape_size)
        block_shape = eval(block_shape_bytes.decode('utf-8'))
        
        # Đọc số byte của dữ liệu đã mã hóa
        data_size = int.from_bytes(f.read(4), byteorder='big')
        
        # Đọc dữ liệu đã mã hóa
        encoded_bytes = f.read(data_size)
        
        # Chuyển đổi các byte thành chuỗi bit
        encoded_data = ''
        for byte in encoded_bytes:
            # Chuyển byte thành 8 bit và bỏ đi tiền tố '0b'
            bits = bin(byte)[2:].zfill(8)
            encoded_data += bits
    
    # Giải mã dữ liệu
    return huffman_decode_jpeg_data(encoded_data, huffman_codes, block_shape)