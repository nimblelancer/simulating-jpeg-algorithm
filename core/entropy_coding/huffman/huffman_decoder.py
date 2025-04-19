import numpy as np
import json
from .huffman_encoder import build_huffman_tree, build_huffman_codes

def huffman_decode_bitstring(encoded_data, dc_codes, ac_codes):
    """
    Giải mã dữ liệu Huffman thành DC và AC coefficients.
    
    Parameters:
    -----------
    encoded_data : bytes
        Dữ liệu mã hóa
    dc_codes : dict
        Bảng mã Huffman cho DC coefficients
    ac_codes : dict
        Bảng mã Huffman cho AC coefficients
    
    Returns:
    --------
    list
        List [channel][block] = (dc, ac) hoặc [block] = (dc, ac)
    
    Raises:
    -------
    ValueError
        Nếu dữ liệu hoặc bảng mã không hợp lệ
    """
    if not encoded_data or not dc_codes or not ac_codes:
        raise ValueError("Dữ liệu và bảng mã phải không rỗng")
    
    # Chuyển bytes thành bitstring
    bitstring = ""
    for byte in encoded_data:
        bitstring += bin(byte)[2:].zfill(8)
    
    reversed_dc = {code: value for value, code in dc_codes.items()}
    reversed_ac = {code: value for value, code in ac_codes.items()}
    
    decoded_data = []
    current_code = ""
    current_block = None
    ac_list = []
    
    for bit in bitstring:
        current_code += bit
        if current_block is None:  # Đang tìm DC
            if current_code in reversed_dc:
                current_block = {'dc': reversed_dc[current_code], 'ac': []}
                current_code = ""
        else:  # Đang tìm AC
            if current_code in reversed_ac:
                ac_list.append(reversed_ac[current_code])
                current_code = ""
                if reversed_ac[current_code] == (0, 0):  # EOB
                    current_block['ac'] = ac_list
                    decoded_data.append((current_block['dc'], ac_list))
                    current_block = None
                    ac_list = []
    
    # Thêm block cuối nếu chưa hoàn thành
    if current_block and ac_list:
        current_block['ac'] = ac_list
        decoded_data.append((current_block['dc'], ac_list))
    
    return decoded_data

def huffman_decode_jpeg_data(encoded_data, dc_codes, ac_codes, block_shape):
    """
    Giải mã dữ liệu JPEG đã mã hóa Huffman.
    
    Parameters:
    -----------
    encoded_data : bytes
        Dữ liệu mã hóa
    dc_codes : dict
        Bảng mã Huffman cho DC coefficients
    ac_codes : dict
        Bảng mã Huffman cho AC coefficients
    block_shape : tuple
        Shape ban đầu: (h, w, 8, 8) hoặc (c, h, w, 8, 8)
    
    Returns:
    --------
    ndarray
        Mảng khối: (h, w, 8, 8) hoặc (c, h, w, 8, 8), dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu đầu vào không hợp lệ
    """
    if len(block_shape) not in (4, 5) or block_shape[-2:] != (8, 8):
        raise ValueError("block_shape phải là (h, w, 8, 8) hoặc (c, h, w, 8, 8)")
    
    # Giải mã Huffman
    decoded_data = huffman_decode_bitstring(encoded_data, dc_codes, ac_codes)
    
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    if len(block_shape) == 4:
        h, w = block_shape[:2]
        if len(decoded_data) != h * w:
            raise ValueError("Số block giải mã không khớp với block_shape")
        
        blocks = np.zeros((h * w, 64), dtype=np.int32)
        for i, (dc, ac) in enumerate(decoded_data):
            array = [dc]
            for run, value in ac:
                if run > 0:
                    array.extend([0] * run)
                if value != 0 or run < 16:
                    array.append(value)
            array = array[:64]
            if len(array) < 64:
                array.extend([0] * (64 - len(array)))
            blocks[i] = array
        
        blocks[:, zigzag_indices] = blocks
        return blocks.reshape(h, w, 8, 8).astype(np.int32)
    
    c, h, w = block_shape[:3]
    if len(decoded_data) != c * h * w:
        raise ValueError("Số block giải mã không khớp với block_shape")
    
    blocks = np.zeros((c, h * w, 64), dtype=np.int32)
    idx = 0
    for ch in range(c):
        for i in range(h * w):
            dc, ac = decoded_data[idx]
            array = [dc]
            for run, value in ac:
                if run > 0:
                    array.extend([0] * run)
                if value != 0 or run < 16:
                    array.append(value)
            array = array[:64]
            if len(array) < 64:
                array.extend([0] * (64 - len(array)))
            blocks[ch, i] = array
            idx += 1
    
    blocks[:, :, zigzag_indices] = blocks
    return blocks.reshape(c, h, w, 8, 8).astype(np.int32)

def decode_huffman_from_file(file_path):
    """
    Đọc và giải mã dữ liệu Huffman từ file.
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file mã hóa
    
    Returns:
    --------
    ndarray
        Mảng khối: (h, w, 8, 8) hoặc (c, h, w, 8, 8), dtype=int32
    
    Raises:
    -------
    IOError
        Nếu không đọc được file
    ValueError
        Nếu dữ liệu trong file không hợp lệ
    """
    try:
        with open(file_path, 'rb') as f:
            # Đọc kích thước JSON
            json_size = int.from_bytes(f.read(4), byteorder='big')
            # Đọc JSON chứa codes và shape
            json_data = json.loads(f.read(json_size).decode('utf-8'))
            dc_codes = json_data['dc_codes']
            ac_codes = json_data['ac_codes']
            block_shape = tuple(json_data['shape'])
            # Đọc dữ liệu mã hóa
            encoded_data = f.read()
        
        return huffman_decode_jpeg_data(encoded_data, dc_codes, ac_codes, block_shape)
    
    except Exception as e:
        raise ValueError(f"Không thể giải mã file {file_path}: {str(e)}")