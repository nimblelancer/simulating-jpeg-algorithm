import numpy as np
from collections import Counter
from core.entropy_coding.huffman.node import Node
import heapq
import json

def build_frequency_table(data):
    dc_freq = Counter()
    for dc, _ in data:
        dc_size = int(dc).bit_length() if dc != 0 else 0
        dc_freq[dc_size] += 1

    # AC
    ac_freq = Counter()
    for _, ac in data:
        for run, value in ac:
            size = int(value).bit_length() if value != 0 else 0
            ac_freq[(run, size)] += 1

    # Thêm EOB và ZRL
    ac_freq[(0, 0)] += 1
    ac_freq[(15, 0)] += 1

    return dc_freq, ac_freq

def build_huffman_tree(freq_table):
    """
    Xây dựng cây Huffman từ bảng tần suất.
    
    Parameters:
    -----------
    freq_table : dict
        Dict với key là giá trị và value là tần suất
    
    Returns:
    --------
    Node
        Node gốc của cây Huffman
    
    Raises:
    -------
    ValueError
        Nếu bảng tần suất rỗng
    """
    if not freq_table:
        raise ValueError("Bảng tần suất không được rỗng")
    
    priority_queue = [Node(value, freq) for value, freq in freq_table.items()]
    heapq.heapify(priority_queue)
    
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(priority_queue, parent)
    
    return priority_queue[0]

def build_huffman_codes(root):
    """
    Xây dựng mã Huffman từ cây Huffman.
    
    Parameters:
    -----------
    root : Node
        Node gốc của cây Huffman
    
    Returns:
    --------
    dict
        Dict với key là giá trị và value là chuỗi bit ('0', '1')
    
    Raises:
    -------
    ValueError
        Nếu root rỗng
    """
    if not root:
        raise ValueError("Node gốc không được rỗng")
    
    codes = {}
    
    def traverse(node, code):
        if node:
            if node.value is not None:
                codes[node.value] = code or "0"  # Đảm bảo mã không rỗng
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
    
    traverse(root, '')
    return codes

def safe_bit_length(x):
    return int(x).bit_length() if x != 0 else 0

def huffman_encode(data, dc_codes, ac_codes):
    """
    Mã hóa dữ liệu sử dụng mã Huffman riêng cho DC và AC.
    
    Parameters:
    -----------
    data : list
        List [channel][block] = (dc, ac) hoặc [block] = (dc, ac)
    dc_codes : dict
        Bảng mã Huffman cho DC coefficients
    ac_codes : dict
        Bảng mã Huffman cho AC coefficients
    
    Returns:
    --------
    tuple
        (encoded_bytes, total_bits): Dữ liệu mã hóa dạng bytes và số bit
    """

    def safe_bit_length(x):
        return int(x).bit_length() if x != 0 else 0

    def encode_magnitude(value, size):
        if value >= 0:
            return bin(value)[2:].zfill(size)
        else:
            max_val = (1 << size) - 1
            inverted = max_val - abs(value)
            return bin(inverted)[2:].zfill(size)

    if not data or not dc_codes or not ac_codes:
        raise ValueError("Dữ liệu và bảng mã phải không rỗng")

    bitstring = ""

    if isinstance(data[0], list):  # Ảnh màu
        for channel in data:
            for dc, ac in channel:
                dc_size = safe_bit_length(dc)
                if dc_size not in dc_codes:
                    raise ValueError(f"DC size {dc_size} không có trong bảng mã")
                bitstring += dc_codes[dc_size]
                if dc_size > 0:
                    bitstring += encode_magnitude(dc, dc_size)

                for run, value in ac:
                    size = safe_bit_length(value)
                    key = (run, size)
                    if key not in ac_codes:
                        raise ValueError(f"AC key {key} không có trong bảng mã")
                    bitstring += ac_codes[key]
                    if size > 0:
                        bitstring += encode_magnitude(value, size)

                if not ac or ac[-1] != (0, 0):
                    bitstring += ac_codes[(0, 0)]

    else:  # Ảnh xám
        for dc, ac in data:
            dc_size = safe_bit_length(dc)
            if dc_size not in dc_codes:
                raise ValueError(f"DC size {dc_size} không có trong bảng mã")
            bitstring += dc_codes[dc_size]
            if dc_size > 0:
                bitstring += encode_magnitude(dc, dc_size)

            for run, value in ac:
                size = safe_bit_length(value)
                key = (run, size)
                if key not in ac_codes:
                    raise ValueError(f"AC key {key} không có trong bảng mã")
                bitstring += ac_codes[key]
                if size > 0:
                    bitstring += encode_magnitude(value, size)

            if not ac or ac[-1] != (0, 0):
                bitstring += ac_codes[(0, 0)]

    # Chuyển bitstring thành bytes
    byte_array = []
    for i in range(0, len(bitstring), 8):
        byte_str = bitstring[i:i+8]
        if len(byte_str) < 8:
            byte_str += '0' * (8 - len(byte_str))
        byte_array.append(int(byte_str, 2))

    return bytes(byte_array), len(bitstring)

def apply_huffman_to_encoded_data(encoded_blocks):
    """
    Áp dụng mã hóa Huffman cho dữ liệu sau zigzag và RLE.
    
    Parameters:
    -----------
    encoded_blocks : list
        List từ apply_zigzag_and_rle: [channel][block] = (dc, ac) hoặc [block] = (dc, ac)
    
    Returns:
    --------
    dict
        {
            'encoded_data': bytes,
            'dc_codes': dict,
            'ac_codes': dict,
            'shape': tuple (h, w) hoặc (c, h, w),
            'total_bits': int
        }
    
    Raises:
    -------
    ValueError
        Nếu dữ liệu đầu vào không hợp lệ
    """
    if not encoded_blocks or not isinstance(encoded_blocks, list):
        raise ValueError("encoded_blocks phải là list không rỗng")
    
    # Xác định shape
    if isinstance(encoded_blocks[0], list):
        c = len(encoded_blocks)
        h = w = int((len(encoded_blocks[0]) ** 0.5))
        shape = (c, h, w)
    else:
        h = w = int((len(encoded_blocks) ** 0.5))
        shape = (h, w)
    
    # Bảng tần suất
    dc_freq, ac_freq = build_frequency_table(encoded_blocks)
    
    # Cây Huffman
    dc_tree = build_huffman_tree(dc_freq)
    ac_tree = build_huffman_tree(ac_freq)
    
    # Bảng mã
    dc_codes = build_huffman_codes(dc_tree)
    ac_codes = build_huffman_codes(ac_tree)
    
    # Mã hóa
    encoded_data, total_bits = huffman_encode(encoded_blocks, dc_codes, ac_codes)
    
    return {
        'encoded_data': encoded_data,
        'dc_codes': dc_codes,
        'ac_codes': ac_codes,
        'shape': shape,
        'total_bits': total_bits
    }