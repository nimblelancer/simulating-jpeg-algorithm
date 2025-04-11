import numpy as np
from collections import Counter, defaultdict
import heapq

def build_frequency_table(data):
    """
    Xây dựng bảng tần suất cho các giá trị trong dữ liệu.
    
    Input:
        data: Mảng 1D chứa dữ liệu cần mã hóa
    
    Output:
        Dict với key là giá trị và value là tần suất
    """
    return Counter(data)

def build_huffman_tree(freq_table):
    """
    Xây dựng cây Huffman từ bảng tần suất.
    
    Input:
        freq_table: Dict với key là giá trị và value là tần suất
    
    Output:
        Node gốc của cây Huffman
    """
    # Tạo hàng đợi ưu tiên với các node lá
    priority_queue = [Node(value, freq) for value, freq in freq_table.items()]
    heapq.heapify(priority_queue)
    
    # Xây dựng cây Huffman
    while len(priority_queue) > 1:
        # Lấy hai node có tần suất thấp nhất
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        
        # Tạo node cha với tần suất là tổng của hai con
        parent = Node(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        
        # Thêm node cha vào hàng đợi
        heapq.heappush(priority_queue, parent)
    
    # Trả về node gốc
    return priority_queue[0] if priority_queue else None

def build_huffman_codes(root):
    """
    Xây dựng mã Huffman từ cây Huffman.
    
    Input:
        root: Node gốc của cây Huffman
    
    Output:
        Dict với key là giá trị và value là mã Huffman dạng chuỗi '0' và '1'
    """
    codes = {}
    
    def traverse(node, code):
        if node:
            # Nếu là node lá
            if node.value is not None:
                codes[node.value] = code
            # Duyệt con trái (thêm bit 0)
            traverse(node.left, code + '0')
            # Duyệt con phải (thêm bit 1)
            traverse(node.right, code + '1')
    
    traverse(root, '')
    return codes

def huffman_encode(data, codes):
    """
    Mã hóa dữ liệu sử dụng mã Huffman.
    
    Input:
        data: Mảng 1D chứa dữ liệu cần mã hóa
        codes: Dict với key là giá trị và value là mã Huffman
    
    Output:
        Chuỗi bit đã mã hóa và bảng mã để giải mã
    """
    # Chuyển dữ liệu sang chuỗi bit
    encoded_data = ''
    for value in data:
        encoded_data += codes[value]
    
    return encoded_data, codes

def apply_huffman_to_encoded_data(encoded_blocks):
    """
    Áp dụng mã hóa Huffman cho dữ liệu đã được zigzag + RLE.
    
    Input:
        encoded_blocks: List hoặc mảng 1D các giá trị sau khi zigzag và RLE.
                        Ví dụ: [0, 5, 0, 3, 2, 0, 0, 1, 7, ...]
    
    Output:
        dict: {
            'encoded_data': Dữ liệu Huffman đã mã hóa (bitstring hoặc bytes),
            'huffman_codes': Bảng mã Huffman (dict: giá trị -> mã bit),
            'original_length': Độ dài chuỗi đầu vào (phục vụ giải mã)
        }
    """
    # Bước 1: Xây dựng bảng tần suất
    freq_table = build_frequency_table(encoded_blocks)

    # Bước 2: Xây dựng cây Huffman
    huffman_tree = build_huffman_tree(freq_table)

    # Bước 3: Sinh bảng mã Huffman
    huffman_codes = build_huffman_codes(huffman_tree)

    # Bước 4: Mã hóa dữ liệu
    encoded_data, original_length = huffman_encode(encoded_blocks, huffman_codes)

    # Trả về kết quả
    return {
        'encoded_data': encoded_data,
        'huffman_codes': huffman_codes,
        'original_length': original_length  # hữu ích cho việc giải mã
    }


