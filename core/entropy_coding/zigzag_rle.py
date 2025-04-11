import numpy as np

def zigzag_scan(block):
    """
    Thực hiện quét zigzag một khối 8x8 thành mảng 1D.
    
    Input:
        block: Ma trận 8x8
        
    Output:
        Mảng 1D sau khi quét zigzag
    """
    # Sắp xếp cho quét zigzag 8x8
    zigzag_indices = [
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    
    # Chuyển khối 8x8 thành mảng 1D
    flattened = block.flatten()
    
    # Sắp xếp lại theo thứ tự zigzag
    result = np.zeros(64, dtype=flattened.dtype)
    for i, idx in enumerate(zigzag_indices):
        result[i] = flattened[idx]
    
    return result

def run_length_encode(array):
    """
    Mã hóa RLE cho mảng 1D sau zigzag.
    """
    # Chuyển array sang list nếu là np.ndarray
    if isinstance(array, np.ndarray):
        array = array.tolist()

    if not array:  # Bây giờ là list → kiểm tra hợp lệ
        return []

    result = []
    count = 1
    current = array[0]

    for i in range(1, len(array)):
        if array[i] == current:
            count += 1
        else:
            result.append((current, count))
            current = array[i]
            count = 1

    result.append((current, count))
    return result

def inverse_rle(rle_data):
    array = []
    for val, count in rle_data:
        array.extend([val] * count)
    return np.array(array)

def inverse_zigzag(array):
    zigzag_indices = [
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    block = np.zeros((8, 8), dtype=np.float32)
    for i, idx in enumerate(zigzag_indices):
        row = idx // 8
        col = idx % 8
        block[row, col] = array[i] if i < len(array) else 0
    return block

def apply_zigzag_and_rle(blocks):
    """
    Áp dụng zigzag scan + RLE cho từng block 8x8.
    
    Input:
        blocks: 4D (h, w, 8, 8) hoặc 5D (c, h, w, 8, 8)
    
    Output:
        List các block đã được zigzag + RLE
    """
    
    result = []

    if blocks.ndim == 5:
        c, h, w, _, _ = blocks.shape
        for ch in range(c):
            channel_result = []
            for i in range(h):
                for j in range(w):
                    zigzag = zigzag_scan(blocks[ch, i, j])
                    rle = run_length_encode(zigzag)
                    channel_result.append(rle)
            result.append(channel_result)

    elif blocks.ndim == 4:
        h, w, _, _ = blocks.shape
        for i in range(h):
            for j in range(w):
                zigzag = zigzag_scan(blocks[i, j])
                rle = run_length_encode(zigzag)
                result.append(rle)

    else:
        raise ValueError("Blocks phải là mảng 4D hoặc 5D")

    return result

def apply_inverse_zigzag_and_rle(rle_blocks, image_shape):
    """
    Giải mã RLE và zigzag để khôi phục lại các khối 8x8.

    Input:
        rle_blocks: List các block đã được mã hóa RLE.
                    Nếu là ảnh màu → List 3 phần tử, mỗi phần tử là list block của từng channel
                    Nếu là ảnh xám → List các block
        image_shape: Tuple thể hiện shape ban đầu: 
                     - (h, w) cho ảnh xám
                     - (c, h, w) cho ảnh màu

    Output:
        Mảng NumPy 4D hoặc 5D các block (h, w, 8, 8) hoặc (c, h, w, 8, 8)
    """
    # Xử lý ảnh xám
    if len(image_shape) == 2:
        h, w = image_shape
        blocks = np.zeros((h, w, 8, 8), dtype=np.float32)
        idx = 0
        for i in range(h):
            for j in range(w):
                rle = rle_blocks[idx]
                restored_1d = inverse_rle(rle)
                blocks[i, j] = inverse_zigzag(restored_1d)
                idx += 1
        return blocks

    # Xử lý ảnh màu
    elif len(image_shape) == 3:
        c, h, w = image_shape
        blocks = np.zeros((c, h, w, 8, 8), dtype=np.float32)
        for ch in range(c):
            idx = 0
            for i in range(h):
                for j in range(w):
                    rle = rle_blocks[ch][idx]
                    restored_1d = inverse_rle(rle)
                    blocks[ch, i, j] = inverse_zigzag(restored_1d)
                    idx += 1
        return blocks

    else:
        raise ValueError("image_shape phải có dạng (h, w) hoặc (c, h, w)")
