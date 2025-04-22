import numpy as np

def zigzag_scan(block):
    """
    Chuyển khối 8x8 thành mảng 1D theo thứ tự zigzag.
    
    Parameters:
    -----------
    block : ndarray
        Khối 8x8, shape (8, 8), dtype=int32, hệ số lượng tử hóa
    
    Returns:
    --------
    ndarray
        Mảng 1D, shape (64,), dtype=int32
    
    Raises:
    -------
    ValueError
        Nếu shape không đúng hoặc dtype không phải int32
    """
    if block.shape != (8, 8):
        raise ValueError("Khối phải có shape (8, 8)")
    if block.dtype != np.int32:
        raise ValueError("Khối phải có dtype int32")
    
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    
    return block.flatten()[zigzag_indices].astype(np.int32)

def run_length_encode(array):
    if array.shape != (64,) or array.dtype != np.int32:
        raise ValueError("Mảng phải có shape (64,) và dtype int32")

    dc = array[0]
    ac = []
    run = 0

    for value in array[1:]:
        if value == 0:
            run += 1
        else:
            while run >= 16:
                ac.append((15, 0))  # ZRL
                run -= 16
            ac.append((run, value))
            run = 0

    # ✅ EOB chuẩn (chỉ 1 cái cuối)
    if run > 0:
        ac.append((0, 0))

    return dc, ac

def apply_zigzag_and_rle(blocks):
    if blocks.ndim not in (4, 5) or blocks.shape[-2:] != (8, 8):
        raise ValueError("blocks phải là mảng 4D hoặc 5D")
    if blocks.dtype != np.int32:
        raise ValueError("blocks phải có dtype int32")

    result = []

    if blocks.ndim == 4:
        h, w = blocks.shape[:2]
        previous_dc = 0
        for i in range(h):
            for j in range(w):
                zigzagged = zigzag_scan(blocks[i, j])
                dc, ac = run_length_encode(zigzagged)
                dc_diff = dc - previous_dc
                previous_dc = dc
                result.append((dc_diff, ac))

    else:  # ảnh màu (5D)
        c, h, w = blocks.shape[:3]
        for ch in range(c):
            channel_result = []
            previous_dc = 0  # ✅ fix: mỗi kênh cần có riêng
            for i in range(h):
                for j in range(w):
                    zigzagged = zigzag_scan(blocks[ch, i, j])
                    dc, ac = run_length_encode(zigzagged)
                    dc_diff = dc - previous_dc  # ✅ chuẩn JPEG
                    previous_dc = dc
                    channel_result.append((dc_diff, ac))
            result.append(channel_result)

    return result

def apply_inverse_zigzag_and_rle(rle_blocks, image_shape):
    """
    Giải mã RLE và zigzag để khôi phục các khối 8x8.
    """
    if len(image_shape) == 2:
        h, w = image_shape
        block_h, block_w = h // 8, w // 8
        if len(rle_blocks) != block_h * block_w:
            raise ValueError("Số lượng rle_blocks không khớp với image_shape")

        blocks = np.zeros((block_h, block_w, 8, 8), dtype=np.int32)
        for idx, rle in enumerate(rle_blocks):
            i, j = divmod(idx, block_w)
            flat = rle_to_array(rle)
            blocks[i, j] = inverse_zigzag(flat)
        return blocks

    elif len(image_shape) == 3:
        c, h, w = image_shape
        block_h, block_w = h // 8, w // 8
        if len(rle_blocks) != c or any(len(channel) != block_h * block_w for channel in rle_blocks):
            raise ValueError("Số lượng rle_blocks không khớp với image_shape")

        blocks = np.zeros((c, block_h, block_w, 8, 8), dtype=np.int32)
        for ch in range(c):
            for idx, rle in enumerate(rle_blocks[ch]):
                i, j = divmod(idx, block_w)
                flat = rle_to_array(rle)
                blocks[ch, i, j] = inverse_zigzag(flat)
        return blocks

    else:
        raise ValueError("image_shape phải có dạng (h, w) hoặc (c, h, w)")

def rle_to_array(rle_data):
    dc, ac = rle_data
    array = [dc] + ac
    array = array[:64]
    if len(array) < 64:
        array.extend([0] * (64 - len(array)))
    return np.array(array, dtype=np.int32)

def inverse_zigzag(array):
    if array.shape != (64,) or array.dtype != np.int32:
        raise ValueError("Mảng phải có shape (64,) và dtype int32")
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
    ])
    block = np.zeros(64, dtype=np.int32)
    block[zigzag_indices] = array
    return block.reshape(8, 8)
