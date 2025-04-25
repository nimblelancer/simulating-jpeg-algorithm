import numpy as np
import json
from .huffman_encoder import build_huffman_tree, build_huffman_codes

def decode_magnitude(bits):
    if not bits:
        return 0
    if bits[0] == '1':
        return int(bits, 2)
    else:
        inverted = ''.join('1' if b == '0' else '0' for b in bits)
        return -int(inverted, 2)

def huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits,
                              image_width, image_height, num_channels=1):
    if not encoded_data or not dc_codes or not ac_codes:
        raise ValueError("Dữ liệu và bảng mã phải không rỗng")

    blocks_x = (image_width + 7) // 8
    blocks_y = (image_height + 7) // 8
    blocks_per_channel = blocks_x * blocks_y
    total_blocks = blocks_per_channel * num_channels

    # Tối ưu: convert trực tiếp sang chuỗi nhị phân
    bit_array = np.unpackbits(np.frombuffer(encoded_data, dtype=np.uint8))[:total_bits]
    bitstring = ''.join(bit_array.astype(str))

    reversed_dc = {code: value for value, code in dc_codes.items()}
    reversed_ac = {code: value for value, code in ac_codes.items()}
    max_dc_len = max(len(k) for k in reversed_dc)
    max_ac_len = max(len(k) for k in reversed_ac)

    decoded_data = []
    i = 0
    previous_dc = 0
    block_idx = 0

    while i < len(bitstring) and len(decoded_data) < total_blocks:
        block_idx += 1

        # === GIẢI MÃ DC ===
        current_code = ""
        start_i = i
        while i < len(bitstring) and len(current_code) <= max_dc_len:
            current_code += bitstring[i]
            i += 1
            if current_code in reversed_dc:
                dc_size = reversed_dc[current_code]
                break
        else:
            raise ValueError(f"❌ Không tìm thấy mã DC tại block {block_idx}, từ bit index {start_i}")

        if dc_size > 0:
            if i + dc_size > len(bitstring):
                raise ValueError(f"❌ Không đủ bit để đọc DC magnitude tại block {block_idx}, index {i}")
            dc_bits = bitstring[i:i+dc_size]
            i += dc_size
            dc_diff = decode_magnitude(dc_bits)
        else:
            dc_diff = 0

        dc_value = previous_dc + dc_diff
        previous_dc = dc_value

        # === GIẢI MÃ AC ===
        ac_list = []
        while True:
            current_code = ""
            start_i = i
            while i < len(bitstring) and len(current_code) <= max_ac_len:
                current_code += bitstring[i]
                i += 1
                if current_code in reversed_ac:
                    runlength, size = reversed_ac[current_code]
                    break
            else:
                raise ValueError(f"❌ Không tìm thấy mã AC tại block {block_idx}, từ bit index {start_i}")

            if (runlength, size) == (0, 0):  # EOB
                ac_list.append((0, 0))
                break

            if i + size > len(bitstring):
                raise ValueError(f"❌ Không đủ bit để đọc AC magnitude tại block {block_idx}, index {i}")

            ac_bits = bitstring[i:i+size]
            i += size
            ac_value = decode_magnitude(ac_bits)

            ac_list.append((runlength, ac_value))

        decoded_data.append((dc_value, ac_list))

    print("✅ Giải mã hoàn tất: tổng số block =", len(decoded_data))

    if num_channels > 1:
        grouped = []
        for ch in range(num_channels):
            offset = ch * blocks_per_channel
            grouped.append(decoded_data[offset : offset + blocks_per_channel])
        return grouped

    return decoded_data
