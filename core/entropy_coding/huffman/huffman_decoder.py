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

def huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits, image_width, image_height):
    if not encoded_data or not dc_codes or not ac_codes:
        raise ValueError("Dữ liệu và bảng mã phải không rỗng")

    # Tính tổng số block cần giải mã
    blocks_x = (image_width + 7) // 8
    blocks_y = (image_height + 7) // 8
    total_blocks = blocks_x * blocks_y

    bitstring = ''.join(bin(byte)[2:].zfill(8) for byte in encoded_data)
    bitstring = bitstring[:total_bits]

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
        if block_idx % 100 == 0:
            print(f"🟢 Đang xử lý block {block_idx}/{total_blocks} tại bit index {i}")

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

        if i + dc_size > len(bitstring):
            raise ValueError(f"❌ Không đủ bit để đọc DC magnitude tại block {block_idx}, index {i}")
        dc_bits = bitstring[i:i+dc_size]
        i += dc_size
        dc_diff = decode_magnitude(dc_bits)
        dc_value = previous_dc + dc_diff
        previous_dc = dc_value

        # === GIẢI MÃ AC ===
        ac_list = []
        while len(ac_list) < 63:
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
                ac_list.extend([0] * (63 - len(ac_list)))
                break

            if i + size > len(bitstring):
                raise ValueError(f"❌ Không đủ bit để đọc AC magnitude tại block {block_idx}, index {i}")

            ac_bits = bitstring[i:i+size]
            i += size
            ac_value = decode_magnitude(ac_bits)

            ac_list.extend([0] * runlength)
            ac_list.append(ac_value)

        while len(ac_list) < 63:
            ac_list.append(0)

        decoded_data.append((dc_value, ac_list))

    print("✅ Giải mã hoàn tất: tổng số block =", len(decoded_data))
    return decoded_data


    