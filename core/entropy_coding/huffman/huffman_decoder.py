import numpy as np
import json
from .huffman_encoder import build_huffman_tree, build_huffman_codes

def huffman_decode_bitstring(encoded_data, dc_codes, ac_codes, total_bits):
    if not encoded_data or not dc_codes or not ac_codes:
        raise ValueError("Dữ liệu và bảng mã phải không rỗng")

    # Chuyển bytes → bitstring
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

    while i < len(bitstring):
        block_idx += 1

        # Decode DC
        current_code = ""
        while i < len(bitstring) and len(current_code) <= max_dc_len:
            current_code += bitstring[i]
            i += 1
            if current_code in reversed_dc:
                dc_diff = reversed_dc[current_code]
                break
        else:
            print("❌ Không tìm thấy mã DC nào khớp tại block", block_idx)
            break

        dc_value = previous_dc + dc_diff
        previous_dc = dc_value

        # Decode AC
        ac_list = []
        current_code = ""
        while i < len(bitstring):
            current_code += bitstring[i]
            i += 1
            if current_code in reversed_ac:
                ac_value = reversed_ac[current_code]
                ac_list.append(ac_value)
                current_code = ""
                if ac_value == (0, 0):  # EOB
                    break
            elif len(current_code) > max_ac_len:
                raise ValueError(f"❌ AC code không hợp lệ tại block {block_idx}, index {i}")

        decoded_data.append((dc_value, ac_list))

    print("✅ Tổng số block giải mã được:", len(decoded_data))
    return decoded_data
    