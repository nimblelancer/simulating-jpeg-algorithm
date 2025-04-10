import numpy as np

# region Convert RGB to YCbCr

def rgb_to_ycbcr(image):
    """
    Chuyển đổi ảnh RGB sang không gian màu YCbCr.
    
    Input:
        image: Mảng NumPy 3D (H, W, 3) với các giá trị pixel RGB
    
    Output:
        Mảng NumPy 3D (3, H, W) gồm 3 kênh Y, Cb, Cr
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có dạng (H, W, 3)")

    image = image.astype(np.float32)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    ycbcr = np.stack((Y, Cb, Cr), axis=0)  # Shape: (3, H, W)
    return ycbcr
# endregion

# region Convert YCbCr to RGB

def ycbcr_to_rgb(ycbcr):
    """
    Chuyển ảnh YCbCr (C, H, W) về RGB (H, W, 3)

    Input:
        ycbcr: Mảng NumPy (3, H, W) gồm Y, Cb, Cr

    Output:
        Mảng NumPy (H, W, 3) với giá trị RGB [0-255]
    """
    Y, Cb, Cr = ycbcr[0], ycbcr[1], ycbcr[2]

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    rgb = np.stack((R, G, B), axis=2)  # (H, W, 3)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

# endregion

# region Split block
def pad_image_to_multiple_of_8(image):
    """
    Pad ảnh để kích thước chiều cao và chiều rộng chia hết cho 8.
    
    Input:
        image: Mảng NumPy 2D (H, W) hoặc 3D (C, H, W)
    
    Output:
        Ảnh sau khi pad
    """
    if image.ndim == 2:
        image = image[np.newaxis, :, :]  # Thêm chiều kênh cho ảnh grayscale

    C, H, W = image.shape
    pad_h = (8 - H % 8) if H % 8 != 0 else 0
    pad_w = (8 - W % 8) if W % 8 != 0 else 0

    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')
    return padded

def split_into_blocks(image):
    """
    Chia ảnh thành các khối 8x8.

    Input:
        image: Mảng NumPy 3D (C, H, W) đã pad

    Output:
        Mảng NumPy 5D (C, H//8, W//8, 8, 8)
    """
    C, H, W = image.shape
    h_blocks = H // 8
    w_blocks = W // 8

    blocks = image.reshape(C, h_blocks, 8, w_blocks, 8)
    blocks = blocks.transpose(0, 1, 3, 2, 4)  # (C, h, w, 8, 8)
    return blocks
# endregion

# region Merge blocks
def merge_blocks(blocks):
    """
    Ghép các khối 8x8 (C, h, w, 8, 8) thành ảnh đầy đủ (C, H, W)

    Input:
        blocks: Mảng NumPy 5D (C, h, w, 8, 8)

    Output:
        Mảng NumPy 3D (C, H, W)
    """
    C, h, w, _, _ = blocks.shape
    image = blocks.transpose(0, 1, 3, 2, 4)  # (C, h, 8, w, 8)
    image = image.reshape(C, h * 8, w * 8)
    return image
# endregion