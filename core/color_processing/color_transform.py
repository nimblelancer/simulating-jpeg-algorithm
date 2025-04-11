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

    ycbcr = np.stack((Y, Cb, Cr), axis=2)  # Shape: (H, W, 3)
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
