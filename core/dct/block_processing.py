import numpy as np

# region Split block
def pad_image_to_multiple_of_8(image):
    """
    Pad ảnh để chiều cao và chiều rộng chia hết cho 8.

    Input:
        image: Mảng NumPy 2D (H, W) hoặc 3D (H, W, C)

    Output:
        Ảnh sau khi pad
    """
    if image.ndim == 2:
        H, W = image.shape
        pad_h = (8 - H % 8) if H % 8 != 0 else 0
        pad_w = (8 - W % 8) if W % 8 != 0 else 0
        return np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')

    elif image.ndim == 3:
        try:
            H, W, C = image.shape
        except Exception as e:
            print("Unpacking failed for image.shape:", image.shape)
            raise e
        pad_h = (8 - H % 8) if H % 8 != 0 else 0
        pad_w = (8 - W % 8) if W % 8 != 0 else 0
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

    else:
        raise ValueError("Input image must be 2D or 3D array.")



def split_into_blocks(image):
    """
    Chia ảnh thành các khối 8x8.

    Input:
        image: Mảng NumPy 2D (H, W) hoặc 3D (C, H, W)

    Output:
        Mảng NumPy 4D hoặc 5D:
            - (H//8, W//8, 8, 8) nếu ảnh 2D
            - (C, H//8, W//8, 8, 8) nếu ảnh 3D
    """
    if image.ndim == 2:
        H, W = image.shape
        h_blocks = H // 8
        w_blocks = W // 8
        blocks = image.reshape(h_blocks, 8, w_blocks, 8)
        blocks = blocks.transpose(0, 2, 1, 3)  # (h, w, 8, 8)
        return blocks

    elif image.ndim == 3:
        C, H, W = image.shape
        h_blocks = H // 8
        w_blocks = W // 8
        blocks = image.reshape(C, h_blocks, 8, w_blocks, 8)
        blocks = blocks.transpose(0, 1, 3, 2, 4)  # (C, h, w, 8, 8)
        return blocks

    else:
        raise ValueError("Image must be 2D or 3D")
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