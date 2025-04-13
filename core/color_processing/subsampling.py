import numpy as np

def apply_chroma_subsampling(ycbcr_image, subsampling='4:2:0'):
    """
    Áp dụng lấy mẫu phụ cho các kênh màu Cb và Cr trong ảnh YCbCr.
    
    Parameters:
    -----------
    ycbcr_image : ndarray
        Ảnh YCbCr, shape (H, W, 3), dtype=float32, giá trị trong [0, 255]
    subsampling : str, optional
        Kiểu lấy mẫu phụ ('4:4:4', '4:2:2', '4:2:0'), default='4:2:0'
    
    Returns:
    --------
    tuple
        (y_channel, cb_channel, cr_channel):
        - y_channel: shape (H, W)
        - cb_channel, cr_channel:
            - 4:4:4: shape (H, W)
            - 4:2:2: shape (H, W//2)
            - 4:2:0: shape (H//2, W//2)
        Tất cả đều là dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng, giá trị pixel ngoài [0, 255], hoặc subsampling không hỗ trợ
    """
    if ycbcr_image.ndim != 3 or ycbcr_image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải có shape (H, W, 3)")
    if ycbcr_image.max() > 255 or ycbcr_image.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    height, width = ycbcr_image.shape[:2]
    y_channel = ycbcr_image[:, :, 0].copy()
    cb_channel = ycbcr_image[:, :, 1].copy()
    cr_channel = ycbcr_image[:, :, 2].copy()
    
    if subsampling == '4:4:4':
        pass
    
    elif subsampling == '4:2:2':
        cb_channel = cb_channel[:, ::2]
        cr_channel = cr_channel[:, ::2]
    
    elif subsampling == '4:2:0':
        # Đảm bảo kích thước chia hết cho 2
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Chiều cao và chiều rộng phải chia hết cho 2 với 4:2:0")
        # Tính trung bình khối 2x2
        cb_channel = cb_channel.reshape(height//2, 2, width//2, 2).mean(axis=(1, 3))
        cr_channel = cr_channel.reshape(height//2, 2, width//2, 2).mean(axis=(1, 3))
    
    else:
        raise ValueError("Kiểu lấy mẫu phụ phải là '4:4:4', '4:2:2', hoặc '4:2:0'")
    
    return y_channel, cb_channel, cr_channel

def apply_chroma_upsampling(channels, subsampling='4:2:0'):
    """
    Khôi phục kích thước gốc cho các kênh Cb và Cr từ dữ liệu đã lấy mẫu phụ.
    
    Parameters:
    -----------
    channels : tuple
        (y_channel, cb_channel, cr_channel) từ apply_chroma_subsampling
    subsampling : str, optional
        Kiểu lấy mẫu phụ ('4:4:4', '4:2:2', '4:2:0'), default='4:2:0'
    
    Returns:
    --------
    ndarray
        Ảnh YCbCr, shape (H, W, 3), dtype=float32, giá trị trong [0, 255]
    
    Raises:
    -------
    ValueError
        Nếu shape đầu vào không đúng, giá trị pixel ngoài [0, 255], hoặc subsampling không hỗ trợ
    """
    y_channel, cb_channel, cr_channel = channels
    height, width = y_channel.shape
    
    if y_channel.ndim != 2 or y_channel.shape != (height, width):
        raise ValueError("y_channel phải có shape (H, W)")
    if cb_channel.max() > 255 or cb_channel.min() < 0 or cr_channel.max() > 255 or cr_channel.min() < 0:
        raise ValueError("Giá trị pixel phải nằm trong [0, 255]")
    
    if subsampling == '4:4:4':
        if cb_channel.shape != (height, width) or cr_channel.shape != (height, width):
            raise ValueError("Shape của Cb, Cr phải là (H, W) với 4:4:4")
    
    elif subsampling == '4:2:2':
        if cb_channel.shape != (height, width//2) or cr_channel.shape != (height, width//2):
            raise ValueError("Shape của Cb, Cr phải là (H, W//2) với 4:2:2")
        cb_channel = np.repeat(cb_channel, 2, axis=1)[:, :width]
        cr_channel = np.repeat(cr_channel, 2, axis=1)[:, :width]
    
    elif subsampling == '4:2:0':
        if cb_channel.shape != (height//2, width//2) or cr_channel.shape != (height//2, width//2):
            raise ValueError("Shape của Cb, Cr phải là (H//2, W//2) với 4:2:0")
        cb_channel = np.repeat(np.repeat(cb_channel, 2, axis=0), 2, axis=1)[:height, :width]
        cr_channel = np.repeat(np.repeat(cr_channel, 2, axis=0), 2, axis=1)[:height, :width]
    
    else:
        raise ValueError("Kiểu lấy mẫu phụ phải là '4:4:4', '4:2:2', hoặc '4:2:0'")
    
    return np.stack((y_channel, cb_channel, cr_channel), axis=2)