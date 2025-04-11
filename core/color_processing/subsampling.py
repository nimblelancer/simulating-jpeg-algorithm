import numpy as np

def apply_chroma_subsampling(ycbcr_image, subsampling='4:2:0'):
    """
    Áp dụng lấy mẫu phụ cho các kênh màu Cb và Cr trong ảnh YCbCr.
    
    Parameters:
    -----------
    ycbcr_image : ndarray
        Ảnh trong không gian màu YCbCr với shape (height, width, 3)
    subsampling : str, optional
        Kiểu lấy mẫu phụ ('4:4:4', '4:2:2', '4:2:0'), default='4:2:0'
    
    Returns:
    --------
    ndarray
        Ảnh sau khi áp dụng lấy mẫu phụ
    """
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    result = ycbcr_image.copy()
    
    # Lấy chiều cao và chiều rộng của ảnh
    height, width = ycbcr_image.shape[:2]
    
    # Tách các kênh màu
    y_channel = result[:, :, 0]
    cb_channel = result[:, :, 1]
    cr_channel = result[:, :, 2]
    
    # Áp dụng lấy mẫu phụ tùy thuộc vào loại được chọn
    if subsampling == '4:4:4':
        # Không thực hiện lấy mẫu phụ, giữ nguyên độ phân giải
        pass
    
    elif subsampling == '4:2:2':
        # Giảm độ phân giải theo chiều ngang (chỉ lấy cột chẵn)
        cb_downsampled = cb_channel[:, ::2]
        cr_downsampled = cr_channel[:, ::2]
        
        # Khôi phục kích thước bằng cách nhân đôi các cột
        cb_channel = np.repeat(cb_downsampled, 2, axis=1)[:, :width]
        cr_channel = np.repeat(cr_downsampled, 2, axis=1)[:, :width]
        
    elif subsampling == '4:2:0':
        # Giảm độ phân giải theo cả chiều ngang và chiều dọc (lấy mẫu mỗi 2x2)
        cb_downsampled = cb_channel[::2, ::2]
        cr_downsampled = cr_channel[::2, ::2]
        
        # Khôi phục kích thước bằng cách nhân đôi cả hàng và cột
        cb_upsampled = np.repeat(cb_downsampled, 2, axis=0)
        cr_upsampled = np.repeat(cr_downsampled, 2, axis=0)
        
        cb_channel = np.repeat(cb_upsampled, 2, axis=1)[:height, :width]
        cr_channel = np.repeat(cr_upsampled, 2, axis=1)[:height, :width]
    
    else:
        raise ValueError(f"Kiểu lấy mẫu phụ '{subsampling}' không được hỗ trợ. Sử dụng '4:4:4', '4:2:2', hoặc '4:2:0'.")
    
    # Gán lại các kênh màu đã xử lý
    result[:, :, 1] = cb_channel
    result[:, :, 2] = cr_channel
    
    return result

def apply_chroma_upsampling(subsampled_image, subsampling='4:2:0'):
    """
    Khôi phục từ lấy mẫu phụ cho các kênh màu Cb và Cr trong quá trình giải nén.
    Trong thực tế, function này là không cần thiết nếu chúng ta đã lưu trữ kết quả đã upsampled
    trong apply_chroma_subsampling, nhưng được giữ cho đầy đủ pipeline.
    
    Parameters:
    -----------
    subsampled_image : ndarray
        Ảnh đã áp dụng lấy mẫu phụ
    subsampling : str, optional
        Kiểu lấy mẫu phụ ('4:4:4', '4:2:2', '4:2:0'), default='4:2:0'
    
    Returns:
    --------
    ndarray
        Ảnh đã khôi phục
    """
    # Trong pipeline JPEG thực tế, thông thường chúng ta lưu trữ phiên bản downsampled
    # và thực hiện upsampling thực sự trong quá trình giải nén.
    # Vì chúng ta đã thực hiện upsampling khi lấy mẫu, chỉ cần trả về ảnh như nó đã có.
    
    return subsampled_image.copy()