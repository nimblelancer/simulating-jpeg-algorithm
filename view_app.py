import streamlit as st

# Cấu hình trang
# st.set_page_config(page_title="JPEG Compression Simulator", layout="wide")

# Tùy chỉnh sidebar
st.sidebar.title("📷 JPEG Compression Simulator")
st.sidebar.markdown("""
### Chào mừng bạn đến với mô phỏng nén JPEG!
Chọn một bước trong quá trình để bắt đầu:
""")

# Thêm danh sách các bước với tên dễ hiểu hơn
st.sidebar.markdown("### Các bước:")
st.sidebar.markdown("1. **Tải ảnh lên & Xem trước**")
st.sidebar.markdown("2. **Mô phỏng DCT (Biến đổi Cosine rời rạc)**")
st.sidebar.markdown("3. **Lượng tử hóa**")
st.sidebar.markdown("4. **Mô phỏng ZigZag**")
st.sidebar.markdown("5. **Mã hóa Huffman**")
st.sidebar.markdown("6. **Giải mã ảnh**")
st.sidebar.markdown("7. **So sánh ảnh gốc & ảnh nén**")

# Thêm phân cách
st.sidebar.markdown("---")
st.sidebar.markdown("Cập nhật lần cuối: **2025**")

# Tiêu đề chính của trang
st.title("📷 JPEG Compression Simulator")

# Giới thiệu về ứng dụng
st.markdown("""
### Mô phỏng toàn bộ quá trình nén ảnh JPEG:
- Từng bước thực hiện sẽ được hiển thị riêng biệt để bạn dễ theo dõi.
- Ảnh, ma trận, cây Huffman, và thông số đều được minh họa trực quan.
""")
