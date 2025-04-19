import streamlit as st
from PIL import Image
st.sidebar.title("Sidebar Tiêu đề")
st.title("1️⃣ Upload Image")

st.write("### 📤 Tải ảnh lên")
uploaded_file = st.file_uploader("Chọn một ảnh JPEG hoặc PNG", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)  # Giữ nguyên ảnh gốc, không chuyển sang grayscale
    image.save("assets/images/test/original.png")  # Lưu lại cho các bước sau dùng

    # 👉 Điều chỉnh chiều rộng ảnh hiển thị (ví dụ: 500px)
    st.image(image, caption="Ảnh đã upload", width=500)

    st.write("### 🧾 Thông tin ảnh")
    st.write(f"**Kích thước:** {image.width} x {image.height} pixels")
    st.write(f"**Định dạng:** {uploaded_file.type}")
else:
    st.info("Chưa có ảnh nào được upload.")
