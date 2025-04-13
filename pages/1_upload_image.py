import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

def app():
    st.title("Step 1: Upload Image")
    
    # Create a session state to store the image data
    if 'image' not in st.session_state:
        st.session_state.image = None
        st.session_state.filename = None
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Save to session state
        st.session_state.image = image
        st.session_state.filename = uploaded_file.name
        
        # Save the image for future steps
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Save the original image
        Image.fromarray(image).save(os.path.join(temp_dir, "original.jpg"))
        
        # Display the image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Display image information
        st.write(f"Image shape: {image.shape}")
        st.write(f"Image size: {image.size} pixels")
        
        # Add a success message
        st.success("Image uploaded successfully! You can proceed to the next step.")
    else:
        st.info("Please upload an image to begin the JPEG compression visualization.")

if __name__ == "__main__":
    app()