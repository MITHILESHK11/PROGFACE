import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("🧠 Real-Time Face Detection (Cloud Compatible)")

# Directory to save detected faces
SAVE_DIR = "huggingface_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# Haar cascade from same directory
haar_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_file):
    st.error("Haar cascade XML file not found in the same directory as app.py")
    st.stop()
face_cascade = cv2.CascadeClassifier(haar_file)

# -------------------------------
# CAMERA INPUT
# -------------------------------
st.write("📸 Take a photo using your browser camera")
camera_input = st.camera_input("Camera input")

# Optional fallback upload
st.write("Or upload a photo if camera doesn't work")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Choose the image to process
image_file = camera_input if camera_input else uploaded_file

if image_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    if len(faces) > 0:
        st.success("✅ Person Detected")
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_only = gray[y:y + h, x:x + w]

            # Save detected face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/face_{timestamp}.jpg"
            cv2.imwrite(filename, face_only)
            st.write(f"💾 Saved: {filename}")
    else:
        st.error("❌ No Person Detected")

    # Convert BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Processed Image", use_column_width=True)
else:
    st.info("Please take a photo using the camera above or upload an image.")
