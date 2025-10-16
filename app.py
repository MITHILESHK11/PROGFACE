import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime

# ------------------------------
# SETUP SECTION
# ------------------------------

# Directory to save detected face images (you can later sync this to Hugging Face)
SAVE_DIR = "huggingface_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Haar Cascade
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Streamlit Page Configuration
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("🧠 Real-Time Face Detection App")
st.markdown("This app detects faces from your webcam using OpenCV.")

# ------------------------------
# CAMERA INPUT
# ------------------------------
camera_input = st.camera_input("📸 Take a photo")

if camera_input is not None:
    # Convert the image captured by Streamlit to OpenCV format
    file_bytes = np.asarray(bytearray(camera_input.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    if len(faces) > 0:
        st.success("✅ Person Detected")
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_only = gray[y:y + h, x:x + w]

            # Save cropped face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/face_{timestamp}.jpg"
            cv2.imwrite(filename, face_only)
            st.write(f"💾 Saved: {filename}")
    else:
        st.error("❌ No Person Detected")

    # Convert BGR -> RGB for display in Streamlit
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Processed Image", use_container_width=True)

else:
    st.info("📷 Please capture a photo using the camera above to detect a face.")
