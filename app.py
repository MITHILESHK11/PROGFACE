import streamlit as st
import cv2
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import datetime

st.set_page_config(page_title="Face Detection Cloud Webcam", layout="centered")
st.title("🧠 Real-Time Face Detection (Cloud Compatible)")

# Directory to save detected faces
SAVE_DIR = "huggingface_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Haar cascade from same directory
haar_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_file):
    st.error("Haar cascade XML file not found in the same directory as app.py")
    st.stop()
face_cascade = cv2.CascadeClassifier(haar_file)

# ---------------------------
# Video transformer
# ---------------------------
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        if len(faces) > 0:
            cv2.putText(img, "✅ Person Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_only = gray[y:y+h, x:x+w]

                # Save each face detected
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = f"{SAVE_DIR}/face_{timestamp}.jpg"
                cv2.imwrite(filename, face_only)
        else:
            cv2.putText(img, "❌ No Person Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,
