import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from pymongo import MongoClient
import datetime

# ----------------------------
# MongoDB Atlas Setup
# ----------------------------
MONGO_URI = st.secrets["MONGO_URI"]  # We'll store MongoDB URI in Streamlit Secrets
client = MongoClient(MONGO_URI)
db = client["face_dataset"]
collection = db["faces"]

# ----------------------------
# App UI
# ----------------------------
st.title("📸 Real-Time Face Capture")
name = st.text_input("Enter your name:")

if not name:
    st.warning("Please enter your name to start capturing faces.")
    st.stop()

st.write(f"Hello, **{name}**! Position your face in front of the webcam.")

# ----------------------------
# Haar Cascade
# ----------------------------
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
WIDTH, HEIGHT = 130, 100
MAX_IMAGES = 10  # Max images per session

# ----------------------------
# Webcam Streamer
# ----------------------------
class FaceCapture(VideoTransformerBase):
    def __init__(self):
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 4)

        if len(faces) > 0:
            st.session_state["message"] = "✅ Person Detected"
        else:
            st.session_state["message"] = "❌ No Person Detected"

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save only a limited number of images per session
            if self.count < MAX_IMAGES:
                face_only = gray[y:y + h, x:x + w]
                resized = cv2.resize(face_only, (WIDTH, HEIGHT))
                # Convert to bytes
                _, img_encoded = cv2.imencode(".jpg", resized)
                img_bytes = img_encoded.tobytes()
                # Insert into MongoDB
                collection.insert_one({
                    "name": name,
                    "image_number": self.count + 1,
                    "image_data": img_bytes,
                    "timestamp": datetime.datetime.utcnow()
                })
                self.count += 1

        return img

if "message" not in st.session_state:
    st.session_state["message"] = ""

webrtc_ctx = webrtc_streamer(
    key="face-capture",
    video_transformer_factory=FaceCapture,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

st.info(st.session_state["message"])
