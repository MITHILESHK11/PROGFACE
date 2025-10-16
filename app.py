import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from pymongo import MongoClient
import datetime
import io

# -----------------------
# MongoDB setup
# -----------------------
MONGO_URI = st.secrets["MONGO"]["MONGO_URI"]  # or os.environ["MONGO_URI"] if using GitHub Actions
client = MongoClient(MONGO_URI)
db = client["face_dataset"]
collection = db["faces"]

# -----------------------
# Haar Cascade
# -----------------------
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
(width, height) = (130, 100)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Real-Time Face Capture & Storage")
st.write("Enter your name and allow webcam access to capture face images.")

user_name = st.text_input("Your Name")

if not user_name:
    st.warning("Please enter your name to continue.")
    st.stop()

# -----------------------
# Video Transformer
# -----------------------
class FaceCapture(VideoTransformerBase):
    def __init__(self):
        self.count = 1

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 4)
        
        if len(faces) > 0:
            st.session_state["status"] = "Person Detected"
        else:
            st.session_state["status"] = "No Person Detected"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_only = gray[y:y+h, x:x+w]
            resize_face = cv2.resize(face_only, (width, height))
            
            # Convert to bytes
            _, img_encoded = cv2.imencode(".jpg", resize_face)
            img_bytes = img_encoded.tobytes()
            
            # Save to MongoDB
            collection.insert_one({
                "name": user_name,
                "image_number": self.count,
                "image_data": img_bytes,
                "timestamp": datetime.datetime.utcnow()
            })
            self.count += 1
        
        return img

# -----------------------
# Run webcam
# -----------------------
if "status" not in st.session_state:
    st.session_state["status"] = "No Person Detected"

webrtc_streamer(
    key="face-capture",
    video_transformer_factory=FaceCapture,
    media_stream_constraints={"video": True, "audio": False},
)

st.info(f"Status: {st.session_state['status']}")
st.write("Captured images are automatically saved to MongoDB per user.")
