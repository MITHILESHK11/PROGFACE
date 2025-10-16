import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from pymongo import MongoClient
import datetime

# -----------------------
# MongoDB setup
# -----------------------
MONGO_URI = "mongodb+srv://User:abc12345678@facedataset.hlmdvpa.mongodb.net/?retryWrites=true&w=majority&appName=FACEDATASET"
client = MongoClient(MONGO_URI)
db = client["face_dataset"]
collection = db["faces"]

# -----------------------
# Haar Cascade
# -----------------------
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
(width, height) = (130, 100)
MAX_IMAGES = 25  # Max images per user

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 Real-Time Face Capture & Storage")
st.write("Enter your name, allow webcam access, and capture face images.")

user_name = st.text_input("Your Name")
show_images = st.checkbox("Show all saved images for this user")

if not user_name:
    st.warning("Please enter your name to continue.")
    st.stop()

# -----------------------
# Video Transformer
# -----------------------
class FaceCapture(VideoTransformerBase):
    def __init__(self):
        # Count images already saved
        self.count = collection.count_documents({"name": user_name}) + 1
        self.done = False

    def transform(self, frame):
        if self.count > MAX_IMAGES:
            st.session_state["status"] = f"✅ Max {MAX_IMAGES} images reached for {user_name}"
            self.done = True
            return frame.to_ndarray(format="bgr24")

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

            if self.count <= MAX_IMAGES:
                _, img_encoded = cv2.imencode(".jpg", resize_face)
                img_bytes = img_encoded.tobytes()

                # Save to MongoDB under user name
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

webrtc_ctx = webrtc_streamer(
    key="face-capture",
    video_transformer_factory=FaceCapture,
    media_stream_constraints={"video": True, "audio": False},
    mode=WebRtcMode.SENDRECV,
)

# Stop the feed automatically if max images reached
if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.done:
    webrtc_ctx.stop()

st.info(f"Status: {st.session_state['status']}")

# -----------------------
# Show all saved images
# -----------------------
if show_images:
    st.write(f"All saved images for {user_name}:")
    saved_images = list(collection.find({"name": user_name}).sort("image_number", 1))

    if saved_images:
        cols = st.columns(5)
        for i, img_doc in enumerate(saved_images):
            img_array = np.frombuffer(img_doc["image_data"], dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            cols[i % 5].image(img, use_column_width=True, caption=f"{img_doc['image_number']}")
    else:
        st.write("No images saved yet for this user.")
