import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# -----------------------
# Streamlit page config
# -----------------------
st.set_page_config(page_title="Webcam Face Capture to Google Drive", layout="centered")
st.title("🧠 Face Capture & Save to Google Drive")

# -----------------------
# Google Drive setup
# -----------------------
st.write("⚠️ Make sure you have your credentials.json for Google Drive API in the repo or upload it.")
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens auth flow in browser
drive = GoogleDrive(gauth)

# -----------------------
# User input
# -----------------------
name = st.text_input("Enter your name:")
if not name:
    st.warning("Please enter a name to start capturing.")
    st.stop()

# Create folder in Google Drive
folders = drive.ListFile({'q': "title='FaceDataset' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
if folders:
    folder_id = folders[0]['id']
else:
    folder_metadata = {'title': 'FaceDataset', 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    folder_id = folder['id']

# Create subfolder for the user
user_folder_metadata = {'title': name, 'parents':[{'id': folder_id}], 'mimeType':'application/vnd.google-apps.folder'}
user_folder = drive.CreateFile(user_folder_metadata)
user_folder.Upload()
user_folder_id = user_folder['id']

st.success(f"Folder for {name} created in Google Drive.")

# -----------------------
# Face detection setup
# -----------------------
haar_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_file):
    st.error("Haar cascade XML file not found in the repo.")
    st.stop()
face_cascade = cv2.CascadeClassifier(haar_file)

width, height = 130, 100

# -----------------------
# Video transformer
# -----------------------
class FaceCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.count = 1

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        if len(faces) > 0:
            cv2.putText(img, "✅ Person Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
                face_only = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_only, (width, height))

                # Save to temporary file
                temp_filename = f"face_{self.count}.jpg"
                cv2.imwrite(temp_filename, face_resized)

                # Upload to Google Drive
                gfile = drive.CreateFile({'title': f"{self.count}.jpg", 'parents':[{'id': user_folder_id}]})
                gfile.SetContentFile(temp_filename)
                gfile.Upload()
                os.remove(temp_filename)

                self.count += 1
        else:
            cv2.putText(img, "❌ No Person Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return img

# -----------------------
# Start webcam stream
# -----------------------
webrtc_streamer(
    key="face-capture",
    video_transformer_factory=FaceCaptureTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
