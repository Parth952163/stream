import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt')  # Make sure to provide the correct path to your trained model

st.title('YOLOv8 Object Detection App')

st.write("Upload an image to detect objects")

# File uploader allows user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Perform object detection
    results = model(img)

    # Plot results
    st.image(results.render()[0], caption='Detected Image', use_column_width=True)

    # Display the results
    st.write("Detection results:")
    for result in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
        st.write(f"Class: {result[-1]}, Confidence: {result[-2]}, BBox: [{result[0]}, {result[1]}, {result[2]}, {result[3]}]")
