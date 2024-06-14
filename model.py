import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# Load YOLO Pretrained model
model = YOLO('best.pt')

# Streamlit app
st.title('YOLO Object Detection')
st.write('Upload an image to detect objects.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Perform inference on the provided image(s)
    results = model.predict(source=image, 
                            imgsz=640,
                            verbose=False,
                            conf=0.5)   # Confidence threshold: 50% (only detections above 50% confidence will be considered)

    # Annotate and convert image to numpy array
    sample_image = results[0].plot(line_width=2)

    # Convert the color of the image from BGR to RGB for correct color representation in matplotlib
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(sample_image, caption='Detected Objects', use_column_width=True)
