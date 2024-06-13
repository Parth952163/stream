import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Load YOLO Pretrained model
model_path = r"C:\Users\ASUS\Desktop\Parth\AIML\Project - 4\Model\best.pt"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please check the file path.")
    st.stop()

try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

def detect_objects(image):
    # Perform inference on the provided image
    results = model.predict(source=image, 
                            imgsz=640,
                            verbose=False,
                            conf=0.5)  # Confidence threshold: 50%

    # Annotate and convert image to numpy array
    annotated_image = results[0].plot(line_width=2)
    
    # Convert the color of the image from BGR to RGB for correct color representation
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image, results

# Streamlit app
st.title("YOLOv8 Object Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert the uploaded file to an image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting objects...")

        # Convert the image to an array format for object detection
        image_np = np.array(image)

        # Perform object detection
        output_image, results = detect_objects(image_np)

        # Convert the output image back to PIL format
        output_image = Image.fromarray(output_image)

        # Display the results
        st.image(output_image, caption='Detected Image.', use_column_width=True)

        # Optionally, display detection details
        for detection in results[0].boxes:
            st.write(f"Class: {detection.cls}, Confidence: {detection.conf:.2f}")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.write("Please upload an image to proceed.")
