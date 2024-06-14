import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def load_image(image_file):
    img = Image.open(image_file)
    return img

def detect_objects(image, model):
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Perform the detection
    detections = model(input_tensor)

    return detections

def draw_detections(image, detections):
    height, width, _ = image.shape
    for i in range(int(detections['num_detections'])):
        score = detections['detection_scores'][0][i].numpy()
        if score > 0.5:  # Confidence threshold
            bbox = detections['detection_boxes'][0][i].numpy()
            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    return image

def main():
    st.title("Object Detection App")
    
    model_path = st.text_input("Enter the path to the model directory:", "")
    
    if model_path:
        model = load_model(model_path)
        st.success("Model loaded successfully!")

        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            image = load_image(image_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            detections = detect_objects(image, model)
            
            image_np = np.array(image)
            detected_image = draw_detections(image_np, detections)
            st.image(detected_image, caption='Detected Image', use_column_width=True)

if __name__ == "__main__":
    main()
