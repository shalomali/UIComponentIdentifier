import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# 1. Import the same layout blueprint function you used in Colab
from train_blueprint import build_sketch_classifier  
from inference import run_ui_inference

st.title("Hand-Drawn UI Component Detector")

# 2. Build the empty architecture shell right inside the app
@st.cache_resource
def load_my_model():
    # Build a clean, uninitialized MobileNetV2 shell (5 target classes)
    empty_model = build_sketch_classifier(input_shape=(64, 64, 1), num_classes=5)
    
    # Inject the raw numerical weights into the shell
    empty_model.load_weights("ui_sketch_model_weights.weights.h5")
    return empty_model

model = load_my_model()
CLASS_NAMES = ['button', 'dropdown_menu', 'image', 'switch_enabled', 'text_field']

# --- Rest of your standard image upload and execution logic ---
uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    temp_path = "temp_uploaded_page.jpg"
    cv2.imwrite(temp_path, opencv_img)
    
    with st.spinner("Analyzing components..."):
        results = run_ui_inference(temp_path, model, CLASS_NAMES)
        
    st.subheader("Detection Results")
    for item in results:
        st.write(f"Detected **{item['class']}** with {item['confidence']:.2f}% confidence")
