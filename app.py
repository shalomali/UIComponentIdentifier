import streamlit as st
import tensorflow as tf
# Import your newly broken-down modules
from inference import run_ui_inference

# Load your model and classes
model = tf.keras.models.load_model("ui_sketch_model.keras")
CLASS_NAMES = ['button', 'dropdown_menu', 'image', 'switch_enabled', 'text_field']

# Use them inside your UI handling step
uploaded_file = st.file_uploader("Upload Sketch Picture", type=["jpg", "png"])
if uploaded_file is not None:
    # ... handle save path ...
    results = run_ui_inference("saved_image.jpg", model, CLASS_NAMES)