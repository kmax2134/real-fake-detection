import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the trained model
model_path = 'resnet50.h5'  # Update with your model path
model = load_model(model_path)

st.title("Deepfake Detection with ResNet50")
st.write("Upload an image to predict if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    prediction_label = "Fake" if prediction[0][0] > 0.5 else "Real"

    # Display the result
    st.write(f"The model predicts this image is: **{prediction_label}**")
    st.write(f"Confidence score: {prediction[0][0]:.4f}")

# Function to display examples
def display_examples():
    examples_path = 'path/to/examples/'  # Update with your examples path
    example_images = [os.path.join(examples_path, fname) for fname in os.listdir(examples_path) if fname.endswith(('jpg', 'jpeg', 'png'))]
    
    st.write("Example Images:")
    for example_img_path in example_images:
        example_img = image.load_img(example_img_path, target_size=(224, 224))
        st.image(example_img, caption=os.path.basename(example_img_path), use_column_width=True)

# Button to show examples
if st.button("Show Examples"):
    display_examples()
