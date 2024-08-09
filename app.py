import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Path to the saved models (Update these paths to your actual model files)
model_path_resnet = 'resnet50.h5'  # Replace with your ResNet50 model path
model_path_xception = 'xception.h5'  # Replace with your Xception model path

# Define class labels (Adjust these labels according to your dataset)
class_labels = ['Fake', 'Real']

# Function to preprocess the image for ResNet50
def preprocess_image_resnet(img):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image array
    return img_array

# Function to preprocess the image for Xception
def preprocess_image_xception(img):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.xception.preprocess_input(img_array)  # Preprocess for Xception
    return img_array

# Load the entire models
try:
    model_resnet = load_model(model_path_resnet)
    st.success("ResNet50 model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the ResNet50 model: {str(e)}")

try:
    model_xception = load_model(model_path_xception)
    st.success("Xception model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the Xception model: {str(e)}")

# Function to make predictions
def predict_image(img, model, preprocess_func):
    img_array = preprocess_func(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

# Streamlit UI
st.title("Real vs Fake Image Detection using ResNet50 and Xception")
st.write("Upload an image to classify it as Real or Fake.")

# Model selection
model_choice = st.selectbox("Choose a model", ("ResNet50", "Xception"))

# Single image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Multiple images uploader
uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Function to display the result
def display_result(label, confidence):
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

# Single image prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        if model_choice == "ResNet50":
            label, confidence = predict_image(image, model_resnet, preprocess_image_resnet)
        else:
            label, confidence = predict_image(image, model_xception, preprocess_image_xception)
        display_result(label, confidence)

# Multiple images prediction
if uploaded_files:
    st.write("Batch Prediction Results:")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f'Image: {file.name}', use_column_width=True)
        
        with st.spinner(f'Classifying {file.name}...'):
            if model_choice == "ResNet50":
                label, confidence = predict_image(image, model_resnet, preprocess_image_resnet)
            else:
                label, confidence = predict_image(image, model_xception, preprocess_image_xception)
            display_result(label, confidence)

