import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Paths to saved model files (Update these paths to your actual model files)
model_path_xception = 'xception.h5'  # Replace with your Xception model path
model_path_densenet = 'densenet121.keras'  # Replace with your DenseNet121 model path

# Define class labels (Adjust these labels according to your dataset)
class_labels = ['Fake', 'Real']

# Function to define the Xception model
def get_xception_model(input_shape=(128, 128, 3), num_classes=2):
    input = tf.keras.Input(shape=input_shape)
    xception_base = Xception(weights='imagenet', include_top=False, input_tensor=input)
    x = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=xception_base.input, outputs=output)
    return model

# Load the models with weights
try:
    model_xception = get_xception_model()
    model_xception.load_weights(model_path_xception)

    model_densenet = tf.keras.models.load_model(model_path_densenet)

    st.success("Models loaded successfully.")
except Exception as e:
    st.error(f"Failed to load models: {str(e)}")

# Function to preprocess the image
def preprocess_image(img, model_name):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = keras_image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if model_name == 'Xception':
        return preprocess_input_xception(img_array)
    elif model_name == 'DenseNet121':
        return densenet_preprocess_input(img_array)

# Function to make predictions
def predict_image(img, model, model_name):
    img_array = preprocess_image(img, model_name)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

# Streamlit UI
st.title("Real vs Fake Image Detection")
st.write("Upload an image to classify it as Real or Fake.")

# Model selection
model_option = st.selectbox(
    "Select the model to use:",
    ("Xception", "DenseNet121")
)

# Load the selected model
selected_model = model_xception if model_option == "Xception" else model_densenet

# Confidence threshold slider
confidence_threshold = st.slider(
    "Confidence threshold:", 0.0, 1.0, 0.5, 0.01
)

# Single image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Multiple images uploader
uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Function to display the result
def display_result(label, confidence):
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
    if confidence < confidence_threshold:
        st.write(f"⚠️ Confidence is below the threshold of {confidence_threshold * 100:.2f}%.")

# Single image prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        label, confidence = predict_image(image, selected_model, model_option)
        display_result(label, confidence)

# Multiple images prediction
if uploaded_files:
    st.write("Batch Prediction Results:")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f'Image: {file.name}', use_column_width=True)
        
        with st.spinner(f'Classifying {file.name}...'):
            label, confidence = predict_image(image, selected_model, model_option)
            display_result(label, confidence)
