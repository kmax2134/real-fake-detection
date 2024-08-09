import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

# Load the trained models

densenet_model_path = 'densenet121.keras'

try:
    # Load the model within a CustomObjectScope if necessary
    with tf.keras.utils.custom_object_scope({'DenseNet121': tf.keras.applications.DenseNet121}):
        densenet_model = tf.keras.models.load_model(densenet_model_path)

    st.success("DenseNet121 model loaded successfully.")
    st.write(densenet_model.summary())  # Display model summary for verification

except Exception as e:
    st.error(f"Error loading DenseNet121 model: {e}")
    st.stop()  # Stop execution if the model can't be loaded

# Function to preprocess and predict using a specified model
def preprocess_and_predict(image_path, model, preprocess_input, target_size):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Debugging: Log image array shape
        st.write(f"Image array shape: {img_array.shape}")

        # Prediction
        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        st.error(f"Error during preprocessing and prediction: {e}")
        return None

# Function to interpret predictions
def interpret_predictions(predictions):
    if predictions is not None:
        real_prob, fake_prob = predictions[0]
        if real_prob > fake_prob:
            label = "Real"
            confidence = real_prob
        else:
            label = "Fake"
            confidence = fake_prob
        return label, confidence
    else:
        return "Unknown", 0.0

# Streamlit app
st.title("Deepfake Detection using DenseNet121")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure the temporary directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Predict using DenseNet121
    if densenet_model:
        st.subheader("DenseNet121 Prediction")
        densenet_predictions = preprocess_and_predict(
            temp_file_path, 
            densenet_model, 
            densenet_preprocess_input, 
            (224, 224)
        )
        if densenet_predictions is not None:
            densenet_label, densenet_confidence = interpret_predictions(densenet_predictions)
            st.write(f"Prediction: {densenet_label} (Confidence: {densenet_confidence:.2f})")
        else:
            st.error("Failed to make a prediction.")
