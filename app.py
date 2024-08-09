import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np
import os

# Define model architectures
def build_resnet50_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(2, activation='softmax')
    ])
    return model

def build_xception_model():
    input_shape = (128, 128, 3)
    num_classes = 2
    input = tf.keras.Input(shape=input_shape)
    xception = tf.keras.applications.Xception(weights=None, include_top=False, input_tensor=input)
    x = tf.keras.layers.GlobalAveragePooling2D()(xception.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(xception.input, output)
    return model

# Load model weights
def load_model_weights(model, weights_path):
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

# Load models
def load_models():
    resnet_model = build_resnet50_model()
    xception_model = build_xception_model()
    
    resnet_weights_path = 'resnet50.h5'
    xception_weights_path = 'xception.h5'
    
    load_model_weights(resnet_model, resnet_weights_path)
    load_model_weights(xception_model, xception_weights_path)
    
    return resnet_model, xception_model

# Load the models
resnet_model, xception_model = load_models()

# Set up Streamlit app
st.title('Deepfake Detection')

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((128, 128))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    
    # Prediction
    st.write("Predicting using ResNet50 model...")
    resnet_prediction = resnet_model.predict(image_array)
    st.write(f"ResNet50 Prediction: {resnet_prediction}")
    
    st.write("Predicting using Xception model...")
    xception_prediction = xception_model.predict(image_array)
    st.write(f"Xception Prediction: {xception_prediction}")

    # Display image
    st.image(image, caption='Uploaded Image', use_column_width=True)
