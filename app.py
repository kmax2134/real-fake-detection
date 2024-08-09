import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import layers, Sequential
from PIL import Image

# Paths to saved models (Update these paths to your actual model files)
model_path_resnet = 'resnet50.h5'
model_path_xception = 'xception.h5'

# Define class labels (Adjust these labels according to your dataset)
class_labels = ['Fake', 'Real']

# Define the ResNet50 model
def get_resnet50_model(input_shape=(128, 128, 3), num_classes=2):
    resnet_base = ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential([
        resnet_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the Xception model using the provided function
def get_xception_model(input_shape=(128, 128, 3), num_classes=2):
    input = tf.keras.Input(shape=input_shape)
    xception_base = Xception(weights="imagenet", include_top=False, input_tensor=input)
    x = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=xception_base.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load the models with weights
try:
    model_resnet = get_resnet50_model()
    model_resnet.load_weights(model_path_resnet)
    st.success("ResNet50 model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the ResNet50 model: {str(e)}")

try:
    model_xception = get_xception_model()
    model_xception.load_weights(model_path_xception)
    st.success("Xception model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load the Xception model: {str(e)}")

# Function to preprocess the image
def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)  # Resize image to target size
    img_array = keras_image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image array
    return img_array

# Function to make predictions
def predict_image(img, model):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence

# Streamlit UI
st.title("Real vs Fake Image Detection using ResNet50 & Xception")
st.write("Upload an image to classify it as Real or Fake using ResNet50 or Xception.")

# Select model
model_choice = st.selectbox("Choose a model", ["ResNet50", "Xception"])

# Single image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Function to display the result
def display_result(label, confidence):
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

# Single image prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if model_choice == "ResNet50":
        with st.spinner('Classifying with ResNet50...'):
            label, confidence = predict_image(image, model_resnet)
            display_result(label, confidence)
    elif model_choice == "Xception":
        with st.spinner('Classifying with Xception...'):
            label, confidence = predict_image(image, model_xception)
            display_result(label, confidence)
