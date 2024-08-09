import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# Path to the saved models (Update these paths to your actual model files)
model_path_resnet = 'resnet50.h5'  # Replace with your ResNet50 model path
model_path_xception = 'xception.h5'  # Replace with your Xception model path

# Define class labels (Adjust these labels according to your dataset)
class_labels = ['Fake', 'Real']

# Function to define the ResNet50 model
def get_resnet50_model(input_shape=(128, 128, 3), num_classes=2):
    input = tf.keras.Input(shape=input_shape)
    resnet_base = ResNet50(weights=None, include_top=False, input_tensor=input)
    x = tf.keras.layers.GlobalAveragePooling2D()(resnet_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=resnet_base.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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

# Function to preprocess the image for ResNet50
def preprocess_image_resnet(img):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = keras_image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image array
    return img_array

# Function to preprocess the image for Xception
def preprocess_image_xception(img):
    img = img.resize((128, 128))  # Resize image to 128x128 pixels
    img_array = keras_image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input_xception(img_array)

# Function to make predictions with ResNet50
def predict_image_resnet(img, model):
    img_array = preprocess_image_resnet(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    return predicted_class, confidence

# Function to make predictions with Xception
def predict_image_xception(img, model):
    img_array = preprocess_image_xception(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    return predicted_class, confidence

# Streamlit UI
st.title("Real vs Fake Image Detection using ResNet50 and Xception")
st.write("Upload an image to classify it as Real or Fake.")

# Single image uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Multiple images uploader
uploaded_files = st.file_uploader("Choose multiple images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# Function to display the result
def display_result(label, confidence, model_name):
    st.write(f"{model_name} Prediction: **{label}**")
    st.write(f"{model_name} Confidence: **{confidence * 100:.2f}%**")

# Function to display the final result
def display_final_result(label, confidence):
    st.write(f"**Final Prediction (Average): {label}**")
    st.write(f"**Final Confidence (Average): {confidence * 100:.2f}%**")

# Single image prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Classifying...'):
        # Predict with ResNet50
        resnet_class, resnet_confidence = predict_image_resnet(image, model_resnet)
        resnet_label = class_labels[resnet_class]
        
        # Predict with Xception
        xception_class, xception_confidence = predict_image_xception(image, model_xception)
        xception_label = class_labels[xception_class]
        
        # Display individual model results
        display_result(resnet_label, resnet_confidence, "ResNet50")
        display_result(xception_label, xception_confidence, "Xception")
        
        # Calculate average confidence and prediction
        avg_confidence = (resnet_confidence + xception_confidence) / 2
        
        # Determine the final prediction based on the average confidence
        if resnet_class == xception_class:
            final_predicted_label = resnet_label
        else:
            # Choose the label with the higher confidence score
            final_predicted_label = resnet_label if resnet_confidence > xception_confidence else xception_label
        
        display_final_result(final_predicted_label, avg_confidence)

# Multiple images prediction
if uploaded_files:
    st.write("Batch Prediction Results:")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f'Image: {file.name}', use_column_width=True)
        
        with st.spinner(f'Classifying {file.name}...'):
            # Predict with ResNet50
            resnet_class, resnet_confidence = predict_image_resnet(image, model_resnet)
            resnet_label = class_labels[resnet_class]
            
            # Predict with Xception
            xception_class, xception_confidence = predict_image_xception(image, model_xception)
            xception_label = class_labels[xception_class]
            
            # Display individual model results
            display_result(resnet_label, resnet_confidence, "ResNet50")
            display_result(xception_label, xception_confidence, "Xception")
            
            # Calculate average confidence and prediction
            avg_confidence = (resnet_confidence + xception_confidence) / 2
            
            # Determine the final prediction based on the average confidence
            if resnet_class == xception_class:
                final_predicted_label = resnet_label
            else:
                # Choose the label with the higher confidence score
                final_predicted_label = resnet_label if resnet_confidence > xception_confidence else xception_label
            
            display_final_result(final_predicted_label, avg_confidence)
