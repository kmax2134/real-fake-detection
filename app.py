from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50, DenseNet121
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Paths to saved model files
model_path_xception = r'C:/Users/yogesh/Documents/New folder (2)/face-forgery-detection/xception.h5'
model_path_resnet = r'C:/Users/yogesh/Documents/New folder (2)/face-forgery-detection/resnet50.h5'
densenet_model_path = r'C:/Users/yogesh/Documents/New folder (2)/face-forgery-detection/densenet121.keras'

# Define class labels
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


# Function to define the ResNet50 model
def get_resnet50_model(input_shape=(128, 128, 3), num_classes=2):
    input = tf.keras.Input(shape=input_shape)
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=input)
    x = tf.keras.layers.GlobalAveragePooling2D()(resnet_base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=resnet_base.input, outputs=output)
    return model


# Load the models with weights
try:
    model_xception = get_xception_model()
    model_xception.load_weights(model_path_xception)

    model_resnet = get_resnet50_model()
    model_resnet.load_weights(model_path_resnet)

    densenet_model = tf.keras.models.load_model(densenet_model_path)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model_xception, model_resnet, densenet_model = None, None, None


# Function to preprocess the image
def preprocess_image(img, model_name):
    if model_name == 'DenseNet121':
        img = img.resize((224, 224))  # Resize image to 224x224 pixels for DenseNet121
    else:
        img = img.resize((128, 128))  # Resize image to 128x128 pixels for Xception and ResNet50

    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if model_name == 'Xception':
        return preprocess_input_xception(img_array)
    elif model_name == 'ResNet50':
        return preprocess_input_resnet(img_array)
    elif model_name == 'DenseNet121':
        return preprocess_input_densenet(img_array)


# Function to make predictions
def predict_image(img, model, model_name):
    img_array = preprocess_image(img, model_name)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class]
    return predicted_label, confidence


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and model_xception and model_resnet and densenet_model:
            img = Image.open(file)

            # Make a prediction with all three models
            label_xception, confidence_xception = predict_image(img, model_xception, 'Xception')
            label_resnet, confidence_resnet = predict_image(img, model_resnet, 'ResNet50')
            label_densenet, confidence_densenet = predict_image(img, densenet_model, 'DenseNet121')

            # Render the results in the template
            return render_template('result.html',
                                   label_xception=label_xception, confidence_xception=confidence_xception,
                                   label_resnet=label_resnet, confidence_resnet=confidence_resnet,
                                   label_densenet=label_densenet, confidence_densenet=confidence_densenet)
        else:
            return "Models not loaded correctly.", 500

    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
