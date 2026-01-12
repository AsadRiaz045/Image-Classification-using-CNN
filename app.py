import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define image dimensions (these should match your training setup)
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Load the trained model
model_path = 'cnn_model.keras'
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define class names (make sure these match your training class names order)
# Based on previous execution, class_names were ['Cat', 'Dog']
class_names = ['Cat', 'Dog']

def preprocess_image(image):
    # Resize the image
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit App Layout
st.set_page_config(layout="wide", page_title="CNN Image Classifier", page_icon=":camera:")
st.title("CNN Image Classifier for Cat and Dog")

st.write("Upload an image (Cat or Dog) for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        score = tf.nn.softmax(predictions[0])

        predicted_class_index = np.argmax(score)
        predicted_class_name = class_names[predicted_class_index]
        prediction_probability = np.max(score) * 100

        st.subheader("Prediction:")
        st.success(f"The image is a **{predicted_class_name}** with **{prediction_probability:.2f}%** confidence.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Awaiting image upload...")
