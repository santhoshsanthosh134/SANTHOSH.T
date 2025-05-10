
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageOps

# Load model and label map
model = tf.keras.models.load_model("handwriting_cnn_model.h5")
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# App title
st.title("Handwritten Digit Recognition")
st.write("Upload an 8x8 grayscale image to recognize the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((8, 8))
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    mean = np.mean(image)
    std = np.std(image)
    image = image.reshape(8, 8, 1)
    mean_channel = np.ones((8, 8, 1)) * mean
    std_channel = np.ones((8, 8, 1)) * std
    image = np.concatenate([image, mean_channel, std_channel], axis=-1)
    return image.reshape(1, 8, 8, 3)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    st.success(f"Predicted Digit: {label_map[predicted_label]}")
