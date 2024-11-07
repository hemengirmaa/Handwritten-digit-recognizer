

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model saved in the new '.keras' format
model = tf.keras.models.load_model('mnist_model.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert image to grayscale and resize to 28x28
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))

    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(image) / 255.0

    # Reshape the array to fit the model input
    return image_array.reshape(1, 28, 28)

# Streamlit UI
st.title("Handwritten Digit Recognizer 🎲")
st.write("Upload a digit image (0-9) to predict it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Display predicted digit
    predicted_digit = np.argmax(prediction)
    st.write(f"Predicted Digit: {predicted_digit}")


