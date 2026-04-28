import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Load model
model = tf.keras.models.load_model("model/digit_model.h5")

def preprocess_array(img_array):
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1,28,28,1)
    return img_array

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28,28))
    image = np.array(image) / 255.0
    image = image.reshape(1,28,28,1)
    return image

st.title("🧠 Advanced Digit Recognition System")

# -------------------------------
# 🔹 OPTION 1: DRAW DIGIT
# -------------------------------
st.subheader("✏️ Draw a Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data[:,:,0]
    img = Image.fromarray(img.astype('uint8'))
    img = img.resize((28,28))
    img = np.array(img)
    
    processed = preprocess_array(img)
    prediction = model.predict(processed)
    
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {digit}")
    st.write(f"Confidence: {confidence*100:.2f}%")

# -------------------------------
# 🔹 OPTION 2: UPLOAD IMAGE
# -------------------------------
st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader("Upload digit image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    processed = preprocess_image(image)

    prediction = model.predict(processed)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {digit}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # Graph
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_title("Digit Probabilities")
    st.pyplot(fig)