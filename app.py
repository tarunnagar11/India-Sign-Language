import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# DEBUG (REMOVE AFTER TESTING)
# -----------------------------
st.write("Files in directory:", os.listdir())

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.keras", compile=False)

model = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = list("abcdefghijklmnopqrstuvwxyz")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ISL Recognition", layout="centered")

st.title("🤟 Indian Sign Language Recognition System")
st.write("Upload image or capture photo")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence

# -----------------------------
# SIDEBAR OPTION
# -----------------------------
option = st.sidebar.radio(
    "Choose Input Method",
    ["Upload Image", "Camera Capture"]
)

# =========================================================
# IMAGE UPLOAD
# =========================================================
if option == "Upload Image":
    st.header("📁 Upload Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        pred_class, conf = predict_image(image)

        st.success(f"Predicted: {classes[pred_class].upper()}")
        st.info(f"Confidence: {conf*100:.2f}%")

# =========================================================
# CAMERA CAPTURE
# =========================================================
elif option == "Camera Capture":
    st.header("📷 Capture from Camera")

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        pred_class, conf = predict_image(image)

        st.success(f"Predicted: {classes[pred_class].upper()}")
        st.info(f"Confidence: {conf*100:.2f}%")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with TensorFlow + Streamlit 🚀")
