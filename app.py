import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# LOAD MODEL (FIXED)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("final_model.h5", compile=False)
        st.success("✅ Model Loaded Successfully")
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {e}")
        return None

model = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -----------------------------
# PREDICTION FUNCTION (SAFE)
# -----------------------------
def predict_image(image):
    if model is None:
        return None, None

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return predicted_class, confidence

# -----------------------------
# UI
# -----------------------------
st.title("🤟 Indian Sign Language Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    pred_class, conf = predict_image(image)

    # -----------------------------
    # SAFE OUTPUT
    # -----------------------------
    if pred_class is not None:
        st.success(f"Prediction: {classes[pred_class]}")
        st.info(f"Confidence: {conf*100:.2f}%")
    else:
        st.error("⚠️ Model not loaded. Please check file.")
