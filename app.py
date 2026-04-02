import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# FIX INPUT LAYER (MAIN SOLUTION)
# -----------------------------
from tensorflow.keras.layers import InputLayer

class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        super().__init__(*args, **kwargs)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "model.h5",
            compile=False,
            custom_objects={"InputLayer": FixedInputLayer}
        )
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
# PREDICTION FUNCTION
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

    if pred_class is not None:
        st.success(f"Prediction: {classes[pred_class]}")
        st.info(f"Confidence: {conf*100:.2f}%")
    else:
        st.error("⚠️ Model not loaded. Please check file.")
