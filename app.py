import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tflite_runtime.interpreter as tflite

# -----------------------------
# LOAD TFLITE MODEL
# -----------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="ISL AI", layout="centered")

st.title("🤟 ISL Recognition AI (Real-Time)")
st.caption("Fast • Lightweight • Accurate")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    conf = np.max(output)

    return pred, conf

# -----------------------------
# MODE SELECT
# -----------------------------
mode = st.radio("Choose Mode", ["Upload", "Camera"])

# -----------------------------
# UPLOAD MODE
# -----------------------------
if mode == "Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img)

        pred, conf = predict(img)

        st.success(f"Prediction: {classes[pred]}")
        st.progress(float(conf))

# -----------------------------
# CAMERA MODE (REAL-TIME FEEL)
# -----------------------------
elif mode == "Camera":
    camera = st.camera_input("Capture Sign")

    if camera:
        img = Image.open(camera).convert("RGB")
        st.image(img)

        pred, conf = predict(img)

        st.success(f"Prediction: {classes[pred]}")
        st.progress(float(conf))

# -----------------------------
# WORD BUILDER (BONUS 🔥)
# -----------------------------
st.markdown("---")
st.subheader("🧠 Word Builder")

if "word" not in st.session_state:
    st.session_state.word = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("Add Last Prediction"):
        if 'pred' in locals():
            st.session_state.word += classes[pred]

with col2:
    if st.button("Clear"):
        st.session_state.word = ""

st.text_input("Generated Word", st.session_state.word)
