import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# For real-time video
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# -----------------------------
# LOAD MODEL (.h5 UPDATED)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.h5", compile=False)

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
st.write("Upload image, capture photo, or use live video")

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

    return predicted_class, confidence, prediction

# -----------------------------
# SIDEBAR OPTION
# -----------------------------
option = st.sidebar.radio(
    "Choose Input Method",
    ["Upload Image", "Camera Capture", "Live Video"]
)

# =========================================================
# 1. IMAGE UPLOAD
# =========================================================
if option == "Upload Image":
    st.header("📁 Upload Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        pred_class, conf, pred = predict_image(image)

        st.success(f"Predicted: {classes[pred_class].upper()}")
        st.info(f"Confidence: {conf*100:.2f}%")

# =========================================================
# 2. CAMERA CAPTURE
# =========================================================
elif option == "Camera Capture":
    st.header("📷 Capture from Camera")

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        pred_class, conf, pred = predict_image(image)

        st.success(f"Predicted: {classes[pred_class].upper()}")
        st.info(f"Confidence: {conf*100:.2f}%")

# =========================================================
# 3. LIVE VIDEO (REAL-TIME)
# =========================================================
elif option == "Live Video":
    st.header("🎥 Real-Time Detection")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            img_resized = cv2.resize(img, (224, 224))
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)

            label = f"{classes[pred_class].upper()} ({confidence*100:.1f}%)"

            cv2.putText(
                img,
                label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            return img

    webrtc_streamer(
        key="isl-live",
        video_transformer_factory=VideoTransformer
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with TensorFlow + Streamlit 🚀")