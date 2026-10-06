import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺")

MODEL_PATH = "best_skin_cancer_model.keras"
IMG_SIZE = (224, 224)

st.title("🩺 Skin Cancer Detection (Benign vs Malignant)")
st.caption("Educational demo only — NOT a medical diagnostic tool. Always consult a dermatologist.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found. Copy it from the Colab zip "
        "(skin_cancer_streamlit_app.zip) into the same folder as app.py."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_artifacts()

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Same preprocessing as in training: tf.image.resize -> scale to [0, 1]
    arr = tf.image.resize(np.array(image), IMG_SIZE).numpy().astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob_malignant = float(model.predict(arr, verbose=0).ravel()[0])
    pred_idx = int(prob_malignant >= 0.5)
    pred_label = class_names[pred_idx]
    confidence = prob_malignant if pred_idx == 1 else 1 - prob_malignant

    st.subheader(f"Prediction: {pred_label.upper()}")
    st.write(f"Confidence: {confidence * 100:.1f}%")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.write(f"P(benign) = {1 - prob_malignant:.3f}  |  P(malignant) = {prob_malignant:.3f}")
