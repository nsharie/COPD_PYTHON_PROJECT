# =====================================
# COPD Detection - Streamlit App
# =====================================

import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="COPD Detection",
    page_icon="🫁",
    layout="centered"
)

# =====================================
# Load Model
# =====================================

@st.cache_resource
def load_model():
    if not os.path.exists('copd_model.h5'):
        return None
    return tf.keras.models.load_model('copd_model.h5')

model = load_model()

# =====================================
# App UI
# =====================================

st.title("🫁 COPD Detection")
st.subheader("Chest X-Ray Analysis using Deep Learning")
st.markdown("---")

if model is None:
    st.error("⚠️ Model file `copd_model.h5` not found!")
    st.info("Please run `train.py` first.")
    st.code("python train.py", language="bash")
    st.stop()

st.success("✅ Model loaded successfully!")
st.write("Upload a chest X-ray or CT scan image below.")

# =====================================
# Upload Image
# =====================================

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

    # =====================================
    # Preprocess Image — Force RGB always
    # =====================================

    img_resized = img.resize((224, 224))

    # Force convert to RGB no matter what mode (L, RGBA, P, etc.)
    img_rgb = img_resized.convert('RGB')

    # Convert to numpy array
    img_array = np.array(img_rgb, dtype=np.float32)

    # Confirm shape is (224, 224, 3)
    st.write(f"Debug — image shape before predict: {img_array.shape}")

    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = img_array / 255.0

    # =====================================
    # Predict
    # =====================================

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)

    confidence = float(prediction[0][0]) * 100

    # =====================================
    # Result
    # =====================================

    with col2:
        st.subheader("Analysis Result")

        if prediction[0][0] < 0.5:
            st.error("🔴 COPD Detected")
            st.metric(label="Confidence", value=f"{(1 - prediction[0][0]) * 100:.2f}%")
            st.warning("Please consult a medical professional.")
        else:
            st.success("🟢 No COPD Detected (Normal)")
            st.metric(label="Confidence", value=f"{prediction[0][0] * 100:.2f}%")
            st.info("Result appears normal. Always confirm with a doctor.")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: For educational purposes only. Not a substitute for medical advice.")