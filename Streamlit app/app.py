import io
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import streamlit as st
from PIL import Image

from src.Inference import predict_image

# Basic page config
st.set_page_config(
    page_title="Aerial Bird vs Drone Classifier",
    page_icon="🛰️",
    layout="centered"
)

st.title("🛰️ Aerial Object Classifier")
st.write(
    "Upload an aerial image and the model will classify whether it contains a **bird** or a **drone**."
)

st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# Optional threshold slider (if you want to tweak decision boundary)
threshold = st.slider(
    "Decision threshold for 'drone' (advanced, default = 0.5)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(" ")

    if st.button("🔍 Run Classification"):
        with st.spinner("Running inference..."):
            label, confidence, probs = predict_image(image)

            # Apply custom threshold logic if user changed it
            # Base model returns prob_drone
            prob_drone = probs["drone"]
            prob_bird = probs["bird"]

            if prob_drone >= threshold:
                final_label = "drone"
                final_conf = prob_drone
            else:
                final_label = "bird"
                final_conf = prob_bird

        st.markdown("### ✅ Prediction")
        st.write(f"**Class:** `{final_label.upper()}`")
        st.write(f"**Confidence:** {final_conf*100:.2f}%")

        st.markdown("### 🔎 Class Probabilities")
        st.write(
            f"- Bird:  **{prob_bird*100:.2f}%**\n"
            f"- Drone: **{prob_drone*100:.2f}%**"
        )

        st.info(
            "Note: The confidence values come from the EfficientNetB0 model "
            "fine-tuned on the aerial bird vs drone dataset."
        )
else:
    st.info("👆 Please upload an aerial image to get started.")
