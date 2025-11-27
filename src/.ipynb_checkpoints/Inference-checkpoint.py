from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Path to your trained model
MODEL_PATH = Path(r"C:\Users\kisho\Downloads\Aeiral object detection project\Models\efficientnetb0_savedmodel.h5")

# Load model once (at import time)
model = tf.keras.models.load_model(MODEL_PATH)

# Class names – IMPORTANT: must match training order
# In flow_from_directory, class_indices are alphabetical, so:
# {'bird': 0, 'drone': 1}
CLASS_NAMES = ["bird", "drone"]


def prepare_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Takes a PIL image, resizes it, converts to numpy batch, 
    and applies EfficientNet preprocessing.
    """
    # Ensure RGB
    image = image.convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    # Use same preprocessing as during training
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(image: Image.Image):
    """
    Runs prediction on a PIL image.
    
    Returns:
        predicted_label (str)
        confidence (float 0–1 for predicted class)
        probs_dict (dict: class_name -> probability)
    """
    processed = prepare_image(image)
    prob = model.predict(processed)[0][0]  # scalar between 0 and 1

    # Since we used a single sigmoid neuron:
    # prob ≈ P(drone), 1 - prob ≈ P(bird)
    prob_drone = float(prob)
    prob_bird = float(1.0 - prob_drone)

    # Decide class based on threshold 0.5
    if prob_drone >= 0.5:
        predicted_label = "drone"
        confidence = prob_drone
    else:
        predicted_label = "bird"
        confidence = prob_bird

    probs_dict = {
        "bird": prob_bird,
        "drone": prob_drone
    }

    return predicted_label, confidence, probs_dict
