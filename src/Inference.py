from pathlib import Path
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.efficientnet import preprocess_input


# Path to your trained Lite model
MODEL_PATH = Path(r"C:\Users\kisho\Downloads\Aeiral object detection project\Models\efficientnetb0_savedmodel.tflite")


# Load the TFLite model
interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Class names (must match training order)
CLASS_NAMES = ["bird", "drone"]


def prepare_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resize, format and preprocess for EfficientNet."""
    image = image.convert("RGB")
    image = image.resize(target_size)

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array


def predict_image(image: Image.Image):
    """Run inference on an image and return classification + confidence."""
    
    processed = prepare_image(image)

    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0][0]

    prob_drone = float(output)
    prob_bird = 1 - prob_drone

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
