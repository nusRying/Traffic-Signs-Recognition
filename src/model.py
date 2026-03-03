import os
import numpy as np
import tensorflow as tf
import keras

# 43 GTSRB class names in order (class ID 0–42)
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

IMG_SIZE = (64, 64)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traffic_signs_model.keras")


def load_model():
    """Load the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    return model


def predict(model, image_array):
    """
    Run inference on a preprocessed image array.
    Args:
        model: Loaded Keras model
        image_array: numpy array of shape (H, W, 3), values 0-255
    Returns:
        (class_id, class_name, confidence)
    """
    # Resize and normalize
    img = tf.image.resize(image_array, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, 0)  # Add batch dimension

    predictions = model.predict(img, verbose=0)
    class_id = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_id])
    class_name = CLASS_NAMES[class_id]
    return class_id, class_name, confidence
