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
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traffic_signs_model_legacy.h5")


WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traffic_signs_weights_robust.npz")

def load_model():
    """
    Manually reconstruct the model architecture and load weights from NPZ.
    This bypasses persistent Keras version compatibility issues.
    """
    import keras
    from keras import layers, models
    import numpy as np

    print(f"[INFO] Reconstructing model architecture (IMG_SIZE={IMG_SIZE})...")
    
    # 1. Define Architecture
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights=None
    )
    
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    # Rescaling [-1, 1] as expected by MobileNetV2
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescaling")(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="predictions")(x)
    
    model = models.Model(inputs, outputs, name="traffic_sign_model_reconstructed")

    # 2. Load Weights from NPZ
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weight file not found at: {WEIGHTS_PATH}")

    print(f"[INFO] Loading weights from {os.path.basename(WEIGHTS_PATH)}...")
    weights_data = np.load(WEIGHTS_PATH, allow_pickle=True)
    
    # Map weights to layers
    # MobileNetV2 weights are usually prefixed with the base model name or similar
    # In our NPZ, keys look like 'Conv1/kernel', 'bn_Conv1/gamma', 'dense/kernel', etc.
    
    # We'll load base model weights first, then top layers
    model_layers = {l.name: l for l in model.layers}
    base_layers = {l.name: l for l in base_model.layers}
    
    # Helper to set weights for a layer
    def set_layer_weights(layer, prefix):
        layer_weights = []
        # Keras weights order: kernels, biases (then gamma, beta, mean, var for BN)
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D, layers.Dense)):
            if f"{prefix}/kernel" in weights_data:
                layer_weights.append(weights_data[f"{prefix}/kernel"])
            if f"{prefix}/bias" in weights_data:
                layer_weights.append(weights_data[f"{prefix}/bias"])
        elif isinstance(layer, layers.BatchNormalization):
            for suffix in ["gamma", "beta", "moving_mean", "moving_variance"]:
                if f"{prefix}/{suffix}" in weights_data:
                    layer_weights.append(weights_data[f"{prefix}/{suffix}"])
        
        if layer_weights:
            try:
                layer.set_weights(layer_weights)
            except Exception as e:
                print(f"[WARN] Could not set weights for {layer.name}: {e}")

    # Load base model weights
    for name, layer in base_layers.items():
        set_layer_weights(layer, name)
        
    # Load top layers
    for name in ["dense", "predictions"]:
        if name in model_layers:
            set_layer_weights(model_layers[name], name)

    print("[INFO] Model reconstructed and weights loaded successfully.")
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
    # Preprocessing using TensorFlow for Keras 2 compatibility
    img = tf.image.resize(image_array, IMG_SIZE)
    img = tf.cast(img, dtype=tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img, verbose=0)
    class_id = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_id])
    class_name = CLASS_NAMES[class_id]
    return class_id, class_name, confidence
