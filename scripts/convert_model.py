"""
scripts/convert_model.py - Model Optimization Engine
Converts .keras model to optimized TFLite format with INT8/Float16 quantization.
"""

import os
import tensorflow as tf
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "traffic_signs_model.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "traffic_signs_model.tflite")

def representative_data_gen():
    """Generator for providing sample data for INT8 quantization calibration."""
    # We use a small subset of the training data
    train_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "train", "GTSRB", "Final_Training", "Images")
    
    # Just grab 100 images from various classes for calibration
    count = 0
    for subdir in sorted(os.listdir(train_dir))[:10]: # First 10 classes
        class_path = os.path.join(train_dir, subdir)
        if not os.path.isdir(class_path): continue
        
        for file in os.listdir(class_path)[:10]: # 10 images per class
            if file.endswith(".ppm"):
                img_path = os.path.join(class_path, file)
                img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = img_array.astype(np.float32) / 255.0
                yield [np.expand_dims(img_array, axis=0)]
                count += 1
                if count >= 100: return

def convert_to_tflite():
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"[ERROR] Source model not found: {KERAS_MODEL_PATH}")
        return

    print(f"[INFO] Loading Keras model from {KERAS_MODEL_PATH}...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    
    # 1. Standard Conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 2. Optimization Settings (Float16)
    print("[INFO] Applying Float16 Quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"[SUCCESS] TFLite model saved to: {TFLITE_MODEL_PATH}")
    
    # Comparison
    keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"[STATS] Keras Size: {keras_size:.2f} MB")
    print(f"[STATS] TFLite Size: {tflite_size:.2f} MB ({100*tflite_size/keras_size:.1f}% of original)")

if __name__ == "__main__":
    convert_to_tflite()
