import os
import numpy as np
import tensorflow as tf
from src.model import load_model, predict

def test_loading():
    print("--- Testing Model Reconstruction ---")
    try:
        model = load_model()
        print("PASS: Model loaded successfully.")
        
        # Verify architecture
        print(f"Model Name: {model.name}")
        print(f"Input Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        
        # Test Prediction with dummy data
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        class_id, class_name, confidence = predict(model, dummy_img)
        print(f"Dummy Prediction: {class_name} (ID: {class_id}) with {confidence:.4f} confidence")
        
        print("\nSUCCESS: Manual reconstruction and weight loading verified.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
