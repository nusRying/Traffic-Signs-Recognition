"""
src/detect.py - Detection pipeline for Traffic Sign Recognition
Handles both image file and live webcam detection.
"""

import cv2
import numpy as np
from src.model import load_model, predict
from src.utils import (
    load_image_from_path,
    load_image_from_camera,
    draw_prediction,
    get_confidence_color,
)


def detect_from_image(image_path):
    """
    Run detection on a single image file.
    Displays the image with prediction overlay and prints result.
    """
    print(f"[INFO] Loading image: {image_path}")
    model = load_model()
    print("[INFO] Model loaded.")

    img_rgb = load_image_from_path(image_path)
    class_id, class_name, confidence = predict(model, img_rgb)

    print(f"\n[RESULT] Predicted: {class_name}")
    print(f"         Class ID : {class_id}")
    print(f"         Confidence: {confidence * 100:.2f}%")

    # Display
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (400, 400))
    color = get_confidence_color(confidence)
    label = f"{class_name}: {confidence * 100:.1f}%"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_bgr, (5, 5), (w + 15, h + 15), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Traffic Sign Detection", img_bgr)
    print("\nPress any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_from_camera(camera_index=0):
    """
    Run real-time detection using the webcam.
    Press 'q' or ESC to quit.
    """
    print("[INFO] Loading model...")
    model = load_model()
    print("[INFO] Model loaded. Starting camera...")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    print("[INFO] Camera opened. Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera.")
            break

        # Predict on frame
        img_rgb = load_image_from_camera(cap, frame)
        class_id, class_name, confidence = predict(model, img_rgb)

        # Overlay
        color = get_confidence_color(confidence)
        label = f"{class_name}: {confidence * 100:.1f}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (5, 5), (w + 15, h + 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Show FPS hint
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Traffic Sign Recognition - Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye!")
