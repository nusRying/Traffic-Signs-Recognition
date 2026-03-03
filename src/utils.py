"""
src/utils.py - Utility helpers for Traffic Sign Recognition
"""

import os
import numpy as np
import cv2
from PIL import Image


def load_image_from_path(image_path):
    """
    Load an image from a file path and return as a numpy array (H, W, 3) in RGB.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def load_image_from_camera(cap, frame):
    """
    Convert an OpenCV BGR frame to an RGB numpy array.
    Args:
        cap: cv2.VideoCapture (unused here, for API consistency)
        frame: numpy array BGR from cv2
    Returns:
        numpy array RGB
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_prediction(frame, class_name, confidence, x=10, y=30):
    """
    Draw prediction text on an OpenCV frame (in-place).
    """
    label = f"{class_name}: {confidence * 100:.1f}%"
    # Draw background rectangle
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x - 5, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def get_confidence_color(confidence):
    """Return a BGR color tuple based on confidence level."""
    if confidence >= 0.9:
        return (0, 200, 0)    # Green
    elif confidence >= 0.7:
        return (0, 165, 255)  # Orange
    else:
        return (0, 0, 255)    # Red
