#!/usr/bin/env python3
"""
app.py - Main entry point for Traffic Sign Recognition System

Usage:
    python app.py                        # Live webcam mode
    python app.py --image path/to/img   # Single image mode
    python app.py --camera 0            # Specify camera index (default: 0)
"""

import argparse
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                        Run with webcam (default)
  python app.py --image sign.jpg       Classify a single image
  python app.py --camera 1             Use second camera
        """
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to an image file to classify"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera index for live detection (default: 0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure the src directory is on the path
    sys.path.insert(0, os.path.dirname(__file__))

    try:
        from src.detect import detect_from_image, detect_from_camera
    except ImportError as e:
        print(f"[ERROR] Could not import detection module: {e}")
        sys.exit(1)

    if args.image:
        # Single image classification mode
        if not os.path.exists(args.image):
            print(f"[ERROR] Image file not found: {args.image}")
            sys.exit(1)
        detect_from_image(args.image)
    else:
        # Live webcam mode
        print("[INFO] Starting real-time webcam detection...")
        print("       (Use --image <path> to classify a single image file)")
        try:
            detect_from_camera(camera_index=args.camera)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            print("[TIP]  Try: python app.py --image path/to/your/sign.jpg")
            sys.exit(1)


if __name__ == "__main__":
    main()
