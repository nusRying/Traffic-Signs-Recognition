"""
batch_predict.py - High-Performance Batch Prediction Engine
Processes entire directories with concurrent inference and exports to CSV/Excel.
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.model import load_model, predict, CLASS_NAMES

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Traffic Sign Classifier")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Number of images per batch (CPU optimization)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of worker threads")
    return parser.parse_args()

def process_image(img_path, model):
    """Worker function to process a single image."""
    try:
        # Load and preprocess using tf.io for performance
        img_raw = tf.io.read_file(img_path)
        # Use decode_image to handle various formats (ppm, jpg, png)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (64, 64))
        img = tf.cast(img, tf.float32) / 255.0
        
        # Predict
        preds = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id])
        class_name = CLASS_NAMES[class_id]
        
        return {
            "Filename": os.path.basename(img_path),
            "Predicted Class": class_name,
            "Class ID": class_id,
            "Confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"Filename": os.path.basename(img_path), "Error": str(e)}

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input directory not found: {args.input}")
        return

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"[INFO] Loading model...")
    model = load_model()
    
    # Collect all images
    image_extensions = ('.ppm', '.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_paths:
        print(f"[WARNING] No images found in {args.input}")
        return

    print(f"[INFO] Processing {len(image_paths)} images using {args.workers} workers...")
    
    results = []
    # Using ThreadPoolExecutor for concurrent I/O + Inference
    # Note: Keras predictions are thread-safe in TF2.
    with tqdm(total=len(image_paths), desc="Classifying") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # We process images individually to keep the progress bar accurate
            # but concurrent.
            futures = [executor.submit(process_image, path, model) for path in image_paths]
            for future in futures:
                results.append(future.result())
                pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Professional Exports
    csv_path = os.path.join(args.output, "batch_prediction_results.csv")
    excel_path = os.path.join(args.output, "batch_prediction_report.xlsx")
    
    df.to_csv(csv_path, index=False)
    
    # Excel with formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Add basic formatting (bold headers)
        from openpyxl.styles import Font, PatternFill
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="333333", end_color="333333", fill_type="solid")
            cell.font = Font(color="FFFFFF", bold=True)

    print(f"\n[SUCCESS] Batch processing complete!")
    print(f"          Results saved to:")
    print(f"          - CSV:   {csv_path}")
    print(f"          - Excel: {excel_path}")

if __name__ == "__main__":
    main()
