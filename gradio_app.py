"""
gradio_app.py - High-Fidelity Traffic Sign Recognition Dashboard
A dark-themed, glassmorphism-inspired web interface.
"""

import gradio as gr
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from src.model import load_model, predict, CLASS_NAMES
from src.utils import load_image_from_camera
import os

# Load model early
print("[INFO] Loading model for Gradio Dashboard...")
model = load_model()
print("[INFO] Model loaded successfully.")

def classify_image(image):
    if image is None:
        return None, None
    
    # Preprocess and predict
    # Gradio provides image as numpy array (usually RGB)
    class_id, class_name, confidence = predict(model, image)
    
    # Get all predictions for Top-5 Chart
    # We need to run model.predict manually for all classes
    img_resized = tf.image.resize(image, (64, 64))
    img_norm = tf.cast(img_resized, tf.float32) / 255.0
    img_batch = tf.expand_dims(img_norm, 0)
    
    preds = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(preds)[-5:][::-1]
    
    top_labels = [CLASS_NAMES[i] for i in top_indices]
    top_confidences = [float(preds[i]) for i in top_indices]
    
    # Create Plotly Bar Chart (Top 5)
    fig = go.Figure(go.Bar(
        x=top_confidences,
        y=top_labels,
        orientation='h',
        marker=dict(
            color=top_confidences,
            colorscale='Viridis',
            line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
        )
    ))
    
    fig.update_layout(
        title="Top 5 Probabilities",
        xaxis_title="Confidence",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    res_text = f"## {class_name}\n### Confidence: {confidence*100:.2f}%"
    
    return res_text, fig

# Custom CSS for Glassmorphism & Dark Theme
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    font-family: 'Inter', sans-serif;
    color: #e94560;
}
.glass {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
}
h1 {
    color: #ffffff !important;
    text-align: center;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.sidebar {
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="rose", secondary_hue="slate")) as demo:
    gr.Markdown("# 🚦 Traffic Sign Intelligence")
    gr.Markdown("### Professional Grade Real-Time Classification Engine")
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_img = gr.Image(label="Source Image", type="numpy", elem_classes="glass")
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Classify", variant="primary")
                
            gr.Examples(
                examples=[
                    os.path.join("data", "raw", "train", "GTSRB", "Final_Training", "Images", "00014", "00000_00000.ppm"),
                    os.path.join("data", "raw", "train", "GTSRB", "Final_Training", "Images", "00013", "00000_00000.ppm")
                ],
                inputs=input_img,
                label="Sample Signs (Click to Load)"
            )
            
        with gr.Column(scale=1):
            output_text = gr.Markdown("## Prediction Results", elem_classes="glass")
            output_chart = gr.Plot(label="Confidence Distribution", elem_classes="glass")
    
    gr.Markdown("---")
    gr.Markdown("Developed with high-performance CNN architecture using TensorFlow & Keras.")
    
    submit_btn.click(fn=classify_image, inputs=input_img, outputs=[output_text, output_chart])
    input_img.change(fn=classify_image, inputs=input_img, outputs=[output_text, output_chart])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
