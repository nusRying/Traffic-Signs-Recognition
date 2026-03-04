# 🚦 Traffic Sign Intelligence

# 🚦 SignIntel: Real-Time Traffic Sign Intelligence

A professional-grade Computer Vision system for real-time traffic sign recognition using a custom-reconstructed MobileNetV2 architecture.

![Dashboard Interface](https://cdn-icons-png.flaticon.com/512/3426/3426033.png)

## 🛰️ Intelligence Control Center (v2.0)

The project features a high-fidelity Streamlit dashboard with a dual-mode detection engine:

- **📷 LIVE RADAR**: Real-time traffic sign capture and analysis via laptop camera sensors.
- **📂 FILE INTAKE**: High-resolution telemetry analysis for uploaded image files.
- **📊 NEURAL LOGITS**: Interactive Top-5 probability distribution visualization via Plotly.
- **🧠 DETECT HISTORY**: Session-based history tracking for recent classifications.

## 🛠️ Neural Architecture

The system uses a custom-reconstructed MobileNetV2 backbone to bypass framework versioning conflicts (Keras 2/3 compatibility).

- **Base Model**: MobileNetV2 (Pre-trained on ImageNet, fine-tuned).
- **Custom Top**: GlobalAveragePooling2D → Dropout (0.2) → Dense (256, ReLU) → Dense (43, Softmax).
- **Weights Logic**: Precision-mapped loading from low-level NumPy arrays (`.npz`).
- **Preprocessing**: Optimized TensorFlow operations for sub-second inference latency.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Conda (Recommended)

### Installation

```powershell
conda activate revival
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4. High-Performance Batch Processing

Process an entire folder and export results:

```bash
python batch_predict.py --input path/to/images --output outputs
```

---

## 🛠️ Project Structure

- `app.py`: CLI entry point (Webcam/Image modes).
- `gradio_app.py`: High-fidelity web interface.
- `batch_predict.py`: Concurrent batch processing engine.
- `src/`: Core logic (model loading, detection, utils).
- `models/`: Contains both Keras and optimized TFLite models.
- `scripts/`: Optimization and utility scripts.

---

## 📊 Model Optimization

We optimized the original 28MB Keras model using **Float16 Quantization**, resulting in a **4.9MB TFLite** model (~82% reduction) without significant accuracy loss, making it ideal for edge deployment.

---

## 📝 License

This project is open-source and available under the MIT License.
