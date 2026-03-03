# 🚦 Traffic Sign Intelligence

A professional-grade Traffic Sign Recognition system using Deep Learning (CNN) with 43-class classification, hardware-optimized inference, and a premium web dashboard.

---

## ✨ Features

- **🎯 High Accuracy**: Trained on the GTSRB dataset, achieving >92% accuracy across 43 traffic sign categories.
- **💎 Premium Dashboard**: High-fidelity Gradio web interface with **Glassmorphism** aesthetics and real-time Plotly confidence charts.
- **⚡ Edge Optimized**: Quantized **TFLite** model (~5MB) for fast inference on mobile and Raspberry Pi devices.
- **📊 Batch Processor**: Multi-threaded engine for processing large image directories with professional Excel/CSV reports.
- **🎥 Live Detection**: Seamless real-time recognition via webcam.

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
pip install gradio plotly pandas openpyxl tqdm
```

### 2. Launch the Premium Dashboard

Experience the high-fidelity web interface:

```bash
python gradio_app.py
```

### 3. Run Real-Time Webcam Detection

```bash
python app.py
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
