import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.model import load_model, predict, CLASS_NAMES
import os
from PIL import Image
import datetime

# Page Configuration
st.set_page_config(
    page_title="SignIntel | High-Perf Traffic Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED UI STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');
    
    :root {
        --primary: #e94560;
        --bg-gradient: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        --glass: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: var(--bg-gradient);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Containers */
    div.stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    
    div.stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: var(--glass);
        border-radius: 10px 10px 0 0;
        border: 1px solid var(--glass-border);
        color: #aaa;
    }

    div.stTabs [aria-selected="true"] {
        background-color: rgba(233, 69, 96, 0.15) !important;
        border-bottom: 2px solid var(--primary) !important;
        color: white !important;
    }

    /* Detection Card */
    .detection-card {
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        border-left: 5px solid var(--primary);
    }

    .metric-text {
        font-family: 'JetBrains Mono', monospace;
        color: var(--primary);
        font-size: 1.8rem;
        font-weight: 800;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- COMPONENT LOGIC ---

@st.cache_resource
def get_model():
    with st.spinner("Initializing Neural Core..."):
        return load_model()

model = get_model()

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(name, conf):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.history.insert(0, {"name": name, "conf": conf, "time": timestamp})
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3426/3426033.png", width=80)
    st.title("SignIntel v2.0")
    st.markdown("---")
    st.subheader("📊 Recent Detections")
    
    if not st.session_state.history:
        st.info("No detections yet.")
    else:
        for item in st.session_state.history:
            st.markdown(f"**{item['name']}**  \n`{item['conf']:.1%}` | *{item['time']}*")
            st.markdown("---")

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# --- MAIN UI ---
st.title("🛰️ Intelligence Control Center")
st.markdown("### Real-time High-Fidelity Traffic Sign Recognition Dashboard")

tab1, tab2 = st.tabs(["[ 📷 LIVE RADAR ]", "[ 📂 FILE INTAKE ]"])

def handle_classification(img_array):
    class_id, class_name, confidence = predict(model, img_array)
    add_to_history(class_name, confidence)
    
    # UI Presentation
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        <div class="detection-card">
            <h4 style="margin:0; opacity:0.7;">IDENTIFIED OBJECT</h4>
            <h2 style="margin:5px 0; color:white;">{class_name}</h2>
            <div class="metric-text">{confidence:.2%} CONFIDENCE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        # Top-5 Chart
        import tensorflow as tf
        img_processed = tf.image.resize(img_array, (64, 64))
        img_processed = tf.cast(img_processed, dtype=tf.float32) / 255.0
        img_processed = tf.expand_dims(img_processed, axis=0)
        
        preds = model.predict(img_processed, verbose=0)[0]
        top_indices = np.argsort(preds)[-5:][::-1]
        
        top_labels = [CLASS_NAMES[i] for i in top_indices]
        top_confidences = [float(preds[i]) for i in top_indices]
        
        fig = go.Figure(go.Bar(
            x=top_confidences,
            y=top_labels,
            orientation='h',
            marker=dict(color='#e94560')
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white"),
            xaxis=dict(showgrid=False, range=[0, 1]),
            yaxis=dict(autorange="reversed"),
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            title="NEURAL LOGITS (TOP 5)"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab1:
    st.markdown("#### Initialize Camera Sensor")
    cam_input = st.camera_input("Scanner Active", help="Capture a photo for instant neural analysis")
    
    if cam_input:
        img = Image.open(cam_input)
        img_array = np.array(img.convert('RGB'))
        handle_classification(img_array)

with tab2:
    st.markdown("#### Upload High-Res Data")
    uploaded_file = st.file_uploader("Drop telemetry data here...", type=["jpg", "png", "ppm", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=400)
        if st.button("EXECUTE ANALYSIS"):
            img_array = np.array(img.convert('RGB'))
            handle_classification(img_array)
    else:
        st.info("Awaiting file input...")

st.markdown("---")
st.caption("Engine: MobileNetV2 Reconstructed | Environment: Revival (Keras 2.10) | UI: Streamlit HUD")
